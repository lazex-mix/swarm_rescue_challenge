"""
This program can be launched directly.
To move the drone, you have to click on the map, then use the arrows on the
keyboard
"""

import sys, gc
from pathlib import Path
from typing import List, Type
from enum import Enum
import cv2
import numpy as np

# Insert the parent directory of the current file's directory into sys.path.
# This allows Python to locate modules that are one level above the current
# script, in this case spg_overlay.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.utils.constants import MAX_RANGE_LIDAR_SENSOR
from swarm_rescue.simulation.utils.grid import Grid
from swarm_rescue.simulation.utils.pose import Pose
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
from swarm_rescue.simulation.elements.rescue_center import RescueCenter
from swarm_rescue.simulation.elements.return_area import ReturnArea
from swarm_rescue.simulation.elements.wounded_person import WoundedPerson
from swarm_rescue.simulation.gui_map.closed_playground import ClosedPlayground
from swarm_rescue.maps.map_intermediate_01 import MapIntermediate01
from swarm_rescue.simulation.reporting.evaluation import ZonesConfig, EvalPlan, EvalConfig
from swarm_rescue.simulation.elements.sensor_disablers import ZoneType, NoGpsZone
from swarm_rescue.simulation.gui_map.gui_sr import GuiSR
from swarm_rescue.simulation.gui_map.map_abstract import MapAbstract
from swarm_rescue.simulation.utils.misc_data import MiscData


from swarm_rescue.team_du_sud.geometry_opti import *
from swarm_rescue.team_du_sud.walls_keyboard import add_walls


class StockageMesh:
    EMPTY = 0
    RETURN_AREA = 1
    RESCUE_CENTER = 2
    WOUNDED = 3
    WALL = 4

    # Priorité des valeurs pour l'écrasement (pas encore fonctionnel)
    PRIORITY = {
        EMPTY: 3,
        RETURN_AREA: 5,
        RESCUE_CENTER: 10,
        WOUNDED: 10,
        WALL: 3
    }

    def __init__(self, cell_size=5, initial_size=5):
        self.cell_size = float(cell_size)
        self.mesh = None
        self.origin = None
        self.offset = np.array([0, 0], dtype=int)
        self.initial_size = int(initial_size)
        self.wounded_positions = {}
        self.rescue_center_positions = {}

    def initialize_mesh(self, drone_pos):
        self.origin = np.array(drone_pos, dtype=float)
        s = max(3, self.initial_size)
        self.mesh = np.zeros((s, s), dtype=float)
        self.offset = np.array([s // 2, s // 2], dtype=int)

    # Convertir des coordonnées réelles en indices de cellule
    def coord_to_cell(self, pos):
        delta = (np.array(pos, dtype=float) - self.origin) / self.cell_size
        i = int(np.floor(delta[1])) + int(self.offset[0])
        j = int(np.floor(delta[0])) + int(self.offset[1])
        return i, j
    
    # Étendre la grille si nécessaire pour inclure une cellule donnée et adapter les positions
    def extend_if_needed(self, i, j):
        h, w = self.mesh.shape
        top = max(0, -i)
        left = max(0, -j)
        bottom = max(0, i - (h - 1))
        right = max(0, j - (w - 1))
        if top == left == bottom == right == 0:
            return
        new_h = top + h + bottom
        new_w = left + w + right
        new_mesh = np.zeros((new_h, new_w), dtype=float)
        new_mesh[top:top+h, left:left+w] = self.mesh
        self.mesh = new_mesh
        self.offset += np.array([top, left], dtype=int)
        for wid in self.wounded_positions:
            old_i, old_j = self.wounded_positions[wid]
            self.wounded_positions[wid] = (old_i + top, old_j + left)
        for cid in self.rescue_center_positions:
            old_i, old_j = self.rescue_center_positions[cid]
            self.rescue_center_positions[cid] = (old_i + top, old_j + left)

    # Définir la valeur d'une cellule
    def set_cell(self, pos, value):
        if self.mesh is None:
            self.initialize_mesh(pos)
        i, j = self.coord_to_cell(pos)
        self.extend_if_needed(i, j)
        i, j = self.coord_to_cell(pos)
        current = int(self.mesh[i, j])
        if value == self.EMPTY and current in [self.WALL, self.WOUNDED, self.RESCUE_CENTER]:
            self.mesh[i, j] = self.EMPTY
            return
        if self.PRIORITY[value] >= self.PRIORITY.get(current, 0):
            self.mesh[i, j] = value

    # Marquer une personne blessée avec un W
    def mark_wounded(self, pos, wid=0):
        i, j = self.coord_to_cell(pos)
        self.extend_if_needed(i, j)
        i, j = self.coord_to_cell(pos)
        if wid in self.wounded_positions:
            old_i, old_j = self.wounded_positions[wid]
            if (old_i, old_j) != (i, j) and self.mesh[old_i, old_j] == self.WOUNDED:
                self.mesh[old_i, old_j] = self.EMPTY
        self.mesh[i, j] = self.WOUNDED
        self.wounded_positions[wid] = (i, j)

    # Marquer le centre de secours avec un C
    def mark_rescue_center(self, pos, cid=0):
        i, j = self.coord_to_cell(pos)
        self.extend_if_needed(i, j)
        i, j = self.coord_to_cell(pos)
        '''if cid in self.rescue_center_positions:
            old_i, old_j = self.rescue_center_positions[cid]
            if (old_i, old_j) != (i, j) and self.mesh[old_i, old_j] == self.RESCUE_CENTER:
                self.mesh[old_i, old_j] = self.EMPTY'''
        self.mesh[i, j] = self.RESCUE_CENTER
        self.rescue_center_positions[cid] = (i, j)

    def mark_empty(self, pos):
        self.set_cell(pos, self.EMPTY)

    # Marquer les murs avec un #
    def mark_wall_between(self, p1, p2):
        p0 = np.array(p1, dtype=float)
        p1 = np.array(p2, dtype=float)
        dist = np.linalg.norm(p1 - p0)
        if dist == 0:
            self.set_cell(p0, self.WALL)
            return
        steps = max(2, int(np.ceil(dist / (self.cell_size * 0.5))))
        for t in np.linspace(0, 1, steps):
            pt = p0 * (1 - t) + p1 * t
            self.set_cell(pt, self.WALL)

    # Mise à jour dynamique d'une cellule d'un mur
    def update_wall_cell(self, pos, is_wall):
        if is_wall:
            self.set_cell(pos, self.WALL)
        else:
            # si pas de mur, redevenir EMPTY (fonctionne pas encore très bien)
            i, j = self.coord_to_cell(pos)
            self.extend_if_needed(i, j)
            i, j = self.coord_to_cell(pos)
            if self.mesh[i, j] == self.WALL:
                self.mesh[i, j] = self.EMPTY

    # Marquer la return area avec un R
    def mark_return_area(self, pos):
        self.set_cell(pos, self.RETURN_AREA)

    # Afficher la grille dynamiquement avec des symboles dans le terminal (bouger le drone en meme temps pour maj la grille)
    def print_mesh(self, symbols=None):
        if self.mesh is None:
            return
        if symbols is None:
            symbols = {self.EMPTY: ".", self.RETURN_AREA: "R", self.RESCUE_CENTER: "C",
                       self.WOUNDED: "W", self.WALL: "#"}
        for row in self.mesh[::-1]:
            print("".join(symbols.get(int(x), "?") for x in row))

    # Afficher la grille statique à partir d'une matrice donnée
    def print_mesh_from_matrix(self, matrix, symbols=None):
        if symbols is None:
            symbols = {
                int(self.EMPTY): ".",
                int(self.RETURN_AREA): "R",
                int(self.RESCUE_CENTER): "C",
                int(self.WOUNDED): "W",
                int(self.WALL): "#"
            }
        for row in matrix[::-1]:
            print("".join(symbols.get(int(x), "?") for x in row))

class MyDrone(DroneAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.iteration = 0
        self.data_mesh = StockageMesh(cell_size=30, initial_size=7)
        self.path = Path()
        self.previous_HP = self.drone_health
        # Rajouter un système permettant de déterminer si le drone est en exploration ou en sauvetage (cf test_keyboard.py)

    def define_message_for_all(self):
        pass

    def control(self):
        command: CommandsDict = {"forward": 0.0,
                                 "lateral": 0.0,
                                 "rotation": 0.0,
                                 "grasper": 0}
        
        # Mettre à jour la position et l'orientation du drone
        gps = np.asarray(self.measured_gps_position())
        orientation = self.measured_compass_angle()
        self.pose = Pose(gps, orientation)

        # Initialisation de la grille si nécessaire
        if self.data_mesh.mesh is None:
            self.data_mesh.initialize_mesh(self.pose.position)

        # Marquer la return area si on y est
        if self.is_inside_return_area:
            self.data_mesh.mark_return_area(self.pose.position)

        if self.drone_health == self.previous_HP and not self.is_inside_return_area:
            self.data_mesh.mark_empty(self.pose.position)

        # Traiter les données des capteurs lidar et sémantiques et update le mesh
        self.process_lidar_semantic_sensors()

        # print le mesh toutes les 20 itérations (pour pas flood le terminal et pour controler les timestep)
        if self.iteration % 20 == 0:
            print("Mesh actuel")
            # On imprime une copie pour éviter les modifications pendant l'affichage du mesh (calculs en moins)
            snapshot = self.data_mesh.mesh.copy()
            self.data_mesh.print_mesh_from_matrix(snapshot)

        self.iteration += 1

        self.previous_HP = self.drone_health
        return command # Non définie pour l'instant

    def process_lidar_semantic_sensors(self):
        # Récupérer les valeurs du lidar et les zones d'intérêts (cf geometry_opti.py)
        lidar_vals = self.lidar_values()
        lidar_angles = self.lidar_rays_angles()
        zones = detect_local_zones(lidar_vals)

        # Récupérer les valeurs sémantiques et itérer dessus pour marquer les entités détectées
        semantic = self.semantic_values()

        if semantic is not None:
            for s in semantic:
                # Calculer la position de l'entité détectée et la marquer dans la grille
                ang = self.pose.orientation + s.angle
                pos = self.pose.position + np.array([np.cos(ang), np.sin(ang)]) * s.distance
                if s.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    self.data_mesh.mark_rescue_center(pos)
                elif s.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:
                    self.data_mesh.mark_wounded(pos)

        # Itérer sur les zones détectées pour marquer les murs dans la grille
        if zones:
            for zone in zones:
                if len(zone) < 2:
                    continue
                idx_start = zone[0]
                idx_end = zone[-1]
                ang0 = self.pose.orientation + lidar_angles[idx_start]
                ang1 = self.pose.orientation + lidar_angles[idx_end]
                p0 = self.pose.position + np.array([np.cos(ang0), np.sin(ang0)]) * lidar_vals[idx_start]
                p1 = self.pose.position + np.array([np.cos(ang1), np.sin(ang1)]) * lidar_vals[idx_end]
                self.data_mesh.mark_wall_between(p0, p1)

    # Appels des fonctions de marquage du mesh (à optimiser si besoin)
    '''
    def mark_rescue_center(self, world_pos):
        self.data_mesh.mark_rescue_center(world_pos)

    def mark_wounded(self, world_pos):
        self.data_mesh.mark_wounded(world_pos)

    def mark_wall(self, p1, p2):
        self.data_mesh.mark_wall_between(p1, p2)
    '''


def print_keyboard_man():
    print("How to use the keyboard to direct the drone?")
    print("\t- up / down key : forward and backward")
    print("\t- left / right key : turn left / right")
    print("\t- shift + left/right key : left/right lateral movement")
    print("\t- W key : grasp wounded person")
    print("\t- L key : display (or not) the lidar sensor")
    print("\t- S key : display (or not) the semantic sensor")
    print("\t- P key : draw position from GPS sensor")
    print("\t- C key : draw communication between drones")
    print("\t- M key : print messages between drones")
    print("\t- Q key : exit the program")
    print("\t- R key : reset")


def main():
    print_keyboard_man()

    eval_plan = EvalPlan()

    zones_config: ZonesConfig = ()
    eval_config = EvalConfig(map_name="MapIntermediate01", zones_config=zones_config, nb_rounds=2)
    eval_plan.add(eval_config=eval_config)

    gc.collect()

    # Retrieve the class object from the global namespace using its name
    map_class = globals().get(eval_config.map_name)
    # Instantiate the map class with the provided zones configuration
    the_map = map_class(drone_type=MyDrone, zones_config=eval_config.zones_config)

    gui = GuiSR(the_map=the_map,
                use_mouse_measure=True,
                use_keyboard=True
                )
    gui.run()

    score_health_returned = the_map.compute_score_health_returned()
    print("score_health_returned = ", score_health_returned)


if __name__ == '__main__':
    main()