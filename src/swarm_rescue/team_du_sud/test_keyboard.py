"""
This program can be launched directly.
To move the drone, you have to click on the map, then use the arrows on the
keyboard
"""

import sys
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
from swarm_rescue.simulation.gui_map.gui_sr import GuiSR
from swarm_rescue.maps.map_intermediate_01 import MapIntermediate01
from swarm_rescue.simulation.gui_map.map_abstract import MapAbstract
from swarm_rescue.simulation.utils.misc_data import MiscData


from swarm_rescue.team_du_sud.geometry import Point, Line, Box, build_box_with_line_and_point, detect_local_zones, build_box_with_2_opposite_points
from swarm_rescue.team_du_sud.walls_keyboard import add_walls

class OccupancyGrid(Grid):
    """Simple occupancy grid"""

    def __init__(self,
                 size_area_world,
                 resolution: float,
                 lidar):
        super().__init__(size_area_world=size_area_world,
                         resolution=resolution)

        self.size_area_world = size_area_world
        self.resolution = resolution

        self.lidar = lidar

        self.x_max_grid: int = int(self.size_area_world[0] / self.resolution
                                   + 0.5)
        self.y_max_grid: int = int(self.size_area_world[1] / self.resolution
                                   + 0.5)

        self.grid = np.zeros((self.x_max_grid, self.y_max_grid))
        self.zoomed_grid = np.empty((self.x_max_grid, self.y_max_grid))

    def update_grid(self, pose: Pose):
        """
        Bayesian map update with new observation
        lidar : lidar data
        pose : corrected pose in world coordinates
        """
        EVERY_N = 3
        LIDAR_DIST_CLIP = 40.0
        EMPTY_ZONE_VALUE = -0.602
        OBSTACLE_ZONE_VALUE = 2.0
        FREE_ZONE_VALUE = -4.0
        THRESHOLD_MIN = -40
        THRESHOLD_MAX = 40

        lidar_dist = self.lidar.get_sensor_values()[::EVERY_N].copy()
        lidar_angles = self.lidar.ray_angles[::EVERY_N].copy()

        # Compute cos and sin of the absolute angle of the lidar
        cos_rays = np.cos(lidar_angles + pose.orientation)
        sin_rays = np.sin(lidar_angles + pose.orientation)

        max_range = MAX_RANGE_LIDAR_SENSOR * 0.9

        # For empty zones
        # points_x and point_y contains the border of detected empty zone
        # We use a value a little bit less than LIDAR_DIST_CLIP because of the
        # noise in lidar
        lidar_dist_empty = np.maximum(lidar_dist - LIDAR_DIST_CLIP, 0.0)
        # All values of lidar_dist_empty_clip are now <= max_range
        lidar_dist_empty_clip = np.minimum(lidar_dist_empty, max_range)
        points_x = pose.position[0] + np.multiply(lidar_dist_empty_clip,
                                                  cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist_empty_clip,
                                                  sin_rays)

        for pt_x, pt_y in zip(points_x, points_y):
            self.add_value_along_line(pose.position[0], pose.position[1],
                                      pt_x, pt_y,
                                      EMPTY_ZONE_VALUE)

        # For obstacle zones, all values of lidar_dist are < max_range
        select_collision = lidar_dist < max_range

        points_x = pose.position[0] + np.multiply(lidar_dist, cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist, sin_rays)

        points_x = points_x[select_collision]
        points_y = points_y[select_collision]

        self.add_points(points_x, points_y, OBSTACLE_ZONE_VALUE)

        # the current position of the drone is free !
        self.add_points(pose.position[0], pose.position[1], FREE_ZONE_VALUE)

        # threshold values
        self.grid = np.clip(self.grid, THRESHOLD_MIN, THRESHOLD_MAX)

        # compute zoomed grid for displaying
        self.zoomed_grid = self.grid.copy()
        new_zoomed_size = (int(self.size_area_world[1] * 0.5),
                           int(self.size_area_world[0] * 0.5))
        self.zoomed_grid = cv2.resize(self.zoomed_grid, new_zoomed_size,
                                      interpolation=cv2.INTER_NEAREST)

# Classe où le drône stocke les données importantes de la map
class Stockage():
    def __init__(self):
        self.drone_start_position = None
        self.rescue_center = None
        self.return_area = None
        self.wounded_position = None
        self.walls_positions = None
        self.NoGPS_position = None

    # RETURN AREA
    def initialize_return_area_position(self, actual_drone_position):
        self.actual_drone_position = Point(actual_drone_position[0], actual_drone_position[1]) 
        if self.drone_start_position.x == self.actual_drone_position.x:
            line = Line(self.actual_drone_position, self.drone_start_position)
            line.update_type("vertical")
            self.return_area = line
        elif self.drone_start_position.y == self.actual_drone_position.y:
            line = Line(self.actual_drone_position, self.drone_start_position)
            line.update_type("horizontal")
            self.return_area = line
        else:
            box = build_box_with_2_opposite_points(self.drone_start_position, self.actual_drone_position)
            self.return_area = box

    def update_return_area_position(self, actual_drone_position):
        self.actual_drone_position = Point(actual_drone_position[0], actual_drone_position[1])
        if self.return_area is not None:
            if isinstance(self.return_area, Line):
                if self.return_area.is_on_line(self.actual_drone_position):
                    pass
                elif self.return_area.is_aligned(self.actual_drone_position):
                    self.return_area.extend_line(self.actual_drone_position)
                else:
                    new_box = build_box_with_line_and_point(self.return_area, self.actual_drone_position)
                    self.return_area = new_box
            elif isinstance(self.return_area, Box):
                if self.return_area.is_inside(self.actual_drone_position):
                    pass
                else:
                    self.return_area.extend_box(self.actual_drone_position)
        else:
            self.initialize_return_area_position(actual_drone_position)

    # RESCUE CENTER    
    def update_rescue_center_position(self, new_point : Point):
        if self.rescue_center is None:
            self.rescue_center = new_point

        elif isinstance(self.rescue_center, Point):
            if new_point == self.rescue_center :
                pass
            else:
                if new_point.x == self.rescue_center.x:
                    self.rescue_center = Line(self.rescue_center, new_point)
                    self.rescue_center.update_type("vertical")
                elif new_point.y == self.rescue_center.y:
                    self.rescue_center = Line(self.rescue_center, new_point)
                    self.rescue_center.update_type("horizontal")
                else:
                    new_box = build_box_with_2_opposite_points(self.rescue_center, new_point)
                    self.rescue_center = new_box

        elif isinstance(self.rescue_center, Line):
            if self.rescue_center.is_on_line(new_point):
                pass
            elif self.rescue_center.is_aligned(new_point):
                self.rescue_center.extend_line(new_point)
            else:
                new_box = build_box_with_line_and_point(self.rescue_center, new_point)
                self.rescue_center = new_box
    
        elif isinstance(self.rescue_center, Box):
            self.rescue_center.extend_box(new_point)

    # WOUNDED
    def update_wounded_position(self, new_point: Point):
        if self.wounded_position is None:
            self.wounded_position = [new_point]
        else:
            for wounded in self.wounded_position:
                if new_point.distance_to(wounded) < 60:
                # Remplace l'ancienne position par la nouvelle plus précise
                    self.wounded_position.remove(wounded)
                    self.wounded_position.append(new_point)
                    break
            else:
                # Aucun blessé existant à moins de 30 pixels → c'est un nouveau
                if new_point not in self.wounded_position:
                    self.wounded_position.append(new_point)

    # WALLS
    def update_walls_positions(self, new_point_1:Point, new_point_2:Point, middle_point:Point):
        aligned_type = middle_point.are_aligned_in_type(new_point_1, new_point_2)
        if aligned_type[0] == "undefined":
            pass
        else:
            if aligned_type[0] == "vertical":
                max_height = max(abs(new_point_1.y), abs(new_point_2.y))
                point_1 = Point(middle_point.x-3, max_height)
                point_2 = Point(middle_point.x+3, max_height)
                point_3 = Point(middle_point.x+3, -max_height)
                point_4 = Point(middle_point.x-3, -max_height)
                new_wall = Box(point_1, point_2, point_3, point_4)
            elif aligned_type[0] == "horizontal":
                max_length = max(abs(new_point_1.x), abs(new_point_2.x))
                point_1 = Point(max_length, middle_point.y-3)
                point_2 = Point(max_length, middle_point.y+3)
                point_3 = Point(-max_length, middle_point.y+3)
                point_4 = Point(-max_length, middle_point.y-3)
                new_wall = Box(point_1, point_2, point_3, point_4)

            if self.walls_positions is None:
                self.walls_positions = [new_wall]
            else:
                self.walls_positions.append(new_wall)

    # NOGPS ZONE
    def update_NoGPS_position(self):
        pass

# Classe qui construit des chemins à suivre selon les données captées
class Path():
    def _init_(self):
        self.strategic_points = []
        self.return_area_postion = None
        self.rescue_center_position = None

class MyDroneTest(DroneAbstract):
    class Activity(Enum):
        """
        The drone is either exploring the map or rescuing wounded persons.
        """
        EXPLORING = 1
        RESCUING = 2

    class Rescuing_Activities(Enum):
        """
        All the states of the drone during the rescue of wounded persons.
        """
        GRASPING_WOUNDED = 3
        DROPPING_AT_RESCUE_CENTER = 4


    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.iteration: int = 0
        self.estimated_pose = Pose()
        resolution = 8
        self.grid = OccupancyGrid(size_area_world=self.size_area,
                                  resolution=resolution,
                                  lidar=self.lidar())

        self.last_inside = False
        self.data = Stockage()
        self.path = Path()
        self.state = self.Activity.EXPLORING

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    def control(self):
        command: CommandsDict = {"forward": 0.0,
                                 "lateral": 0.0,
                                 "rotation": 0.0,
                                 "grasper": 0}
    
        self.iteration += 1

        # UPDATING THE OCCUPANCY GRID
        self.estimated_pose = Pose(np.asarray(self.measured_gps_position()),
                                   self.measured_compass_angle())

        self.grid.update_grid(pose=self.estimated_pose)
        if self.iteration % 5 == 0:
            self.grid.display(self.grid.grid,
                              self.estimated_pose,
                              title="occupancy grid")
            self.grid.display(self.grid.zoomed_grid,
                              self.estimated_pose,
                              title="zoomed occupancy grid")
    
        # EXPLORING STATE
        if self.state == self.Activity.EXPLORING:

            # Updates the return area position if the drone enters it or leaves it
            if self.data.drone_start_position is None and self.is_inside_return_area == True:
                self.data.drone_start_position = Point(self.measured_gps_position()[0], self.measured_gps_position()[1])

            elif self.is_inside_return_area != self.last_inside:
                self.last_inside = self.is_inside_return_area
                if self.is_inside_return_area is not None:
                    self.data.update_return_area_position(self.measured_gps_position())
                    self.path.return_area_postion = self.data.return_area.get_center()

            # Checks the surroundigs using both lidar THEN semantic sensors
            self.process_lidar_semantic_sensors()

        # RESCUING STATE
        elif self.state == self.Activity.RESCUING:
            pass

        return command
    

    def process_lidar_semantic_sensors(self):
        lidar_sensor_values = self.lidar_values()
        lidar_sensor_angles = self.lidar_rays_angles()
        values_features = detect_local_zones(lidar_sensor_values)
        angles_features = [[lidar_sensor_angles[i] for i in zone] for zone in values_features]

        semantic_sensor_values = self.semantic_values()
        if semantic_sensor_values is not None:
            for data in semantic_sensor_values:
                for i in range(len(angles_features)):
                    if angles_features[i][0] <= data.angle <= angles_features[i][-1]:
                        self.update_data(data, None, None)
                        values_features.pop(i)
                        angles_features.pop(i)
                        break
    
        if values_features != []:
            for i in range(len(values_features)):
                if len(values_features[i]) == 2:
                    pass
                else:
                    self.update_data(None, [values_features[i][0], values_features[i][-1], values_features[i][int(len(values_features[i])/2)+1]], [angles_features[i][0], angles_features[i][-1], angles_features[i][int(len(angles_features[i])/2)+1]])


    def update_data(self, data, values_features, angles_features):
        orientation = self.compass_values()
        if data is not None: # WOUNDED or RESCUE CENTER
            gps_position = self.measured_gps_position()
            self.actual_drone_position = Point(gps_position[0], gps_position[1])
            new_point = self.actual_drone_position.add(orientation, data.distance, data.angle)

            if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                self.data.update_rescue_center_position(new_point)
                if isinstance(self.data.rescue_center, Box):
                    self.path.rescue_center_position = self.data.rescue_center.get_center()

            elif data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON:
                self.data.update_wounded_position((new_point))
                print(len(self.data.wounded_position))

        else: # WALL
            new_point_1 = self.actual_drone_position.add(orientation, values_features[0], angles_features[0])
            new_point_2 = self.actual_drone_position.add(orientation, values_features[1], angles_features[1])
            middle_point = self.actual_drone_position.add(orientation, values_features[2], angles_features[2])
            self.data.update_walls_positions(new_point_1, new_point_2, middle_point)


class MyMapKeyboard(MapAbstract):

    def __init__(self, drone_type: Type[DroneAbstract]):
        super().__init__(drone_type=drone_type)

        # PARAMETERS MAP
        self._size_area = (600, 600)

        self._rescue_center = RescueCenter(size=(100, 100))
        self._rescue_center_pos = ((0, 100), 0)

        self._return_area = ReturnArea(size=(150, 100))
        self._return_area_pos = ((0, -20), 0)

        self._wounded_persons_pos = [(200, 0), (-200, 0),
                                     (200, -200), (-200, -200)]

        self._number_wounded_persons = len(self._wounded_persons_pos)
        self._wounded_persons: List[WoundedPerson] = []

        self._number_drones = 1
        self._drones_pos = [((0, 0), 0)]
        self._drones = []

        self._playground = ClosedPlayground(size=self._size_area)

        self._playground.add(self._rescue_center, self._rescue_center_pos)

        self._playground.add(self._return_area, self._return_area_pos)

        # POSITIONS OF THE WOUNDED PERSONS
        for i in range(self._number_wounded_persons):
            wounded_person = WoundedPerson(rescue_center=self._rescue_center)
            self._wounded_persons.append(wounded_person)
            pos = (self._wounded_persons_pos[i], 0)
            self._playground.add(wounded_person, pos)

        # POSITIONS OF THE DRONES
        misc_data = MiscData(size_area=self._size_area,
                             number_drones=self._number_drones,
                             max_timestep_limit=self._max_timestep_limit,
                             max_walltime_limit=self._max_walltime_limit)
        for i in range(self._number_drones):
            drone = drone_type(identifier=i, misc_data=misc_data)
            self._drones.append(drone)
            self._playground.add(drone, self._drones_pos[i])

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
    the_map = MapIntermediate01(drone_type=MyDroneTest)

    # draw_lidar_rays : enable the visualization of the lidar rays
    # draw_semantic_rays : enable the visualization of the semantic rays
    gui = GuiSR(the_map=the_map,
                draw_lidar_rays=False,
                draw_semantic_rays=False,
                use_keyboard=True,
                )
    gui.run()

    score_health_returned = the_map.compute_score_health_returned()
    print("score_health_returned = ", score_health_returned)


if __name__ == '__main__':
    main()