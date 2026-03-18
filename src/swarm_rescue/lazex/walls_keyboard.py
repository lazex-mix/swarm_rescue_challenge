import sys
from pathlib import Path

# Insert the parent directory of the current file's directory into sys.path.
# This allows Python to locate modules that are one level above the current
# script, in this case spg_overlay.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarm_rescue.simulation.elements.normal_wall import NormalWall, NormalBox


# Dimension of the map : (1113, 750)
# Dimension factor : 1.0


def add_walls(playground):
    # vertical wall 0  
    wall = NormalWall(pos_start=(200, 100),
                      pos_end=(200, -100))
    playground.add(wall, wall.wall_coordinates)

def add_boxes(playground):
    pass