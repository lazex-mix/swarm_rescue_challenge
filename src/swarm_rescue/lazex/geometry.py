import math

import numpy as np

# On définit ici un ensemble d'outils géométriques utiles pour le mapping et le pathing du drone


class Point:
    def __init__(self, position_x, position_y):
        self.x = position_x
        self.y = position_y

    def add(self, drone_orientation, distance, angle):
        total_angle = drone_orientation + angle
        vec = np.array([np.cos(total_angle), np.sin(total_angle)]) * distance
        return Point(self.x + vec[0], self.y + vec[1])

    def are_aligned_in_type(self, point_1, point_2, epsilon=1e-6):
        coords = np.array([[self.x, self.y], [point_1.x, point_1.y], [point_2.x, point_2.y]])
        var_x, var_y = np.var(coords, axis=0)

        if var_x < epsilon and var_y < epsilon:
            return ("undefined", 1.0)
        elif var_x < epsilon:
            return ("vertical", float("inf"))
        elif var_y < epsilon:
            return ("horizontal", float("inf"))

        ratio = var_y / var_x
        if ratio < 1:
            return ("vertical", 1 / ratio)
        else:
            return ("horizontal", ratio)

    def distance_to(self, point):
        return np.linalg.norm(np.array([self.x - point.x, self.y - point.y]))

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))


class Line:
    def __init__(self, point_1: Point, point_2: Point):
        self.point_1 = point_1
        self.point_2 = point_2
        # Automatically determine line type
        if abs(point_1.x - point_2.x) < 1e-9:  # Vertical line
            self.type = "vertical"
        elif abs(point_1.y - point_2.y) < 1e-9:  # Horizontal line
            self.type = "horizontal"
        else:
            self.type = "diagonal"

    def update_type(self, type: str):
        self.type = type

    def get_position(self):
        if self.type == "vertical":
            return [self.point_1.x, self.point_1.y, self.point_2.y]
        elif self.type == "horizontal":
            return [self.point_1.y, self.point_1.x, self.point_2.x]

    def get_length(self):
        p1 = np.array([self.point_1.x, self.point_1.y])
        p2 = np.array([self.point_2.x, self.point_2.y])
        return np.linalg.norm(p2 - p1)

    def is_on_line(self, point: Point):
        x1, y1, x2, y2 = self.point_1.x, self.point_1.y, self.point_2.x, self.point_2.y
        if self.type == "vertical":
            return point.x == x1 and min(y1, y2) <= point.y <= max(y1, y2)
        elif self.type == "horizontal":
            return point.y == y1 and min(x1, x2) <= point.x <= max(x1, x2)

    def is_adjacent(self, line: "Line"):
        if self.type != line.type:
            return False
        if self.type == "vertical":
            if self.point_1.x != line.point_1.x:
                return False
            y0, y1 = sorted((self.point_1.y, self.point_2.y))
            ly0, ly1 = sorted((line.point_1.y, line.point_2.y))
            return min(y1, ly1) > max(y0, ly0) 
        else:  # horizontal
            if self.point_1.y != line.point_1.y:
                return False
            x0, x1 = sorted((self.point_1.x, self.point_2.x))
            lx0, lx1 = sorted((line.point_1.x, line.point_2.x))
            return min(x1, lx1) > max(x0, lx0)



    def is_aligned(self, point: Point):
        x1, y1, x2, y2 = self.point_1.x, self.point_1.y, self.point_2.x, self.point_2.y
        px, py = point.x, point.y
        if self.type == "vertical":
            return px == x1 and (py <= min(y1, y2) or py >= max(y1, y2))
        elif self.type == "horizontal":
            return py == y1 and (px <= min(x1, x2) or px >= max(x1, x2))
        return False

    def extend_line(self, point_3: Point):
        if self.type == "vertical":
            y_positions = [self.point_1.y, self.point_2.y, point_3.y]
            new_line = Line(
                Point(self.point_1.x, min(y_positions)),
                Point(self.point_1.x, max(y_positions)),
            )
        elif self.type == "horizontal":
            x_positions = [self.point_1.x, self.point_2.x, point_3.x]
            new_line = Line(
                Point(min(x_positions), self.point_1.y),
                Point(max(x_positions), self.point_1.y),
            )
        self.point_1 = new_line.point_1
        self.point_2 = new_line.point_2

    def __str__(self):
        return f"Line from {self.point_1} to {self.point_2} of type {self.type}"

    def __eq__(self, other):
        if not isinstance(other, Line):
            return NotImplemented
        return (self.point_1 == other.point_1 or self.point_1 == other.point_2) and (
            self.point_2 == other.point_1 or self.point_2 == other.point_2
        )

    def __hash__(self):
        return hash((self.point_1, self.point_2))


class Box:
    def __init__(self, point_1: Point, point_2: Point, point_3: Point, point_4: Point):
        self.point_1 = point_1
        self.point_2 = point_2
        self.point_3 = point_3
        self.point_4 = point_4
        self.points = [point_1, point_2, point_3, point_4]

    def order_points(self):
        # Sort points by y first, then x using NumPy lexsort
        coords = np.array([[p.x, p.y] for p in self.points])
        # lexsort sorts by last key first, so we pass (x, y) to sort by (y, x)
        idx = np.lexsort((coords[:, 0], coords[:, 1]))
        pts = [self.points[i] for i in idx]

        p1 = pts[0]  # top-left or bottom-left depending on your system

        # All points with same y as p1
        same_row = [p for p in pts if p.y == p1.y]

        # Rightmost on same row
        def sort_by_x(p):
            return p.x

        p2 = max(same_row, key=sort_by_x)

        remaining = [p for p in pts if p not in (p1, p2)]

        # The one with smallest x becomes p4
        p4 = min(remaining, key=sort_by_x)

        # The last remaining one is p3
        p3 = [p for p in remaining if p != p4][0]

        self.point_1, self.point_2, self.point_3, self.point_4 = p1, p2, p3, p4
        self.points = [p1, p2, p3, p4]
        return self

    def extend_box(self, point: Point):
        min_x, max_x, min_y, max_y = self.get_limits()
        if min_x >= point.x:
            min_x = point.x
        if max_x <= point.x:
            max_x = point.x
        if min_y >= point.y:
            min_y = point.y
        if max_y <= point.y:
            max_y = point.y
        new_box = Box(
            Point(min_x, min_y),
            Point(max_x, min_y),
            Point(max_x, max_y),
            Point(min_x, max_y),
        )
        self.point_1 = new_box.point_1
        self.point_2 = new_box.point_2
        self.point_3 = new_box.point_3
        self.point_4 = new_box.point_4

    def get_lines(self):
        self.order_points()
        self.points = [self.point_1, self.point_2, self.point_3, self.point_4]
        lines = []
        for i in range(4):
            line = Line(self.points[i], self.points[(i + 1) % 4])
            lines.append(line)
        return (
            lines  # Les lignes sont dans le sens horaire ou trigo à partir du point 1
        )

    def get_area(self):
        width, height = self.get_dimensions()
        return width * height

    def get_center(self):
        min_x, max_x, min_y, max_y = self.get_limits()
        return Point((min_x + max_x) / 2, (min_y + max_y) / 2)

    def get_limits(self):
        coords = np.array([[self.point_1.x, self.point_1.y],
                          [self.point_2.x, self.point_2.y],
                          [self.point_3.x, self.point_3.y],
                          [self.point_4.x, self.point_4.y]])
        min_x, min_y = coords.min(axis=0)
        max_x, max_y = coords.max(axis=0)
        return min_x, max_x, min_y, max_y

    def get_dimensions(self):
        self.limits = self.get_limits()
        width = self.limits[1] - self.limits[0]
        height = self.limits[3] - self.limits[2]
        return width, height

    def is_inside(self, point: Point):
        self.order_points()
        self.points = [self.point_1, self.point_2, self.point_3, self.point_4]
        if (
            self.points[0].x <= point.x <= self.points[2].x
            and self.points[0].y <= point.y <= self.points[2].y
        ):
            return True
        return False

    def covers_more_than_half(self, other):
        min_x1, max_x1, min_y1, max_y1 = self.get_limits()
        min_x2, max_x2, min_y2, max_y2 = other.get_limits()

        inter_w = max(0, min(max_x1, max_x2) - max(min_x1, min_x2))
        inter_h = max(0, min(max_y1, max_y2) - max(min_y1, min_y2))
        inter_area = inter_w * inter_h
        return inter_area >= self.get_area() / 2

    def are_neighbors(self, other):
        self_lines = self.get_lines()
        other_lines = other.get_lines()
        return (
            Line.is_adjacent(self_lines[0], other_lines[2])
            or Line.is_adjacent(self_lines[2], other_lines[0])
            or Line.is_adjacent(self_lines[1], other_lines[3])
            or Line.is_adjacent(self_lines[3], other_lines[1])
        )

    def __str__(self):
        self.order_points()
        return f"Box with points {self.point_1}, {self.point_2}, {self.point_3}, {self.point_4}"

    def __eq__(self, other):
        if not isinstance(other, Box):
            return NotImplemented
        return (
            self.point_1 == other.point_1
            and self.point_2 == other.point_2
            and self.point_3 == other.point_3
            and self.point_4 == other.point_4
        )

    def __hash__(self):
        return hash((self.point_1, self.point_2, self.point_3, self.point_4))


# Fonctions supplémentaires
def build_box_with_2_opposite_points(point_1: Point, point_2: Point):
    point_3 = Point(point_1.x, point_2.y)
    point_4 = Point(point_1.y, point_2.x)
    return Box(point_1, point_2, point_3, point_4)


def build_box_with_line_and_point(line: Line, point: Point):
    x_positions = [line.point_1.x, line.point_2.x, point.x]
    y_positions = [line.point_1.y, line.point_2.y, point.y]
    min_x = min(x_positions)
    max_x = max(x_positions)
    min_y = min(y_positions)
    max_y = max(y_positions)
    return Box(
        Point(min_x, min_y),
        Point(max_x, min_y),
        Point(max_x, max_y),
        Point(min_x, max_y),
    )


def detect_local_zones(liste, max_value=150, max_diff=50, min_zone_length=2):
    zones = []
    current_zone = []
    prev_val = None

    for i, val in enumerate(liste):
        if val > max_value:
            if len(current_zone) >= min_zone_length:
                zones.append(current_zone)
            current_zone = []
            prev_val = None
            continue

        if not current_zone:
            current_zone.append(i)
        else:
            if abs(val - prev_val) <= max_diff:
                current_zone.append(i)
            else:
                if len(current_zone) >= min_zone_length:
                    zones.append(current_zone)
                current_zone = [i]
        prev_val = val

    if len(current_zone) >= min_zone_length:
        zones.append(current_zone)
    return zones


# NUMPY-OPTIMIZED BATCH OPERATIONS FOR DIJKSTRA/PATHFINDING

def batch_distances_from_point(point, others):
    """
    Calculate distance from one point to many points (vectorized).
    ~10-100× faster than calling distance_to() in a loop.
    
    Args:
        point: Point object
        others: List of Point objects
    
    Returns:
        numpy array of distances
    """
    p_array = np.array([point.x, point.y])
    others_array = np.array([[o.x, o.y] for o in others])
    return np.linalg.norm(others_array - p_array, axis=1)


def batch_distances_from_center(box, other_boxes):
    """
    Calculate distance from one box's center to many boxes' centers.
    
    Args:
        box: Box object
        other_boxes: List of Box objects
    
    Returns:
        numpy array of distances
    """
    center = box.get_center()
    other_centers = [b.get_center() for b in other_boxes]
    return batch_distances_from_point(center, other_centers)


def find_box_containing_point(boxes, point):
    """
    Find which box (if any) contains a given point.
    
    Args:
        boxes: List of Box objects
        point: Point object
    
    Returns:
        First Box that contains the point, or None
    """
    for box in boxes:
        if box.is_inside(point):
            return box
    return None
