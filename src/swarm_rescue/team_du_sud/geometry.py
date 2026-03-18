import math
import numpy as np    

# On définit ici un ensemble d'outils géométriques utiles pour le mapping et le pathing du drone

class Point():
    def __init__(self, position_x, position_y):
        self.x = position_x
        self.y = position_y

    def add(self, drone_orientation, distance, angle):
        total_angle = drone_orientation + angle
        dx = distance * math.cos(total_angle)
        dy = distance * math.sin(total_angle)
        return Point(self.x + dx, self.y + dy)

    def are_aligned_in_type(self, point_1, point_2, epsilon=1e-6):
        x_coords = [point_1.x, point_2.x, self.x]
        y_coords = [point_1.y, point_2.y, self.y]

        mean_x = sum(x_coords) / 3
        mean_y = sum(y_coords) / 3
        var_x = sum((x - mean_x) ** 2 for x in x_coords) / 3
        var_y = sum((y - mean_y) ** 2 for y in y_coords) / 3

        if var_x < epsilon and var_y < epsilon:
            return ("undefined", 1.0)
        elif var_x < epsilon:
            return ("vertical", float('inf'))
        elif var_y < epsilon:
            return ("horizontal", float('inf'))

        ratio = var_y / var_x
        if ratio < 1:
            return ("vertical", 1 / ratio)
        else:
            return ("horizontal", ratio)
    
    def distance_to(self, point):
        return ((self.x - point.x) ** 2 + (self.y - point.y) ** 2) ** 0.5
    
    def __str__(self):
        return f"({self.x}, {self.y})"

    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))


class Line():
    def __init__(self, point_1: Point, point_2: Point):
        self.point_1 = point_1
        self.point_2 = point_2
        self.type = None

    def update_type(self, type: str):
        self.type = type
        
    def get_position(self):
        if self.type == "vertical":
            return [self.point_1.x, self.point_1.y, self.point_2.y]
        elif self.type == "horizontal":
            return [self.point_1.y, self.point_1.x, self.point_2.x]
        
    def get_length(self):
        return math.hypot(self.point_2.x - self.point_1.x, self.point_2.y - self.point_1.y)

    def is_on_line(self, point: Point):
        x1, y1, x2, y2 = self.point_1.x, self.point_1.y, self.point_2.x, self.point_2.y
        if self.type == "vertical":
            return point.x == x1 and min(y1, y2) <= point.y <= max(y1, y2)
        elif self.type == "horizontal":
            return point.y == y1 and min(x1, x2) <= point.x <= max(x1, x2)

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
            new_line = Line(Point(self.point_1.x, min(y_positions)), Point(self.point_1.x, max(y_positions)))
        elif self.type == "horizontal":
            x_positions = [self.point_1.x, self.point_2.x, point_3.x]
            new_line = Line(Point(min(x_positions), self.point_1.y), Point(max(x_positions), self.point_1.y))
        self.point_1 = new_line.point_1
        self.point_2 = new_line.point_2

    def __str__(self):
        return f"Line from {self.point_1} to {self.point_2} of type {self.type}"

    def __eq__(self, other):
        if not isinstance(other, Line):
            return NotImplemented
        return (self.point_1 == other.point_1 or self.point_1 == other.point_2) and (self.point_2 == other.point_1 or self.point_2 == other.point_2)
    
    def __hash__(self):
        return hash((self.point_1, self.point_2))


class Box():
    def __init__(self, point_1: Point, point_2: Point, point_3: Point, point_4: Point):
        self.point_1 = point_1
        self.point_2 = point_2
        self.point_3 = point_3
        self.point_4 = point_4
        self.points = [point_1, point_2, point_3, point_4]

    def order_points(self): # Méthode pertiente car tout est horizontal/vertical 
        index = 1
        for i in  range(3):
            if self.point_1.x != self.points[i+1].x and self.point_1.y != self.points[i+1].y:
                index = i+1
                break
        if index != 2:
            inter_point = self.points[2]
            self.points[2] = self.points[index]
            self.points[index] = inter_point
        # Les points sont ordonnées dans le sens horaire ou trigo à partir du point 1
        self.point_1 = self.points[0]
        self.point_2 = self.points[1]
        self.point_3 = self.points[2]
        self.point_4 = self.points[3]
    
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
        new_box = Box(Point(min_x, min_y), Point(max_x, min_y), Point(max_x, max_y), Point(min_x, max_y))
        self.point_1 = new_box.point_1
        self.point_2 = new_box.point_2
        self.point_3 = new_box.point_3
        self.point_4 = new_box.point_4

    def get_lines(self):
        self.order_points()
        self.points = [self.point_1, self.point_2, self.point_3, self.point_4]
        lines = []
        for i in range(4):
            line = Line(self.points[i], self.points[(i+1)%4])
            lines.append(line)
        return lines # Les lignes sont dans le sens horaire ou trigo à partir du point 1

    def get_area(self):
        width, height = self.get_dimensions()
        return width * height
    
    def get_center(self):
        min_x, max_x, min_y, max_y = self.get_limits()
        return Point((min_x + max_x) / 2, (min_y + max_y) / 2)

    def get_limits(self):
        xs = (self.point_1.x, self.point_2.x, self.point_3.x, self.point_4.x)
        ys = (self.point_1.y, self.point_2.y, self.point_3.y, self.point_4.y)
        return min(xs), max(xs), min(ys), max(ys)
    
    def get_dimensions(self):
        self.limits = self.get_limits()
        width = self.limits[1] - self.limits[0]
        height = self.limits[3] - self.limits[2]
        return width, height
    
    def is_inside(self, point: Point):
        self.order_points()
        self.points = [self.point_1, self.point_2, self.point_3, self.point_4]
        if (self.points[0].x <= point.x <= self.points[2].x and
            self.points[0].y <= point.y <= self.points[2].y):
            return True
        return False
    
    def covers_more_than_half(self, other):
        min_x1, max_x1, min_y1, max_y1 = self.get_limits()
        min_x2, max_x2, min_y2, max_y2 = other.get_limits()

        inter_w = max(0, min(max_x1, max_x2) - max(min_x1, min_x2))
        inter_h = max(0, min(max_y1, max_y2) - max(min_y1, min_y2))
        inter_area = inter_w * inter_h
        return inter_area >= self.get_area() / 2

    def __str__(self):
        self.order_points()
        return f"Box with points {self.point_1}, {self.point_2}, {self.point_3}, {self.point_4}"
    
    def __eq__(self, other):
        if not isinstance(other, Box):
            return NotImplemented
        return (self.point_1 == other.point_1 and
                self.point_2 == other.point_2 and
                self.point_3 == other.point_3 and
                self.point_4 == other.point_4)
    
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
    return Box(Point(min_x, min_y), Point(max_x, min_y), Point(max_x, max_y), Point(min_x, max_y))

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
