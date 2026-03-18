import numpy as np

# --------------------------
# POINTS (2 coords) -> (x,y)
# --------------------------

def point(x, y):
    return np.hstack([float(x), float(y)])

def add(p, drone_orientation, distance, angle):
    total_angle = drone_orientation + angle
    vec = np.array([np.cos(total_angle), np.sin(total_angle)]) * distance
    return p + vec

def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def are_aligned(points, eps=1e-6):
    pts = np.array(points)
    var = pts.var(axis=0)
    if np.all(var < eps):
        return ("undefined", 1.0)
    if var[0] < eps:
        return ("vertical", np.inf)
    if var[1] < eps:
        return ("horizontal", np.inf)
    ratio = var[1] / var[0]
    return ("vertical", 1/ratio) if ratio < 1 else ("horizontal", ratio)

# --------------------------
# LINES (4 coords) -> (x1,y1,x2,y2)
# --------------------------

def line(p1, p2):
    return np.hstack([p1, p2])

def line_points(l):
    return l[:2], l[2:]

def line_length(l):
    p1, p2 = line_points(l)
    return np.linalg.norm(p2 - p1)

def line_type(l, eps=1e-6):
    p1, p2 = line_points(l)
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    if dx < eps and dy > eps:
        return "vertical"
    if dy < eps and dx > eps:
        return "horizontal"
    return "undefined"

def is_on_line(l, p, eps=1e-6):
    t = line_type(l)
    p1, p2 = line_points(l)
    if t == "vertical":
        return abs(p[0] - p1[0]) < eps and min(p1[1], p2[1]) <= p[1] <= max(p1[1], p2[1])
    if t == "horizontal":
        return abs(p[1] - p1[1]) < eps and min(p1[0], p2[0]) <= p[0] <= max(p1[0], p2[0])
    return False

def extend_line(l, p, eps=1e-6):
    """
    Étend une ligne pour inclure un nouveau point p.
    Si la ligne est verticale ou horizontale, on ajuste les coordonnées.
    Sinon, on transforme la ligne en box avec le point ajouté.
    
    l : np.array de shape (4,) -> [x1, y1, x2, y2]
    p : np.array de shape (2,) -> [x, y]
    """
    t = line_type(l, eps)
    
    if t == "vertical":
        ys = np.array([l[1], l[3], p[1]])
        return np.array([l[0], ys.min(), l[0], ys.max()], dtype=float)
    
    elif t == "horizontal":
        xs = np.array([l[0], l[2], p[0]])
        return np.array([xs.min(), l[1], xs.max(), l[1]], dtype=float)
    
    else:
        # Diagonal → transformer en box
        xs = [l[0], l[2], p[0]]
        ys = [l[1], l[3], p[1]]
        return make_box_from_limits(min(xs), max(xs), min(ys), max(ys))


# --------------------------
# BOXES (8 coords) -> (x1,y1,x2,y2,x3,y3,x4,y4)
# --------------------------

# Order:
# P1 bottom-left
# P2 bottom-right
# P3 top-right
# P4 top-left

def make_box_from_limits(min_x, max_x, min_y, max_y):
    p1 = point(min_x, min_y)
    p2 = point(max_x, min_y)
    p3 = point(max_x, max_y)
    p4 = point(min_x, max_y)
    return np.hstack([p1, p2, p3, p4])

def box_from_two_points(p1, p2):
    xs = sorted([p1[0], p2[0]])
    ys = sorted([p1[1], p2[1]])
    return make_box_from_limits(xs[0], xs[1], ys[0], ys[1])

def box_points(box):
    return box.reshape(4, 2)

def box_limits(box):
    pts = box_points(box)
    xs = pts[:, 0]
    ys = pts[:, 1]
    return xs.min(), xs.max(), ys.min(), ys.max()

def box_dimensions(box):
    min_x, max_x, min_y, max_y = box_limits(box)
    return max_x - min_x, max_y - min_y

def box_area(box):
    w, h = box_dimensions(box)
    return w * h

def box_center(box):
    min_x, max_x, min_y, max_y = box_limits(box)
    return np.array([(min_x + max_x)/2, (min_y + max_y)/2])

def box_contains(box, p):
    min_x, max_x, min_y, max_y = box_limits(box)
    return (min_x <= p[0] <= max_x) and (min_y <= p[1] <= max_y)

def extend_box(box, p):
    pts = box_points(box)
    xs = np.append(pts[:, 0], p[0])
    ys = np.append(pts[:, 1], p[1])
    return make_box_from_limits(xs.min(), xs.max(), ys.min(), ys.max())

def box_overlap_half(b1, b2):
    min_x1, max_x1, min_y1, max_y1 = box_limits(b1)
    min_x2, max_x2, min_y2, max_y2 = box_limits(b2)
    inter_w = max(0, min(max_x1, max_x2) - max(min_x1, min_x2))
    inter_h = max(0, min(max_y1, max_y2) - max(min_y1, min_y2))
    inter_area = inter_w * inter_h
    return inter_area >= box_area(b1) / 2

# --------------------------
# LOCAL ZONES DETECTION
# --------------------------

def detect_local_zones(arr, max_value=150, max_diff=50, min_zone_length=2):
    arr = np.array(arr)
    zones = []
    current = []
    prev = None
    for i, val in enumerate(arr):
        if val > max_value:
            if len(current) >= min_zone_length:
                zones.append(current)
            current = []
            prev = None
            continue
        if not current:
            current = [i]
        else:
            if prev is not None and abs(val - prev) <= max_diff:
                current.append(i)
            else:
                if len(current) >= min_zone_length:
                    zones.append(current)
                current = [i]
        prev = val
    if len(current) >= min_zone_length:
        zones.append(current)
    return zones