import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from swarm_rescue.lazex.QuadTree import QuadTree
from swarm_rescue.lazex.geometry import Point
from swarm_rescue.lazex.dijkstra import GraphBuilder


# ============================================================================
# 1. SETUP - Map dimensions and quadtree initialization
# ============================================================================
W, H = 900, 650
MIN_SIZE = 16

qt = QuadTree(W, H, MIN_SIZE)
obstacle_points = []
graph = None


# ============================================================================
# 2. TEST DATA - Replicate the image pattern
# ============================================================================
def hline(x0, x1, y, step=1):
    """Add horizontal line of obstacle points."""
    x_start, x_end = sorted((x0, x1))
    for x in range(x_start, x_end + 1, step):
        obstacle_points.append(Point(x, y))
        qt.insert_point(Point(x, y))


def vline(x, y0, y1, step=1):
    """Add vertical line of obstacle points."""
    y_start, y_end = sorted((y0, y1))
    for y in range(y_start, y_end + 1, step):
        obstacle_points.append(Point(x, y))
        qt.insert_point(Point(x, y))


def box(x0, y0, x1, y1):
    """Create a filled rectangular box."""
    for y in range(int(min(y0, y1)), int(max(y0, y1)) + 1):
        hline(x0, x1, y)


# Recreate the pattern from the image
steps = [
    ("Top-left L-shape", lambda: (
        hline(10, 80, 640),
        vline(10, 570, 640),
        hline(10, 200, 570),
        vline(200, 470, 570)
    )),
    
    ("Top left chamber", lambda: (
        hline(10, 200, 470),
        vline(60, 470, 540),
        vline(120, 470, 540)
    )),
    
    ("Bottom left chamber", lambda: (
        hline(10, 200, 180),
        vline(10, 10, 180),
        vline(200, 10, 180),
        box(10, 10, 65, 60),
        box(105, 10, 145, 60)
    )),
    
    ("Central pillar and structures", lambda: (
        box(260, 470, 315, 570),
        box(260, 50, 500, 330)
    )),
    
    ("Top barriers", lambda: (
        hline(380, 450, 640),
        box(380, 580, 430, 640),
        box(470, 580, 520, 640),
        box(760, 560, 810, 620)
    )),
    
    ("Right side structures", lambda: (
        hline(630, 890, 330),
        vline(630, 170, 330),
        hline(245, 760, 360),
        vline(820, 10, 550),
        box(820, 10, 890, 90)
    )),
    
    ("Maze-like structures", lambda: (
        hline(245, 340, 410),
        vline(280, 410, 450),
        hline(340, 390, 480),
        hline(520, 890, 480),
        vline(520, 410, 480),
        vline(610, 480, 510)
    )),
]


# ============================================================================
# 3. EXECUTE - Run incremental steps and build graph
# ============================================================================
for label, action in steps:
    # Add new obstacles
    action()
    
    # Get quadtree state
    pruned = qt.get_pruned_nodes()
    unocc = qt.get_unoccupied_nodes()
    
    # Build/update graph
    builder = GraphBuilder(graph, unocc, pruned)
    graph = builder.build()
    
    # Print stats
    total_edges = sum(len(v) for v in graph.values())
    print(f"\n{label}")
    print(f"  Obstacles:      {len(obstacle_points)} points")
    print(f"  Unoccupied:     {len(unocc)} leaves")
    print(f"  Pruned:         {len(pruned)} nodes")
    print(f"  Graph nodes:    {len(graph)}")
    print(f"  Graph edges:    {total_edges} (directed)")


# ============================================================================
# 4. VISUALIZE - Draw quadtree and graph overlay
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_title("Quadtree + Graph: Image Pattern Replication", fontsize=14)


def draw_quadtree(ax, node):
    """Recursively draw quadtree cell outlines."""
    xs = [p.x for p in node.box.points]
    ys = [p.y for p in node.box.points]
    x_min, y_min = min(xs), min(ys)
    w, h = max(xs) - x_min, max(ys) - y_min
    edgecolor = "darkred" if node.occupied else "lightgray"
    linewidth = 1.1 if node.occupied else 0.4
    rect = plt.Rectangle((x_min, y_min), w, h, edgecolor=edgecolor, facecolor="none", linewidth=linewidth)
    ax.add_patch(rect)
    if node.children:
        for child in node.children:
            draw_quadtree(ax, child)


# Draw quadtree cells
draw_quadtree(ax, qt.root)

# Draw obstacle points (sampled to reduce clutter)
sampled_obstacles = obstacle_points[::5]  # Show every 5th point
ax.scatter([p.x for p in sampled_obstacles], [p.y for p in sampled_obstacles], 
           c="black", s=4, marker="s", alpha=0.7, zorder=3)

# Draw graph nodes (centers of unoccupied cells)
node_centers_x = [node.box.get_center().x for node in graph.keys()]
node_centers_y = [node.box.get_center().y for node in graph.keys()]
ax.scatter(node_centers_x, node_centers_y, c="blue", s=20, marker="o", alpha=0.6, zorder=4)

# Draw graph edges
for node, neighbors in graph.items():
    center1 = node.box.get_center()
    for neighbor in neighbors:
        center2 = neighbor.box.get_center()
        ax.plot([center1.x, center2.x], [center1.y, center2.y], 
                color="green", alpha=0.3, linewidth=0.8, zorder=2)

# Finalize plot
ax.set_xlim(-20, W + 20)
ax.set_ylim(-20, H + 20)
ax.set_aspect("equal")
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# Legend
legend_elements = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor="black", markersize=5, label="Obstacles (sampled)", alpha=0.7),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=6, label="Graph nodes", alpha=0.6),
    Line2D([0], [0], color="green", linewidth=1, label="Graph edges", alpha=0.3),
]
ax.legend(handles=legend_elements, loc="upper right")

plt.tight_layout()
plt.show()
