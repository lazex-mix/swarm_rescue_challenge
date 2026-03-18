import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from swarm_rescue.lazex.QuadTree import QuadTree
from swarm_rescue.lazex.geometry import Point
from swarm_rescue.lazex.dijkstra import GraphBuilder


# ============================================================================
# 1. SETUP - Map dimensions and quadtree initialization
# ============================================================================
W, H = 200, 150
MIN_SIZE = 8

qt = QuadTree(W, H, MIN_SIZE)
obstacle_points = []
graph = None


# ============================================================================
# 2. TEST DATA - Define obstacle patterns
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


steps = [
    ("Step 1: L-shaped corner", lambda: (vline(50, 30, 80, step=1), hline(50, 90, 80, step=1))),
    ("Step 2: central pillar", lambda: (hline(100, 120, 50), hline(100, 120, 70), vline(100, 50, 70), vline(120, 50, 70))),
    ("Step 3: top barrier", lambda: hline(20, 180, 120, step=1)),
    ("Step 4: diagonal maze walls", lambda: (vline(70, 90, 110, step=1), vline(130, 90, 110, step=1), hline(70, 130, 100, step=1))),
    ("Step 5: bottom chamber", lambda: (hline(40, 80, 25), hline(120, 160, 25), vline(40, 25, 45), vline(80, 25, 45), vline(120, 25, 45), vline(160, 25, 45))),
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
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title("Quadtree + Graph: L-corner, pillar, barriers & chambers", fontsize=12)


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

# Draw obstacle points
ax.scatter([p.x for p in obstacle_points], [p.y for p in obstacle_points], 
           c="black", s=6, marker="s", alpha=0.8, zorder=3)

# Draw graph nodes (centers of unoccupied cells)
node_centers_x = [node.box.get_center().x for node in graph.keys()]
node_centers_y = [node.box.get_center().y for node in graph.keys()]
ax.scatter(node_centers_x, node_centers_y, c="blue", s=30, marker="o", alpha=0.7, zorder=4)

# Draw graph edges
for node, neighbors in graph.items():
    center1 = node.box.get_center()
    for neighbor in neighbors:
        center2 = neighbor.box.get_center()
        ax.plot([center1.x, center2.x], [center1.y, center2.y], 
                color="green", alpha=0.4, linewidth=1, zorder=2)

# Finalize plot
ax.set_xlim(-5, W + 5)
ax.set_ylim(-5, H + 5)
ax.set_aspect("equal")
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# Legend
legend_elements = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor="black", markersize=6, label="Obstacles", alpha=0.8),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=8, label="Graph nodes", alpha=0.7),
    Line2D([0], [0], color="green", linewidth=1, label="Graph edges", alpha=0.4),
]
ax.legend(handles=legend_elements, loc="upper right")

plt.tight_layout()
plt.show()
