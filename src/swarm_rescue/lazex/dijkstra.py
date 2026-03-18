
from swarm_rescue.lazex.QuadTree import Node
import heapq


class GraphBuilder:

    def __init__(self, existing_graph, unoccupied_nodes, nodes_to_prune):
        self.existing_graph = existing_graph
        self.unoccupied_nodes = unoccupied_nodes
        self.nodes_to_prune = nodes_to_prune

    def _calculate_weight(self, node_from, node_to):
        center_from = node_from.box.get_center()
        center_to = node_to.box.get_center()
        return center_from.distance_to(center_to)

    def _prune(self):

        if self.existing_graph is None :
            return
        
        for node in self.nodes_to_prune :
            if node in self.existing_graph:
                for edge in self.existing_graph[node] :
                    neighbor = edge[0]
                    if neighbor in self.existing_graph :
                        for e in self.existing_graph[neighbor]:
                            if e[0] == node:
                                self.existing_graph[neighbor].remove(e)
                                break
                del self.existing_graph[node]

    def _update_existing(self, new_nodes) :
        existing_nodes = [n for n in self.existing_graph if n not in new_nodes]
        for n in existing_nodes :
            for m in new_nodes :
                if n.box.are_neighbors(m.box) :
                    weight = self._calculate_weight(n, m)
                    self.existing_graph[n].append((m, weight))


    def build(self):     #complexity is O(K x U) with K the new nodes processed and U the total unoccupied nodes in the quadtree
        
        if self.existing_graph is None:
            self.existing_graph = {}

        self._prune()

        graph = self.existing_graph
        
        new_nodes = [n for n in self.unoccupied_nodes if n not in graph]
        for node in new_nodes:
            neighbors = node.adjacency_list(self.unoccupied_nodes)
            graph[node] = [(neighbor, self._calculate_weight(node, neighbor)) 
                           for neighbor in neighbors]
            
        self._update_existing(new_nodes)

        return graph


class Path_finding:
    def __init__(self, graph):
        self.graph = graph

    def Dijkstra(self, start_node, end_node=None):
        """
        Find shortest path from start_node to end_node (or all nodes if end_node is None).
        Uses a min-heap for O((V + E) log V) complexity.
        
        Args:
            start_node: Starting node
            end_node: Target node (optional - if None, computes distances to all nodes)
            
        Returns:
            (distances, parents) - distances dict and parent dict for path reconstruction
        """
        distances = {node: float('inf') for node in self.graph}
        distances[start_node] = 0
        parents = {start_node: None}  # Track path
        
        # Priority queue: (distance, node)
        heap = [(0, id(start_node), start_node)]  # id() for tie-breaking
        visited = set()
        
        while heap:
            current_dist, _, current = heapq.heappop(heap)
            
            # Skip if already visited (we may have duplicate entries)
            if current in visited:
                continue
            visited.add(current)
            
            # Early exit if we reached the target
            if end_node and current == end_node:
                break
            
            # Explore neighbors
            for neighbor, edge_weight in self.graph[current]:
                if neighbor in visited:
                    continue
                    
                new_distance = current_dist + edge_weight
                
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    parents[neighbor] = current
                    heapq.heappush(heap, (new_distance, id(neighbor), neighbor))
        
        return distances, parents

    def get_path(self, parents, end_node):
        """
        Reconstruct the path from start to end_node using parent pointers.
        
        Args:
            parents: Parent dict from Dijkstra
            end_node: Target node
            
        Returns:
            List of nodes from start to end (or empty if unreachable)
        """
        if end_node not in parents:
            return []  # Unreachable
        
        path = []
        current = end_node
        while current is not None:
            path.append(current)
            current = parents[current]
        
        path.reverse()  # Start -> End order
        return path

    def find_path(self, start_node, end_node):
        """
        Convenience method: find the shortest path between two nodes.
        
        Returns:
            List of nodes from start to end (empty if unreachable)
        """
        distances, parents = self.Dijkstra(start_node, end_node)
        return self.get_path(parents, end_node)

        return distances


