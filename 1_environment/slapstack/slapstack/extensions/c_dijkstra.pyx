# distutils: language=c++

from libc.math cimport hypot
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.set cimport set

cdef float compute_distance(pair[int, int] a, pair[int, int] b):
    return hypot(a.first - b.first, a.second - b.second)

# Defines the relative index differences for the neighbours of any node
# Each item inside the list is a tuple of the format (x_offset, y_offset)
cdef vector[pair[int, int]] relative_positions = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]

cdef vector[pair[int, int]] get_adjacent(pair[int, int] node, set[pair[int, int]] nodes):
    """
    Computes the adjacent neighbours of a node/tile, which are available for movement.
    The adjacency is calculated on the basis of relative_positions.
    To check if the node is available for movement, we see if it is present in "nodes"
    """
    # if row < 0 or row >= self.n_rows or column < 0 or column >= \
    #         self.n_columns:
    #     raise Exception("Out of bounds")

    cdef vector[pair[int, int]] adj_nodes

    cdef pair[int, int] rel_pos
    cdef pair[int, int] adj_node
    cdef int index

    for index in range(relative_positions.size() - 1):
        rel_pos = relative_positions[index]
        #  if rel_pos is within bounds of graph
        adj_node = pair[int, int](node.first + rel_pos.first, node.second + rel_pos.second)
        if nodes.find(adj_node) != nodes.end():
            adj_nodes.push_back(adj_node)

    return adj_nodes

def perform_cdijkstra(pair[int, int] start, pair[int, int] end,
                      set[pair[int, int]] nodes, float unit_distance,
                      float speed):  # dijkstra
    """dijkstra's algorithm to find shortest path
    1) create a new dictionary 'unseen_nodes' that contains all nodes in
        the graph by creating a copy of the grid's nodes dictionary
    2) create a new dictionary 'distance' that will save the shortest
        distance to get to that node. initially all distances are infinity
    3) create a dictionary 'previous_node' that will save the node that has
        the shortest distance to each node in unseen nodes
    4) sets distance of starting node to 0 so it gets picked first
        in next step
    5) loop while there are unseen nodes:
        1) save node with shortest distance to variable node_min_dist
            and remove from unseen_nodes
        2) get all adjacent nodes to node_min_dist and find distance
            between node_min_dist and each adjacent node, taking into
            account stacking penalties
        3) if the distance to get to node_min_dist plus the distance to
            get to an adjacent node is less than the distance that was
            already saved for the same adjacent node
            1) update the shortest distance to that adjacent node
            2) in previous_node dict, set the value for the adjacent_node
            key to the current node_min_dist
        4) continue until all unseen_nodes are visited
    5) set current node to the end node and work backwards from the end
        node in order to finally create the shortest path
    6) create path as list with end node as first value
    6) loop while there is a previous node (there won't be a previous node
        for the starting node)
        1) get the previous node from previous_node for the current node
        2) append the previous node to the path
        2) set the previous node as the current node
    7) reverse path to get path from start to end
    """

    # unseen_nodes = self.location_manager.grid.get_nodes_without_aisles_and_storage().copy()

    cdef:
        pair[int, int] node_min_dist
        float min_dist
        vector[pair[int, int]] adjacent_nodes
        float dist_btwn_adj_node_and_cur_node
        float alternate_path_dist
        int i

        map[pair[int, int], pair[int, int]] previous_node
        map[pair[int, int], float] distance
        set[pair[int, int]] visited
        vector[pair[int, int]] explored_nodes

    explored_nodes.push_back(start)
    previous_node[start] = pair[int, int](-1, -1)
    distance[start] = 0.
    while True:  # while there are still unseen_nodes in list
        node_min_dist = pair[int, int](-1, -1)
        min_dist = float('inf')
        for i in range(explored_nodes.size()):
            if visited.find(explored_nodes[i]) == visited.end() and distance[explored_nodes[i]] < min_dist:
                min_dist = distance[explored_nodes[i]]
                node_min_dist = explored_nodes[i]

        if node_min_dist.first == -1 or node_min_dist == end:
            break

        visited.insert(node_min_dist)

        adjacent_nodes = get_adjacent(node_min_dist, nodes)

        for i in range(adjacent_nodes.size()):
            dist_btwn_adj_node_and_cur_node = compute_distance(node_min_dist, adjacent_nodes[i]) * unit_distance
            alternate_path_dist = distance[node_min_dist] + \
                                  dist_btwn_adj_node_and_cur_node
            if distance.find(adjacent_nodes[i]) == distance.end():
                distance[adjacent_nodes[i]] = float('inf')
                explored_nodes.push_back(adjacent_nodes[i])
            if alternate_path_dist < distance[adjacent_nodes[i]]:
                distance[adjacent_nodes[i]] = alternate_path_dist
                previous_node[adjacent_nodes[i]] = node_min_dist

    if distance.find(end) != distance.end():
        total_distance = distance[end]

        path = [(end.first, end.second)]
        edge_distances = []

        current = end
        while previous_node[current].first != -1:
            path.append(tuple(previous_node[current]))
            edge_distances.append(compute_distance(current, previous_node[current]) * unit_distance)
            current = previous_node[current]
        path = list(reversed(path))
        edge_distances = list(reversed(edge_distances))
        edge_times = [i / speed for i in edge_distances]

        edge_distance_midpoints = []
        counter = 0
        for i in range(len(edge_distances)):
            edge_distance_midpoints.append(counter + (edge_distances[i] / 2))
            counter += edge_distances[i]

        edge_time_midpoints = [(i / speed) for i in
                               edge_distance_midpoints]

        duration = sum(edge_times)
        return path, total_distance, duration, edge_time_midpoints
    else:
        raise Exception("No path found to the destination in dijkstra search.")


def perform_cdijkstra_without_end_node(pair[int, int] start, set[pair[int, int]] nodes, float unit_distance):  # dijkstra
        """dijkstra's algorithm to find shortest path
        1) create a new dictionary 'unseen_nodes' that contains all nodes in
            the graph by creating a copy of the grid's nodes dictionary
        2) create a new dictionary 'distance' that will save the shortest
            distance to get to that node. initially all distances are infinity
        3) create a dictionary 'previous_node' that will save the node that has
            the shortest distance to each node in unseen nodes
        4) sets distance of starting node to 0 so it gets picked first
            in next step
        5) loop while there are unseen nodes:
            1) save node with shortest distance to variable node_min_dist
                and remove from unseen_nodes
            2) get all adjacent nodes to node_min_dist and find distance
                between node_min_dist and each adjacent node, taking into
                account stacking penalties
            3) if the distance to get to node_min_dist plus the distance to
                get to an adjacent node is less than the distance that was
                already saved for the same adjacent node
                1) update the shortest distance to that adjacent node
                2) in previous_node dict, set the value for the adjacent_node
                key to the current node_min_dist
            4) continue until all unseen_nodes are visited
        5) set current node to the end node and work backwards from the end
            node in order to finally create the shortest path
        6) create path as list with end node as first value
        6) loop while there is a previous node (there won't be a previous node
            for the starting node)
            1) get the previous node from previous_node for the current node
            2) append the previous node to the path
            2) set the previous node as the current node
        7) reverse path to get path from start to end
        """

        # unseen_nodes = self.location_manager.grid.get_nodes_without_aisles_and_storage().copy()

        cdef:
            pair[int, int] node_min_dist
            float min_dist
            vector[pair[int, int]] adjacent_nodes
            float dist_btwn_adj_node_and_cur_node
            float alternate_path_dist
            int i

            map[pair[int, int], pair[int, int]] previous_node
            map[pair[int, int], float] distance
            set[pair[int, int]] visited
            vector[pair[int, int]] explored_nodes

        explored_nodes.push_back(start)
        previous_node[start] = pair[int, int](-1, -1)
        distance[start] = 0.
        while True:  # while there are still unseen_nodes in list
            node_min_dist = pair[int, int](-1, -1)
            min_dist = float('inf')
            for i in range(explored_nodes.size()):
                if visited.find(explored_nodes[i]) == visited.end() and distance[explored_nodes[i]] < min_dist:
                    min_dist = distance[explored_nodes[i]]
                    node_min_dist = explored_nodes[i]

            if node_min_dist.first == -1:
                break

            visited.insert(node_min_dist)

            adjacent_nodes = get_adjacent(node_min_dist, nodes)

            for i in range(adjacent_nodes.size()):
                dist_btwn_adj_node_and_cur_node = compute_distance(node_min_dist, adjacent_nodes[i]) * unit_distance
                alternate_path_dist = distance[node_min_dist] + \
                                      dist_btwn_adj_node_and_cur_node
                if distance.find(adjacent_nodes[i]) == distance.end():
                    distance[adjacent_nodes[i]] = float('inf')
                    explored_nodes.push_back(adjacent_nodes[i])
                if alternate_path_dist < distance[adjacent_nodes[i]]:
                    distance[adjacent_nodes[i]] = alternate_path_dist
                    previous_node[adjacent_nodes[i]] = node_min_dist

        return previous_node, distance