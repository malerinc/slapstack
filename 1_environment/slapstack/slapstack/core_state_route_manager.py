from math import hypot, inf
from typing import Tuple, Dict, Set
from collections import OrderedDict
import numpy as np
import sys
import traceback

from slapstack.core_state_lane_manager import TileAccessPointIndex, LaneManager

try:
    from slapstack.extensions.c_dijkstra import perform_cdijkstra, \
        perform_cdijkstra_without_end_node
except ModuleNotFoundError as e:
    print(traceback.format_exc(), file=sys.stderr)
    print("Error: cythonized extensions not found under slapstack.extensions",
          file=sys.stderr)
    exit(0)

from slapstack.helpers import faster_deepcopy, StorageKeys, PathKeys


class RouteManager:
    """
    tile_access_points: dict
        keys are storage tile locations, values are the location of the
        vertically closest aisle.
        if direction is above then the storage tile is above the aisle,
        i.e. the row of the storage tile is a lower number the the row of the
        aisle.
        example =   {
                       (16, 6): {'aisle': (10, 6),
                                  'distance': 6,
                                  'direction' : above},
                                  ...}

    lane_clusters: dict
        keys are middle aisle indices, values are clusters of tiles in one lane
        separated by the direction relative to the middle aisle. lists are
        sorted by distance to aisle.
        if the direction says below, that means the lane is below the aisle,
        i.e. the rows of the lane are a higher number than the row of the aisle
        direction =
        example =  {
                        (10, 6): {
                                    'below': [(13, 6), (12, 6), ...],
                                    'above': [(8, 6), (7, 6), ...]
                                    },
                        (10, 7): {
                                    'above': [(13, 7), (12, 7), ...],
                                    'below': [(8, 7), (7, 7), ...]
                                    },
                    ...}
    distance_dictionary: dict(dict(dict))
        keys in 1st level are starting points of a route, keys in 2nd level are
        ending points of a route, keys in 3rd level are inpt desired
        (either path or distance)
        example =    {
                        source1 : {
                            aisle1 : {
                                path: [a, b, c],
                                dist: 3,
                                edge_time_midpoints: []
                                },
                            aisle2: {
                                path: [a, b, c, d],
                                dist: 4
                                }
                            },
                        source2: {
                            aisle1 : {
                                path: [b, c],
                                dist: 2
                                },
                            aisle2: {
                                path: [b, c, d],
                                dist: 3
                                }
                            }
                    }


    """
    def __init__(self, lane_manager: LaneManager, storage_matrix: np.ndarray,
                 speed: float, unit_distance: float):
        self.lane_manager: LaneManager = lane_manager
        self.distance_dictionary = dict()
        self.grid = Grid(storage_matrix)
        self.speed = speed
        self.unit_distance = unit_distance

    def add_path_to_distance_dictionary(self, start, end, indices, distance,
                                        duration, midpoints):
        path_parameters = {'indices': indices, 'distance': distance,
                           'duration': duration, 'midpoints': midpoints}
        value_to_add = {end: path_parameters}
        # if the starting point is in the dictionary already,
        # make a new key with the end point
        if start in self.distance_dictionary:
            self.distance_dictionary[start][end] = path_parameters
        else:  # if the start point is not in the dictionary already, add it
            self.distance_dictionary[start] = value_to_add

    def path_in_distance_dictionary(self, start, end):
        """checks if a dijkstra path has already been calculated"""
        if start in self.distance_dictionary:
            if end in self.distance_dictionary[start]:
                return True
            else:
                return False
        else:
            return False

    def precompute_paths(self, landmark_locations):
        """
        Performs dijkstra from the locations passed as the landmark_location
        parameter to all aisle, source and sink node storing all intermediary
        paths.

        For all locations all the paths and the associated distances found
        during the call to perform_cdijkstra_without_end_node are returned. The
        reversed paths are stored implicitly in a dictionary mapping warehouse
        routable nodes to their predecessor in the path. The distance to the
        predecessor is stored in a corresponding distance dictionary.

        For all routable nodes in present in warehouse and the predecessor
        mapping, the reverse route to the current location is reversed, and all
        its segments cached in the distance_dictionary.

        :param landmark_locations: A list of routable nodes (aisle, source or
        sink positions).

        :return: None.
        """
        nodes = self.grid.get_nodes_without_aisles_and_storage()
        for location in landmark_locations:
            if location not in self.distance_dictionary:
                self.distance_dictionary[location] = {}
            results = perform_cdijkstra_without_end_node(
                location, nodes, self.unit_distance)
            # minimum distances from location to all other nodes; the
            previous_node_map: Dict[(int, int), (int, int)] = results[0]
            distance_from_landmark: Dict[(int, int), float] = results[1]
            for node in nodes:
                if (node not in previous_node_map
                        or (location in self.distance_dictionary
                            and node in self.distance_dictionary[location])):
                    continue
                path = [node]
                distances = [distance_from_landmark[node]]
                while node != location:
                    node = previous_node_map[node]
                    path.append(node)
                    distances.append(distance_from_landmark[node])
                path = list(reversed(path))
                distances = list(reversed(distances))
                midpoints = []
                i = len(path) - 1
                for i in range(1, len(path)):
                    path_till_i = path[:i+1]
                    dist = distances[i]
                    midpoints.append((distances[i - 1] + distances[i])
                                     / (2 * self.speed))
                    self.add_path_to_distance_dictionary(
                        start=location,
                        end=path[i],
                        indices=path_till_i,
                        distance=dist,
                        duration=dist / self.speed,
                        midpoints=midpoints.copy()
                    )


class Grid(object):
    """a grid object is created upon initialization of the state cache. its
    main features is a set of nodes, where a node is a 2D tuple that represents
    a tile in the warehouse. the grid is used to make paths in the simulation.

    nodes: dictionary
        key: position: Tuple[int, int]
        value: number of obstacles: int
    """

    def __init__(self, storage_matrix: np.ndarray):
        self.n_rows, self.n_columns, self.stack_size = storage_matrix.shape
        # {(1, 1): 0, (1, 2): 0,..., (2,2): 2}
        # key is index, value is n_pallets
        nfo = Grid.create_nodes_from_matrix(storage_matrix)
        self.nodes: Dict[Tuple[int, int], int] = nfo[0]
        self.nodes_without_aisles_and_storage: Set[Tuple[int, int]] = nfo[1]

    @staticmethod
    def create_nodes_from_matrix(storage_matrix: np.ndarray):
        """goes through each tile the storage matrix, creates a node, and adds
        it to self.nodes if it is a middle_aisle, empty, or aisle tile. This
        is used for paths created to traverse lanes. A node is added to
        nodes_routable if it is a middle_aisle, source, or
        sink. This is used for paths created to navigate the warehouse and is
        used in dijkstra's algorithm"""
        nodes = {}
        nodes_routable = set()
        storage_matrix = storage_matrix[:, :, 0]
        for i, row in enumerate(storage_matrix):
            for j, elem in enumerate(row):
                if elem == StorageKeys.MID_AISLE or \
                        elem == StorageKeys.EMPTY or \
                        elem == StorageKeys.AISLE:
                    nodes[(i, j)] = 0
                if elem == StorageKeys.MID_AISLE or \
                        elem == StorageKeys.SOURCE or \
                        elem == StorageKeys.SINK:
                    nodes_routable.add((i, j))
        return nodes, nodes_routable

    def add_obstacle(self, node: Tuple[int, int]):
        """adds 1 obstacle to a node. increases the value of the self.nodes
        item with the given key by 1.
        """
        self.nodes[node] = self.nodes[node] + 1
        assert self.nodes[node] <= self.stack_size

    def remove_obstacle(self, node: Tuple[int, int]):
        """removes 1 obstacle from a node. decreases value of the self.nodes
        item with the given key by 1
        """
        self.nodes[node] = self.nodes[node] - 1
        assert self.nodes[node] >= 0

    def get_adjacent(self, node: Tuple[int, int]):
        """this function is used in the path-finding algorithm. input is a node
        and it returns a set of nodes that are adjacent to """
        adjacent_nodes = set()
        # adjacent_nodes = {}
        row = node[0]
        column = node[1]

        if row < 0 or row >= self.n_rows or column < 0 or column >= \
                self.n_columns:
            raise Exception("Out of bounds")
        relative_positions = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]
        for index, rel_pos in enumerate(relative_positions[0:-1]):
            #  if rel_pos is within bounds of graph
            adjacent_node_row = row + rel_pos[0]
            adjacent_node_column = column + rel_pos[1]
            adjacent_node_index = (adjacent_node_row, adjacent_node_column)

            if adjacent_node_index in self.nodes_without_aisles_and_storage:
                adjacent_nodes.add(adjacent_node_index)
        return adjacent_nodes

    def adjacent_nodes_arent_occupied(self, index, relative_positions, node):
        """this function prevents diagonal paths from being created between two
        occupied nodes
        """
        previous_node_row = node[0] + relative_positions[index - 1][0]
        previous_node_column = node[1] + relative_positions[index - 1][1]
        previous_node_index = (previous_node_row, previous_node_column)
        next_node_row = node[0] + relative_positions[index + 1][0]
        next__node_column = node[1] + relative_positions[index + 1][1]
        next_node_index = (next_node_row, next__node_column)
        if previous_node_index in self.nodes and next_node_index in self.nodes:
            if self.nodes[previous_node_index] and self.nodes[next_node_index]:
                return False
            else:
                return True
        else:
            return True

    def char_map(self):
        """returns a printable string that represents the graph and path"""
        arr = [["-".center(5)] * self.n_columns for _ in
               range(self.n_rows)]

        for node in self.get_nodes().items():
            index = node[0]
            n_pallets = node[1]
            if n_pallets > 0:
                arr[index[0]][index[1]] = str(n_pallets).center(5)

        return arr

    def get_nodes(self):
        return self.nodes

    def get_nodes_without_aisles_and_storage(self):
        return {i: 0 for i in self.nodes_without_aisles_and_storage}

    def __str__(self):
        return "\n".join("".join(x) for x in self.char_map())

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)


class Route:
    """
    Route is a class used to create lists of coordinates that AGVs can use
    to safely and efficiently drive around the warehouse. Route consists of
    one or more paths. All tiles except for walls can be driven
    on. If a tile has pallets on it, it just has a drive time penalty. That
    means if an AGV wants to drive through a pallet, it can do so, but it will
    take longer than if it drove around
    """

    def __init__(self, route_manager: RouteManager, grid: Grid,
                 start, end, through_pallet_weight=1):
        """
        grid: Grid
        start: Tuple[int,int]
            starting position of AGV
        end: Tuple[int,int]
            desired ending position of AGV
        through_pallet_weight: float
            parameter that scales how big of a delay driving through pallets
            is
        """
        assert through_pallet_weight >= 0
        self.route_manager = route_manager
        self.grid = grid
        self.through_pallet_weight = through_pallet_weight
        self.start = tuple(start)
        self.end = tuple(end)
        self.route_type = self.get_route_type()
        self.aisle_access = self.get_aisle_access()
        self.route = self.get_route()
        self.indices = self._get_indices()
        self.midpoints = self._get_midpoints()
        assert len(self.indices) - 1 == len(self.midpoints)

    def get_route_type(self):
        """
        Depending on where the route starts and ends and if it needs to go
        from a storage tile to an aisle tile, it is given a different route type

        :return: TODO!
        """
        route_type = None
        r: RouteManager = self.route_manager
        # storage tile access points ;)
        stap: TileAccessPointIndex = r.lane_manager.tile_access_points
        if self.start in stap and self.end in stap:
            route_type = PathKeys.STORAGE_TO_STORAGE
        elif self.start in stap and self.end not in stap:
            route_type = PathKeys.STORAGE_TO_ACCESS
        elif self.start not in stap and self.end in stap:
            route_type = PathKeys.ACCESS_TO_STORAGE
        elif self.start not in stap and self.end not in stap:
            route_type = PathKeys.ACCESS_TO_ACCESS
        assert route_type
        return route_type

    def get_aisle_access(self):
        """if the route needs to have a turning point at an aisle tile, it is
        found here using the routing.closet_aisle dictionary"""
        aisle_access_points = []
        r: RouteManager = self.route_manager
        # storage tile access points ;)
        stap: TileAccessPointIndex = r.lane_manager.tile_access_points
        if self.route_type == PathKeys.STORAGE_TO_STORAGE:
            aisle_pos_s = stap[self.start].position
            aisle_pos_e = stap[self.end].position
            aisle_access_points.append(aisle_pos_s)
            aisle_access_points.append(aisle_pos_e)
        elif self.route_type == PathKeys.STORAGE_TO_ACCESS:
            aisle_pos_s = stap[self.start].position
            aisle_access_points.append(aisle_pos_s)
        elif self.route_type == PathKeys.ACCESS_TO_STORAGE:
            aisle_pos_e = stap[self.end].position
            aisle_access_points.append(aisle_pos_e)
        elif self.route_type == PathKeys.ACCESS_TO_ACCESS:
            pass
        return aisle_access_points

    def get_route(self):
        """depending on the route type, one to three paths are created and
        added to the route. defines what type of paths are needed"""
        route = []
        if self.route_type == PathKeys.STORAGE_TO_STORAGE:
            assert self.aisle_access
            if self.start[1] != self.end[1]:  # if agv has to change lanes
                path1 = Path(
                    self.route_manager, self.grid, PathKeys.STORAGE_TO_AISLE,
                    self.start, self.aisle_access[0], 0,
                    through_pallet_weight=self.through_pallet_weight)
                path2 = Path(
                    self.route_manager, self.grid, PathKeys.AISLE_TO_AISLE,
                    self.aisle_access[0], self.aisle_access[1],
                    path1.get_duration(),
                    through_pallet_weight=self.through_pallet_weight)
                path3 = Path(
                    self.route_manager, self.grid, PathKeys.AISLE_TO_STORAGE,
                    self.aisle_access[1], self.end, path2.get_duration(),
                    through_pallet_weight=self.through_pallet_weight)
                route.extend([path1, path2, path3])
            elif self.start[1] == self.end[1]:  # if agv can stay in same lane
                path = Path(
                    self.route_manager, self.grid, PathKeys.STORAGE_TO_STORAGE,
                    self.start, self.end, 0,
                    through_pallet_weight=self.through_pallet_weight)
                route.extend([path])
        elif self.route_type == PathKeys.STORAGE_TO_ACCESS:
            assert self.aisle_access
            path1 = Path(
                self.route_manager, self.grid, PathKeys.STORAGE_TO_AISLE,
                self.start, self.aisle_access[0], 0,
                through_pallet_weight=self.through_pallet_weight)
            path2 = Path(
                self.route_manager, self.grid, PathKeys.AISLE_TO_ACCESS,
                self.aisle_access[0], self.end, path1.get_duration(),
                through_pallet_weight=self.through_pallet_weight)
            route.extend([path1, path2])
        elif self.route_type == PathKeys.ACCESS_TO_STORAGE:
            assert self.aisle_access
            path1 = Path(
                self.route_manager, self.grid, PathKeys.ACCESS_TO_AISLE,
                self.start, self.aisle_access[0], 0,
                through_pallet_weight=self.through_pallet_weight)
            path2 = Path(
                self.route_manager, self.grid, PathKeys.AISLE_TO_STORAGE,
                self.aisle_access[0], self.end, path1.get_duration(),
                through_pallet_weight=self.through_pallet_weight)
            route.extend([path1, path2])
        elif self.route_type == PathKeys.ACCESS_TO_ACCESS:
            route.append(
                Path(self.route_manager, self.grid, PathKeys.ACCESS_TO_ACCESS,
                     self.start, self.end, 0,
                     through_pallet_weight=self.through_pallet_weight))
        assert route
        return route

    def get_first_node(self):
        """returns first node in the route"""
        return self.get_indices()[0]

    def get_last_node(self):
        """returns last node in the route"""
        return self.get_indices()[-1]

    def get_indices(self):
        return self.indices

    def _get_indices(self):
        """returns list of nodes in route"""
        indices = []
        for path in self.route:
            indices.extend(path.get_indices())
        indices = list(OrderedDict.fromkeys(indices))  # removes duplicates
        return indices

    def get_total_distance(self):
        total_distance = 0
        for path in self.route:
            total_distance = total_distance + path.total_distance
        return total_distance

    def get_duration(self):
        """return route duration"""
        duration = 0
        for path in self.route:
            duration = duration + path.duration
        return duration

    def get_unweighted_duration(self):
        """return route duration"""
        unweighted_duration = 0
        for path in self.route:
            unweighted_duration += path.unweighted_duration
        return unweighted_duration

    def _get_midpoints(self):
        """return route duration"""
        route_midpoints = []
        for path in self.route:
            for path_midpoints in path.edge_time_midpoints:
                route_midpoints.append(path_midpoints)
        return route_midpoints

    def update_route(self, time_to_simulate: float):
        for i, midpoint in enumerate(self.midpoints):
            if time_to_simulate >= midpoint:
                del self.indices[0]
                del self.midpoints[0]

    def __str__(self):
        return '{0}'.format(self.get_indices())

    def plot_route(self):
        """returns string to print route on a grid"""
        grid_character_arr = self.grid.char_map()
        for node in self.get_indices():
            existing_elem = grid_character_arr[node[0]][node[1]][2]
            grid_character_arr[node[0]][node[1]] = "({0})".format(
                existing_elem).center(5)
        return "\n".join("".join(x) for x in grid_character_arr)

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)


class Path:
    """
    Type can take on values of access_to_access, access_to_aisle,
    aisle_to_access, storage_to_aisle, and aisle_to_storage.
    """
    def __init__(self, route_manager: RouteManager, grid: Grid, path_type: str,
                 start, end, starting_time, through_pallet_weight=10):
        # TODO: move to route manager
        self.through_pallet_weight = through_pallet_weight
        self.route_manager = route_manager
        self.grid = grid
        self.type = path_type
        self.start = start
        self.end = end
        self.starting_time = starting_time
        path_components = self.create_path()
        self.indices = path_components[0]
        self.total_distance = path_components[1]
        self.duration = path_components[2]
        self.edge_time_midpoints = path_components[3]
        self.unweighted_duration = path_components[4]
        assert self.start
        assert self.end

    def get_unweighted_duration(self):
        return self.unweighted_duration

    def get_duration(self):
        return self.duration

    def create_path(self):
        """gets (or creates) a dijkstra path for certain types of paths and
        creates a lane traversal path for other types. indices, total distance,
        total duration, and edge time midpoints are all returned as inpt
        of a path."""
        indices, distance, duration, midpoints = None, None, None, None
        unweighted_duration = None
        if (self.type == PathKeys.ACCESS_TO_ACCESS
                or self.type == PathKeys.ACCESS_TO_AISLE
                or self.type == PathKeys.AISLE_TO_ACCESS
                or self.type == PathKeys.AISLE_TO_AISLE):
            indices, distance, duration, midpoints = self.get_dijkstra_path()
            unweighted_duration = duration
        elif (self.type == PathKeys.STORAGE_TO_AISLE
              or self.type == PathKeys.AISLE_TO_STORAGE
              or self.type == PathKeys.STORAGE_TO_STORAGE):
            lane_traversal_info = self.traverse_lane()
            indices = lane_traversal_info[0]
            distance = lane_traversal_info[1]
            duration = lane_traversal_info[2]
            midpoints = lane_traversal_info[3]
            unweighted_duration = lane_traversal_info[4]
        assert indices
        midpoints = [m + self.starting_time for m in midpoints]
        return indices, distance, duration, midpoints, unweighted_duration

    def get_dijkstra_path(self):
        """get dijkstra path from routing.distance_dictionary if it has been
        calculated once before, otherwise perform dijkstra's algorithm,
        calculate it, and add it to routing.distance_dictionary"""
        if self.route_manager.path_in_distance_dictionary(
                self.start, self.end):
            indices = self.route_manager.distance_dictionary[
                self.start][self.end]['indices']
            distance = self.route_manager.distance_dictionary[
                self.start][self.end]['distance']
            duration = self.route_manager.distance_dictionary[
                self.start][self.end]['duration']
            midpoints = self.route_manager.distance_dictionary[
                self.start][self.end]['midpoints']
        else:
            # TODO: calculate distance independent of speed!
            indices, distance, duration, midpoints = perform_cdijkstra(
                self.start,
                self.end,
                self.grid.get_nodes_without_aisles_and_storage(),
                self.route_manager.unit_distance,
                self.route_manager.speed,
            )
            # indices, distance, duration, midpoints = self.perform_dijkstra()
            self.route_manager.add_path_to_distance_dictionary(
                self.start, self.end, indices, distance, duration, midpoints
            )
        return indices, distance, duration, midpoints

    def traverse_lane(self):
        """creates a simple path that goes from an aisle access tile to a
        desired storage tile in the same lane"""
        starting_row = self.start[0]
        ending_row = self.end[0]
        direction = 1 if starting_row < ending_row else -1
        rows_to_traverse_through = list(range(starting_row,
                                              ending_row + 1*direction,
                                              direction))
        indices = []
        weighted_edge_distances = []
        unweighted_edge_distances = []
        edge_distance_midpoints = []
        indices.append((starting_row, self.start[1]))
        # create path and get weighted distances
        for r in range(len(rows_to_traverse_through)-1):
            next_row = rows_to_traverse_through[r+1]
            next_node = (next_row, self.start[1])
            next_node_weight = self.grid.get_nodes()[next_node]
            edge_length = 1
            weighted_edge_length = edge_length + (self.through_pallet_weight *
                                                  next_node_weight)
            weighted_edge_distance = (weighted_edge_length
                                      * self.route_manager.unit_distance)
            unweighted_edge_distance = (edge_length
                                        * self.route_manager.unit_distance)
            weighted_edge_distances.append(weighted_edge_distance)
            unweighted_edge_distances.append(unweighted_edge_distance)
            indices.append(next_node)

        edge_times = [(i / self.route_manager.speed)
                      for i in weighted_edge_distances]
        unweighted_edge_times = [
            (i / self.route_manager.speed) for i in unweighted_edge_distances
        ]

        for i in range(0, len(weighted_edge_distances)):
            edge_distance_midpoints.append(
                sum(weighted_edge_distances[0:i]) +
                (weighted_edge_distances[i] / 2))

        edge_time_midpoints = [
            (i / self.route_manager.speed) for i in edge_distance_midpoints]
        duration = sum(edge_times)
        unweighted_duration = sum(unweighted_edge_times)
        return (indices, sum(weighted_edge_distances), duration,
                edge_time_midpoints, unweighted_duration)

    def perform_dijkstra(self):  # dijkstra
        """
        DEPRECATED! Function was moved to the dedicated package and cythonized.

        dijkstra's algorithm to find shortest path
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
        unseen_nodes = (self.grid
                        .get_nodes_without_aisles_and_storage().copy())
        #  dictionary with shortest distances to every node
        #  initially set to infinity
        distance = {node: inf for node in unseen_nodes}
        #  initialize dictionary with no previous node for every node
        previous_node = {node: None for node in unseen_nodes}

        distance[self.start[0], self.start[1]] = 0
        while unseen_nodes:  # while there are still unseen_nodes in list
            # TODO: Use heap instead of a list for the storage and popping of
            #  unseen nodes.
            node_min_dist = min(unseen_nodes, key=distance.__getitem__)
            unseen_nodes.pop(node_min_dist)
            adjacent_nodes = self.grid.get_adjacent(node_min_dist)
            for adjacent_node in adjacent_nodes:
                dist_btwn_adj_node_and_cur_node = \
                    hypot(node_min_dist[0] - adjacent_node[0],
                          node_min_dist[1] - adjacent_node[1])
                alternate_path_dist = (distance[node_min_dist]
                                       + dist_btwn_adj_node_and_cur_node)
                if alternate_path_dist < distance[adjacent_node]:
                    distance[adjacent_node] = alternate_path_dist
                    previous_node[adjacent_node] = node_min_dist
        total_distance = distance[self.end]
        path = [self.end]
        current = self.end

        while previous_node[current[0], current[1]]:
            path.append(previous_node[current[0], current[1]])
            current = previous_node[current[0], current[1]]
        indices = [tuple(i) for i in list(reversed(path))]

        edge_distances = []
        for i in range(len(indices)-1):
            current_node = indices[i]
            next_node = indices[i+1]

            distance = hypot(current_node[0] - next_node[0], current_node[1]
                             - next_node[1]) * self.route_manager.unit_distance
            edge_distances.append(distance)

        edge_times = [i / self.route_manager.speed for i in edge_distances]

        edge_distance_midpoints = []
        for i in range(len(edge_distances)):
            edge_distance_midpoints.append(
                sum(edge_distances[0:i]) + (edge_distances[i] / 2))

        edge_time_midpoints = [(i/self.route_manager.speed) for i in
                               edge_distance_midpoints]

        duration = sum(edge_times)
        return indices, total_distance, duration, edge_time_midpoints

    def get_indices(self):
        return self.indices

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)
