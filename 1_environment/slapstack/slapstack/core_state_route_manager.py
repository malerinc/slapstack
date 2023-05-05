from copy import deepcopy
from os.path import exists
from collections import deque
import numpy as np
import pandas as pd

from scipy.sparse.csgraph import floyd_warshall
from slapstack.core_state_lane_manager import TileAccessPointIndex, LaneManager
from slapstack.helpers import faster_deepcopy, StorageKeys, PathKeys
from slapstack.helpers import ravel2, unravel2


class RouteManager:
    def __init__(self, lane_manager: LaneManager, storage_matrix: np.ndarray,
                 speed: float, unit_distance: float,
                 use_case_name: str = 'random'):
        self.lane_manager = lane_manager
        self.distance_dictionary = dict()
        layout = storage_matrix[:, :, 0]
        routing_grid = ((layout == StorageKeys.MID_AISLE) |
                        (layout == StorageKeys.SINK) |
                        (layout == StorageKeys.SOURCE))
        self.dims = layout.shape
        self.s = storage_matrix
        # os.remove('distance_mtrix.npy')
        dm_name = f'distance_matrix_{use_case_name}.npy'
        pred_name = f'predecessors_{use_case_name}.npy'
        if exists(dm_name) and exists(pred_name):
            self.distance_matrix = np.load(dm_name)
            self.predecessors = np.load(pred_name)
        else:
            adjacency_matrix = RouteManager.__get_adjacency(
                routing_grid.astype('int'))
            self.distance_matrix, self.predecessors = floyd_warshall(
                adjacency_matrix, return_predecessors=True, unweighted=True,
                directed=False)
            # some small tests
            n_dist = np.argwhere(self.distance_matrix != np.inf).shape[0]
            n_adj = adjacency_matrix.sum()
            assert n_dist > n_adj
            assert (pd.DataFrame(self.distance_matrix).replace(
                np.inf, -1).max() > layout.shape[0]).any()
            np.save(dm_name, self.distance_matrix)
            np.save(pred_name, self.predecessors)
        # self.grid = Grid(storage_matrix)
        self.speed = speed
        self.unit_distance = unit_distance

    @staticmethod
    def __get_adjacency(im: np.ndarray):
        """
        Computes the adjacency of the warehouse layout given a binary matrix of
        routable points.

        :param im: The binary grid.
        :return: The grid adjacency matrix in dense format.
        """
        nonzeros = np.argwhere(im == 1)
        adjacency = np.zeros((im.size, im.size))
        for nonzero in nonzeros:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    neigh_x, neigh_y = nonzero[0] + i, nonzero[1] + j
                    if (0 <= neigh_x < im.shape[0]
                            and 0 <= neigh_y < im.shape[1]
                            and im[neigh_x, neigh_y] == 1):
                        pts1 = ravel2((neigh_x, neigh_y), im.shape)
                        pts2 = ravel2((nonzero[0], nonzero[1]), im.shape)
                        adjacency[pts1, pts2] = 1
                        adjacency[pts1, pts2] = 1
        n_adj = adjacency.sum()
        n_nodes = nonzeros.shape[0]
        assert n_adj > n_nodes
        return adjacency

    def add_path_to_distance_dictionary(self, start, end, indices,
                                        distance, duration):
        path_parameters = {'indices': deepcopy(indices),
                           'distance': distance,
                           'duration': duration}
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

    def get_aisle_distance(self, src, dest):
        src_idx = ravel2(src, self.dims)
        dest_idx = ravel2(dest, self.dims)
        return self.distance_matrix[src_idx, dest_idx]

    def get_ap_distance(self, tile):
        return self.lane_manager.tile_access_points[tile].distance

    def get_distance(self, src, tgt):
        return Route(self, src, tgt, distance_only=True).distance

    def get_neirest_neighbor(self):
        pass


class Route:
    """
    Route is a class used to create lists of coordinates that AGVs can use
    to safely and efficiently drive around the warehouse. Route consists of
    one or more paths. All tiles except for walls can be driven
    on. If a tile has pallets on it, it just has a drive time penalty. That
    means if an AGV wants to drive through a pallet, it can do so, but it will
    take longer than if it drove around
    """

    def __init__(self, route_manager: RouteManager, start, end,
                 distance_only=False):
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
        self.route_manager = route_manager
        self.start = tuple(start)
        self.end = tuple(end)
        self.route_type = self.get_route_type()
        self.aisle_access = self.get_aisle_access()
        if distance_only:
            self.distance = self.get_distance()
        else:
            self.path = self.get_path()
            self.midpoints = self.__set_midpoints()
            if len(self.path.indices) > 0:
                assert self.path.indices[0] == self.end
                assert self.path.indices[-1] == self.start

    def get_distance(self):
        """
        Get distance only using the distance dictionary and the access point
        dictionary.

        :return: The distance between any points in the warehouse.
        """
        if self.route_type == PathKeys.STORAGE_TO_STORAGE:
            if self.aisle_access[0] != self.aisle_access[1]:  # change lanes
                # PathKeys.AISLE_TO_AISLE --> main path
                d1 = self.route_manager.get_aisle_distance(
                    self.aisle_access[0], self.aisle_access[1])
                # PathKeys.STORAGE_TO_AISLE --> left
                d2 = self.route_manager.get_ap_distance(self.start)
                #  PathKeys.AISLE_TO_STORAGE --> right
                d3 = self.route_manager.get_ap_distance(self.end)
                return d1 + d2 + d3
            else:  # only vertical difference
                return np.hypot(
                    self.start[0] - self.end[0],
                    self.start[0] - self.end[0]
                )
        elif self.route_type == PathKeys.STORAGE_TO_ACCESS:
            # PathKeys.STORAGE_TO_AISLE
            d1 = self.route_manager.get_aisle_distance(
                self.aisle_access[0], self.end)
            # PathKeys.AISLE_TO_ACCESS
            d2 = self.route_manager.get_ap_distance(self.start)
            return d1 + d2
        elif self.route_type == PathKeys.ACCESS_TO_STORAGE:
            # PathKeys.ACCESS_TO_AISLE
            d1 = self.route_manager.get_aisle_distance(
                self.start, self.aisle_access[0])
            # PathKeys.AISLE_TO_STORAGE
            d2 = self.route_manager.get_ap_distance(self.end)
            return d1 + d2
        elif self.route_type == PathKeys.ACCESS_TO_ACCESS:
            return self.route_manager.get_aisle_distance(self.start, self.end)

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

    def get_path(self):
        """depending on the route type, one to three paths are created and
        added to the route. defines what type of paths are needed"""
        path = None
        if self.route_type == PathKeys.STORAGE_TO_STORAGE:
            assert self.aisle_access
            if self.aisle_access[0] != self.aisle_access[1]:  # change lanes
                path = Path(self.route_manager)
                # PathKeys.AISLE_TO_AISLE --> main path
                path.add_aisle_path(self.aisle_access[0], self.aisle_access[1])
                # PathKeys.STORAGE_TO_AISLE --> left
                path.add_lane_path(self.start, self.aisle_access[0], left=False)
                #  PathKeys.AISLE_TO_STORAGE --> right
                path.add_lane_path(self.aisle_access[1], self.end, left=True)
            else:  # if agv can stay in same lane
                path = Path(self.route_manager)
                # PathKeys.STORAGE_TO_STORAGE
                path.add_lane_path(self.start, self.end, left=False)
        elif self.route_type == PathKeys.STORAGE_TO_ACCESS:
            assert self.aisle_access
            path = Path(self.route_manager)
            # PathKeys.STORAGE_TO_AISLE
            path.add_aisle_path(self.aisle_access[0], self.end)
            # PathKeys.AISLE_TO_ACCESS
            path.add_lane_path(self.start, self.aisle_access[0], left=False)
        elif self.route_type == PathKeys.ACCESS_TO_STORAGE:
            assert self.aisle_access
            path = Path(self.route_manager)
            # PathKeys.ACCESS_TO_AISLE
            path.add_aisle_path(self.start, self.aisle_access[0])
            # PathKeys.AISLE_TO_STORAGE
            path.add_lane_path(self.aisle_access[0], self.end, left=True)
        elif self.route_type == PathKeys.ACCESS_TO_ACCESS:
            path = Path(self.route_manager)
            path.add_aisle_path(self.start, self.end)
        # if len(path.indices) == 0: pallet gets retrieved immediately after
        # delivery
        # assert path.indices
        if len(path.indices) == 0:
            assert self.start == self.end
        if len(path.indices) != 0 and path.indices[0] != self.end:
            path.indices.appendleft(self.end)
        return path

    def get_first_node(self):
        """
        Returns first node in the route. Note that since the route indices
        are a deque, the last node corresponds to the element at position -1.

        :return: The first tile in the route.
        """
        return self.get_indices()[-1]

    def get_last_node(self):
        """
        Returns last node in the route. Note that since the route indices
        are a deque, the last node corresponds to the element at position 0.

        :return: The last tile in the route.
        """
        return self.get_indices()[0]

    def get_indices(self):
        return self.path.indices

    def get_total_distance(self):
        return self.path.distance

    def get_duration(self):
        """return route duration"""
        return self.path.duration

    def __set_midpoints(self):
        """return route duration"""
        route_midpoints = np.arange(len(self.path.indices) + 1, 1, -1.0)
        time_unit = self.route_manager.unit_distance / self.route_manager.speed
        route_midpoints *= time_unit
        route_midpoints += time_unit / 2
        return deque(route_midpoints)

    def update_path(self, elapsed_time: float):
        """
        Updates the route of an AGV depending on the time elapsed since the last
        update. Tiles already traveled are removed from the route.

        If the AGV has arrived at its destination but has not
        finished handling the pallets (see material_handling_time), its
        destination tile is kept in place.

        :param elapsed_time: The time since the last simulation event.
        :return: None.
        """
        n_tiles = len(self.midpoints)
        i = 0
        while i < n_tiles:
            if self.midpoints[-1] <= elapsed_time and len(self.midpoints) > 1:
                i += 1
                self.path.indices.pop()
                self.midpoints.pop()
            else:
                # pointless to keep iterating since midpoint times are
                # monotonously increasing ;)
                break

    def __str__(self):
        return '{0}'.format(self.get_indices())

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)


class Path:
    """
    Type can take on values of access_to_access, access_to_aisle,
    aisle_to_access, storage_to_aisle, and aisle_to_storage.
    """
    def __init__(self, route_manager: RouteManager):
        # TODO: move to route manager
        self.route_manager = route_manager
        self.indices = deque()
        self.distance = 0
        self.duration = 0

    def get_duration(self):
        return self.duration

    def add_aisle_path(self, start, end):
        """get dijkstra path from routing.distance_dictionary if it has been
        calculated once before, otherwise perform dijkstra's algorithm,
        calculate it, and add it to routing.distance_dictionary"""
        assert len(self.indices) == 0
        if self.route_manager.path_in_distance_dictionary(
                start, end):
            indices = self.route_manager.distance_dictionary[
                start][end]['indices']
            self.indices = deepcopy(indices)
            self.distance = self.route_manager.distance_dictionary[
                start][end]['distance']
            self.duration = self.route_manager.distance_dictionary[
                start][end]['duration']
        else:
            # TODO: calculate distance independent of speed!
            dims = self.route_manager.dims
            node = ravel2(end, dims)
            start_node_r = ravel2(start, dims)
            self.indices = deque([end])  # indices go from finish to start!!
            distance = self.route_manager.distance_matrix[start_node_r, node]
            self.duration = (distance * self.route_manager.unit_distance
                             / self.route_manager.speed)
            self.distance = distance
            if distance != 0:
                predecessor = self.route_manager.predecessors[
                    start_node_r, node]
                while predecessor != start_node_r:
                    self.indices.append(unravel2(predecessor, dims))
                    predecessor = self.route_manager.predecessors[
                        start_node_r, predecessor]
                self.indices.append(start)
            self.route_manager.add_path_to_distance_dictionary(
                start, end, deepcopy(self.indices), self.distance, self.duration
            )

    def add_lane_path(self, start, end, left=False):
        """creates a simple path that goes from an aisle access tile to a
        desired storage tile in the same lane"""
        if left:
            starting_row = end[0]
            ending_row = start[0]
        else:
            starting_row = start[0]
            ending_row = end[0]
        direction = 1 if starting_row < ending_row else -1
        distance = 0
        # create path with last tile left and first tile right
        for r in range(ending_row - direction,
                       starting_row - direction, -direction):
            next_row = r
            next_node = (next_row, start[1])
            distance += self.route_manager.unit_distance
            if len(self.indices) > 0 and next_node == self.indices[-1]:
                continue
            if left:
                self.indices.appendleft(next_node)
            else:
                self.indices.append(next_node)
        self.distance += distance
        self.duration += distance / self.route_manager.speed

    def get_indices(self):
        return self.indices

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)
