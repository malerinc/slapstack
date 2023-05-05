from collections import deque
from typing import Tuple, Dict, List, Union, Set, Iterable, Deque

import numpy as np

from slapstack.helpers import AccessDirection, StorageKeys, ravel
from slapstack.interface_templates import SimulationParameters


class TileAccessPoint:
    def __init__(self,
                 position: Tuple[int, int],
                 distance: int,
                 direction: AccessDirection):
        """
        Structure representing the tile access point. It maintains the position
        of the access point, the distance from the tile and the direction in
        which a vehicle needs to travel from the storage tile to reach the
        aisle access point.

        :param position: The access point position.
        :param distance: The distance between tile and access point.
        :param direction: The orientation of the access point relative to tile.
        """
        self.position = position
        self.distance = distance
        self.direction = direction


class TileAccessPointIndex:
    """
    Maintains a dictionary indexed by storage locations with values
    corresponding to TileAccessPoint objects which in turn contain the tile
    access point position, the tile to access point distance and the direction
    in which a vehicle needs to travel to reach the access point from the tile.

    The dictionary is stored in the idx field.
    """
    def __init__(self, storage_matrix: np.ndarray):
        """
        Initializes an index of tile access points by iterating over all
        storage locations (on the first level) and matching individual locations
        with the closest middle aisle tile from the same column.

        :param storage_matrix: The three dimensional storage matrix S.
        """
        self.idx: Dict[Tuple[int, int], TileAccessPoint] = dict()

        # select ground locations
        storage_matrix = storage_matrix[:, :, 0]
        tiles = {
            (i[0], i[1]) for i in
            np.argwhere(storage_matrix == StorageKeys.EMPTY)
        }
        for tile in tiles:
            # get the aisle tiles in the column
            ap_x, ap_dist = TileAccessPointIndex.__get_access_point(
                storage_matrix, tile)
            assert storage_matrix[ap_x, tile[1]] == -5
            assert ap_x
            if tile[0] < ap_x:
                direction = AccessDirection.ABOVE
            else:
                direction = AccessDirection.BELOW
            self.idx[tile] = TileAccessPoint(
                position=(ap_x, tile[1]),
                distance=ap_dist,
                direction=direction)

    @staticmethod
    def __get_access_point(
            storage_matrix: np.ndarray, tile: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Finds the access point for a given storage tile. For now accessing the
        tiles in a lane is only possible from one direction (top bottom or
        bottom aisle).

        Finding the tile access point is tantamount to finding the closest
        MID_AISLE tile to the given storage location. Hence, we isolate the all
        the mid aisle points in the same column as the targeted tile, compute
        the distances to the tile along the x axis and return the colosest x
        position (row in the storage matrix) along with the associated distance.

        :param storage_matrix: The two dimensional layout matrix.
        :param tile: The tile for which to find the access point.
        :return: The row index of the access point along with the distance from
            the tile.
        """
        # retrieve all tiles in aisle
        access_points_x = (np.argwhere(
            storage_matrix[:, tile[1]] == StorageKeys.MID_AISLE)
                           .squeeze().tolist())
        if isinstance(access_points_x, int):
            access_points_x = np.array([access_points_x])
        # get closest aisle tile and its distance
        access_point_distances = np.abs((tile[0] - access_points_x))
        idx_closest = np.argmin(access_point_distances)
        return (access_points_x[idx_closest],
                access_point_distances[idx_closest])

    def __getitem__(self, tile_position: Tuple[int, int]) -> TileAccessPoint:
        """
        [] operator override. Retrieves the TileAccessPoint for the tile at the
        position passed as a parameter.

        :param tile_position: The position of the tile for which to retrieve
            the access point.
        :return: The indexed tile access point.
        """
        return self.idx[tile_position]

    def __iter__(self):
        """
        Returns an iterator over the tile access point collection by leveraging
        the dictionary structure of the idx field.

        :return: The iterator over the access index; calls to next will return
            (tile position, TileAccessPoint) pairs.
        """
        return iter(self.idx.items())

    def __contains__(self, tile_location):
        """
        Override for membership checks using the "in" operator.

        :param tile_location: The tile position to check membership for.
        :return: True if the tile is in the index and false otherwise.
        """
        return tile_location in self.idx


class Lane:
    def __init__(self, tiles: List[Tuple[int, int]]):
        self.tiles = tiles
        self.sku_pos: Dict[int, Deque[Tuple[int, int, int]]] = dict()

    def update_sku_border(self, sku, pos, added=True):
        # TODO: assertion to ensure that added tile is always below/above prev
        #  border
        if added:
            if sku in self.sku_pos:
                self.sku_pos[sku].append(pos)
            else:
                self.sku_pos[sku] = deque([pos])
        else:
            # when pallet shifting, the hole is plugged with the forward pallet
            # before removing the forward pallet; as such, the pallet to be
            # removed should be found at the position -2 in the stack
            stack_top_buff = deque([self.sku_pos[sku].pop()])
            while stack_top_buff[-1] != pos:
                stack_top_buff.append(self.sku_pos[sku].pop())
            pos_popped = stack_top_buff.pop()
            while stack_top_buff:
                self.sku_pos[sku].append(stack_top_buff.pop())
            # pos_popped = self.sku_pos[sku].pop()
            try:
                assert pos_popped == pos
            except AssertionError:
                print(pos_popped, pos)

    def get_border_tile(self, sku):
        return self.sku_pos[sku][-1]

    def __getitem__(self, key):
        return self.tiles[key]

    def __bool__(self):
        return True if len(self.tiles) else False

    def __len__(self):
        return len(self.tiles)


class LaneManager:
    def __init__(
            self, storage_matrix: np.ndarray, params: SimulationParameters):
        self.tile_access_points = TileAccessPointIndex(storage_matrix)
        self.lane_clusters, self.n_lanes = self.create_lane_clusters()
        self.locked_lanes = set()
        self.full_lanes = set()
        self.pure_lanes = params.pure_lanes
        self.sku_lanes: Dict[int, List[Tuple[int, int, int]]] = dict()
        self.lane_assigned = self.create_lane_assigned_dict()
        self.occupied_lanes = dict({})  # maps sku to lanes
        self.S = storage_matrix

    def lock_lane(self, location: Tuple[int, int]):
        """
        Identifies the lane corresponding to the passed tile, adds it to the
        locked lane dictionary and return the positions that are no accessible
        anymore.

        :param location: The location of the tile leading to the lane lock.
        :return: All locked positions in the lane.
        """
        ap_pos, ap_dir = self.locate_access_point(location)
        self.locked_lanes.add(ap_pos + (ap_dir.value,))
        return self.lane_clusters[ap_pos][ap_dir]

    def unlock_lane(self, delivery_action: Tuple[int, int, int],
                    tile: Tuple[int, int]):
        """
        Removes the lane from the locked_lanes set and possibly add it to the
        full lane set. The latter only happens if the lane unlock came as a
        result of a delivery action on the last open tile in the lane.

        :param delivery_action: The delivery lockation triple or None, if the
             unlock came as a result of a RetrievalFirstLeg having finished.
        :param tile: The tile indicative of the lane to unlock.
        :return: None.
        """
        ap_pos, ap_dir = self.locate_access_point(tile)
        self.locked_lanes.discard(ap_pos + (ap_dir.value,))
        # possibly add to filled land set
        if delivery_action:
            if self.__is_first_in_lane(delivery_action):
                self.full_lanes.add(ap_pos + (ap_dir.value,))
        else:
            self.full_lanes.discard(ap_pos + (ap_dir.value,))

    def __is_first_in_lane(self, position):
        """
        Checks whether the next position moving against the lane direction is
        in an aisle.

        :param position:
        :return: True if the next position is part of a different lan, false
        otherwise.
        """
        next_in_lane = self.__get_next_in_lane(position, -1)
        if self.S[next_in_lane] == StorageKeys.AISLE:
            return True
        return False

    def __get_next_in_lane(self, position, sign):
        x, y, z = position[0], position[1], position[2]
        direction = self.get_access_point_direction((x, y, z)) * sign
        next_in_lane = [x, y, z + 1]
        if self.level_too_high(next_in_lane):
            next_in_lane[2] = 0  # z
            next_in_lane[0] = next_in_lane[0] + direction  # x
        return tuple(next_in_lane)

    def level_too_high(self, position):
        """if the z-coordinate (stack level) is higher than what the warehouse
        is set up for, returns True"""
        if position[2] > self.S.shape[2] - 1:
            return True
        else:
            return False

    def create_lane_assigned_dict(self):
        lane_assigned = dict()
        for key, value in self.lane_clusters.items():
            if value[AccessDirection.ABOVE]:
                lane_assigned[key + (-1,)] = False
            if value[AccessDirection.BELOW]:
                lane_assigned[key + (1,)] = False
        return lane_assigned

    def add_lane_assignment(self, loc: Tuple[int, int, int], sku: int):
        """
        Updates the sku to lane mapping whenever a new pallet is delivered. If
        the sku was delivered to a lane where it was not present before, that
        lane will be marked and reserved for (mainly!) this type of sku.

        :param loc: The storage location that was chose for a delivery.
        :param sku: The sku of the delivered pallet.
        :return: None.
        """
        ap_pos, ap_dir = self.locate_access_point(loc[0:2])
        lane_key = ap_pos + (ap_dir.value,)
        if not self.lane_assigned[lane_key]:
            self.lane_assigned[lane_key] = True
            if sku in self.sku_lanes:
                self.sku_lanes[sku].add(lane_key)
            else:
                self.sku_lanes[sku] = {lane_key}

    def create_lane_clusters(self) -> (
            Dict[Tuple[int, int],
                 Dict[AccessDirection, List[Tuple[int, int]]]],
            int):
        """ creates a dictionary that groups storage tiles in a single lane
        into separate lists"""
        lane_clusters_dict = dict()
        lane_clusters_dict_with_keys = dict()
        # count number of lanes (note that one lane goes across an aisle)
        n_lanes = 0
        tile: Tuple[int, int]
        access_point: Union[TileAccessPoint, None]
        for tile, access_point in self.tile_access_points:
            if access_point.position in lane_clusters_dict:
                lane_clusters_dict[access_point.position].append(tile)
            else:
                lane_clusters_dict[access_point.position] = [tile]
        for lane, lane_clusters in lane_clusters_dict.copy().items():
            lane_above = []
            lane_below = []
            access_point = None
            for tile in lane_clusters:
                if tile[0] > lane[0]:
                    lane_above.append(tile)
                else:
                    lane_below.append(tile)
                access_point = self.tile_access_points[tile]
            lane_above.sort(key=lambda x: x[0])  # sort lane by aisle distance
            lane_above.reverse()
            lane_below.sort(key=lambda x: x[0])
            lane_clusters_dict_with_keys[access_point.position] = {
                AccessDirection.BELOW: Lane(lane_above),
                AccessDirection.ABOVE: Lane(lane_below)
            }
            n_lanes += 1 if lane_above else 0
            n_lanes += 1 if lane_below else 0
            # {'below': lane_above, 'above': lane_below, 'locked': set()}
        return lane_clusters_dict_with_keys, n_lanes

    def get_lane(self, storage_position):
        xy_position = storage_position[0:2]
        aisle, direction = self.locate_access_point(xy_position)
        lane: Lane = self.lane_clusters[aisle][direction]
        return lane

    def get_lane_locations(self, storage_position: Tuple[int, int, int]):
        """
        Returns all the positions in the lane that storage_position belongs to.

        :param storage_position: The storage position for which to retrieve
            all lane positions.
        :return:
        """
        lane = self.get_lane(storage_position)
        return lane.tiles

    def get_access_point_direction(self, storage_position):
        xy_pos = storage_position[0:2]
        direction = self.tile_access_points[xy_pos].direction
        return direction.value

    def locate_access_point(
            self, storage_location: Tuple[int, int]) -> (Tuple[int, int], int):
        """
        Retrieves the closest aisle position relative to the passed
        storage_location as well as the direction in which a vehicle needs to
        travel to reach the aisle from the storage position.

        :param storage_location: The location for which to find the closest
            aisle position and travel direction.
        :return: An aisle position tuple and 1 if the vehicle needs to travel
            down from the storage location to reach the aisle or -1 if the
            vehicle needs to travel up.
        """
        tile_acess_point = self.tile_access_points[storage_location]
        aisle = tile_acess_point.position
        direction = tile_acess_point.direction
        return aisle, direction

    def is_lane(self, position_to_shift):
        if tuple(position_to_shift[0:2]) in self.tile_access_points:
            return False
        else:
            return True

    def unassign_empty_lane(self, storage_location, sku):
        """if a lane is completely empty (i.e. after many repeated retrieval
        orders), it gets unassigned so that it can be assigned to a new sku"""
        aisle, direction = self.locate_access_point(
            storage_location[0:2])
        lane = aisle + (direction.value,)
        unassign_lane = True
        for storage_location in self.get_locations_in_lanes([lane]):
            if self.S[storage_location] > StorageKeys.EMPTY:
                unassign_lane = False
        if unassign_lane:
            # unassign lane
            if sku in self.sku_lanes:
                self.sku_lanes[sku].discard(lane)
            self.lane_assigned[lane] = False

    def get_locations_in_lanes(
            self, lanes: Iterable[Tuple[(int, int, int)]], asint=False
    ) -> Union[Set[Tuple[int, int, int]], Set[int]]:
        """takes a list of lanes (row, column, direction) and returns all of
        the storage locations in those lanes"""
        locations_in_lanes = set({})
        for lane in lanes:
            direction = (AccessDirection.ABOVE
                         if lane[2] == -1 else AccessDirection.BELOW)
            lane_cluster = self.lane_clusters[lane[0:2]][direction]
            for storage_tile in lane_cluster:
                for i in range(self.S.shape[2]):
                    if asint:
                        locations_in_lanes.add(
                            ravel(storage_tile + (i,), self.S.shape))
                    else:
                        locations_in_lanes.add(storage_tile + (i,))
            # if 'above' in lane_cluster:
            #     for storage_tile in lane_cluster['above']:
            #         for i in range(self.n_levels):
            #             locations_in_lanes.append(storage_tile + (i,))
            # if 'below' in lane_cluster:
            #     for storage_tile in lane_cluster['below']:
            #         for i in range(self.n_levels):
            #             locations_in_lanes.append(storage_tile + (i,))
        return locations_in_lanes

    def remove_from_occupied_lane(
            self, storage_position: Tuple[int, int, int], sku: int):
        ap_pos, ap_dir = self.locate_access_point(storage_position[:2])
        self.occupied_lanes[sku][ap_pos][ap_dir].remove(
            ravel(storage_position, self.S.shape))
        if len(self.occupied_lanes[sku][ap_pos][ap_dir]) == 0:
            del self.occupied_lanes[sku][ap_pos][ap_dir]
        if len(self.occupied_lanes[sku][ap_pos]) == 0:
            del self.occupied_lanes[sku][ap_pos]
        if len(self.occupied_lanes[sku]) == 0:
            del self.occupied_lanes[sku]

    def add_to_occupied_lane(self, storage_location, sku):
        aisle, direction = self.locate_access_point(storage_location[:2])
        if sku not in self.occupied_lanes:
            self.occupied_lanes[sku] = {}
        if aisle not in self.occupied_lanes[sku]:
            self.occupied_lanes[sku][aisle] = {}
        if direction not in self.occupied_lanes[sku][aisle]:
            self.occupied_lanes[sku][aisle][direction] = set()
        self.occupied_lanes[sku][aisle][direction].add(
            ravel(storage_location, self.S.shape))
