from collections import defaultdict
from typing import Dict, Tuple, Set, Optional, List, cast, Union, TYPE_CHECKING

import numpy as np

from slapstack.core_state_lane_manager import LaneManager, Lane
from slapstack.core_state_zone_manager import ZoneManager
from slapstack.helpers import (StorageKeys, AccessDirection, TimeKeys,
                               BatchKeys, faster_deepcopy, unravel, ravel)

if TYPE_CHECKING:
    from slapstack.core_state import TravelEventTrackers, EventManager


class LocationManager:
    """
    This class is used for datastructures whose values change frequently,
    such as open storage locations, occupied storage locations, and free agv
    positions.

    Attributes:
        open_storage_locations (Set[int]): Locations in lanes which can be used
        next for storage. For all lanes, the only location usable for storage is
        the one directly adjacent to the last occupied lane location, or, if the
        lane is empty, the first lane position from the back. Not to be confused
        with the total number of open locations (n_open_locations).

        ...
    """
    def __init__(self, storage_matrix: np.ndarray, arrival_matrix: np.ndarray,
                 batch_id_matrix: np.ndarray, lane_manager: LaneManager,
                 events: 'EventManager'):
        self.S = storage_matrix
        self.T = arrival_matrix
        self.B = batch_id_matrix
        self.batch_arrivals: Dict[np.float32, float] = {}
        self.n_levels = storage_matrix.shape[2]
        self.lane_manager = lane_manager
        self.zone_manager = ZoneManager()

        # The length of this list can be at most equal to the number of lanes.
        self.open_storage_locations = self.find_open_locations()
        self.open_storage_loc_prev = set()
        # location supersets and locked locations subsets
        self.occupied_locations = dict({})  # maps sku to occupied locations
        self.occupied_locations_cache = dict({})
        self.locked_open_storage = set()
        self.locked_occupied_storage = set()
        # locations from lanes reserved for 1st delivery legs being executed.
        self.reserved_locations = set()

        self.sku_pick_time = {}
        self.sku_cycle_time: Dict[int, float] = {}
        # cycle_time = alpha * cycle_time + (1-alpha) * current_cycle_time
        self.CYCLE_TIME_ALPHA = 0.7

        self.legal_actions = []
        self.legal_retrieval_actions = defaultdict(set)
        self.legal_delivery_actions = defaultdict(set)

        self.n_open_locations = len(np.argwhere(self.S == StorageKeys.EMPTY))
        self.n_total_locations = self.n_open_locations

        self.n_legal_actions_total = 0
        self.n_legal_actions_squared_total = 0
        self.n_legal_actions_current = 0

        self.n_actions_taken = 0
        # Conditional
        self.lane_wise_entropies = {
            (aisle, direction): 0
            for aisle, dirs in self.lane_manager.lane_clusters.items()
            for direction, locs in dirs.items()
        }
        self.lane_wise_sku_counts: Dict[Dict[Dict[int, int]]] = {}
        self.sku_counts = defaultdict(int)
        self.n_open_locations_per_lane: \
            Dict[Tuple[Tuple[int, int], str], int] = {
                (aisle, direction): len(locs) * self.n_levels
                for aisle, dirs in self.lane_manager.lane_clusters.items()
                for direction, locs in dirs.items()
            }
        self.n_total_locations_per_lane = self.n_open_locations_per_lane.copy()
        self.events = events
        self.sink_location = None
        self.source_location = None
        outline = self.S[:, :, 0]
        io_locations = np.argwhere((outline == StorageKeys.SOURCE)
                                   | (outline == StorageKeys.SINK))
        self.raveled_io_locs = [ravel(tuple(i) + (0,), self.S.shape)
                                for i in io_locations]
        # self.perform_sanity_check()

    def set_source_location(self, source_loc: int):
        """
        Used in cross-docking to force the next delivery event to go from dock
        to dock. See BatchLIFO retrieval policy.

        :param source_loc:
        :return:
        """
        self.source_location = source_loc

    def perform_sanity_check(self):
        """
        Performs some sanity checks to ensure that the information stored
        within various datastructures is not contradictory. The list of
        attributes tested here are non-exhaustive and can be expanded as seen
        fit.
        :return:
        """
        import traceback
        try:
            assert (self.n_total_locations ==
                    (self.n_open_locations +
                     sum(map(len, self.occupied_locations.values()))))
            assert (self.n_open_locations ==
                    sum(self.n_open_locations_per_lane.values()))
            assert (self.n_open_locations ==
                    sum(self.zone_manager.n_open_locations_per_zone.values()))
            assert (self.n_total_locations ==
                    sum(self.zone_manager.n_total_locations_per_zone.values()))
        except AssertionError:
            traceback.print_exc()
            pass

    def invalidate_sku_location_cache(self):
        """
        Resets the sku location cache. This function should be allways called
        at the beginning of the step() function.

        :return: None.
        """
        self.occupied_locations_cache = dict({})

    # TODO: move all distance/routing computation to this class
    def get_free_spaces_in_lane(self, storage_location):
        assert storage_location[0:2] in self.lane_manager.tile_access_points
        tiles_in_lane = self.lane_manager.get_lane_locations(storage_location)
        assert storage_location[0:2] in tiles_in_lane
        n_free_spaces = 0
        for tile in tiles_in_lane:
            for i in range(0, self.n_levels):
                tile_with_level = tile + (i,)
                if self.S[tile_with_level] == StorageKeys.EMPTY:
                    n_free_spaces += 1
        return n_free_spaces

    def get_locs_lanes_reserved_for_delivery(
            self, travel_event_stats: 'TravelEventTrackers'
    ) -> Set[Tuple[int, int, int]]:
        """
        This function returns any locations which are reserved for the delivery
        1st legs that have been initiated. If the number of pending delivery
        1st legs are greater or equal to the number of open_locations, then the
        lanes containing those open locations are reserved to be used for the
        corresponding delivery 2nd legs.

        :return:
        """
        ubound_delivery = travel_event_stats.get_unbound_travel_events()
        if ubound_delivery >= len(self.open_storage_locations):
            if (self.open_storage_loc_prev
                    != self.open_storage_locations):
                self.open_storage_loc_prev = self.open_storage_locations
                self.reserved_locations = set()
                for loc_i in self.open_storage_locations:
                    loc = unravel(loc_i, self.S.shape)
                    locations = self.lane_manager.get_lane_locations(loc)
                    for loc in locations:
                        for level in range(self.n_levels):
                            self.reserved_locations.add(loc + (level,))
            return self.reserved_locations
        return set()

    # <editor-fold desc="OUT OF ZONE LOCATION FUNCTIONS">
    def get_out_of_zone_sku_locations(
            self, sku: int, tes: 'TravelEventTrackers') -> Set[Tuple[int, int]]:
        locations = self.zone_manager.out_of_zone_skus[sku]
        if self.locked_occupied_storage != set({}):
            exclusion_set = set()
            for loc in self.locked_occupied_storage:
                loc_tup = unravel(loc, self.S.shape)
                exclusion_set.add(loc_tup)
            locations = set.difference(locations, exclusion_set)
        reserved_locs = self.get_locs_lanes_reserved_for_delivery(tes)
        if reserved_locs:
            # As a minor optimization,
            # only compute intersection if reserved_locs is non-empty
            locations = set.difference(locations, reserved_locs)
        return locations

    def lock_lane(self, location: Tuple[int, int]) -> \
            Tuple[Optional[Tuple[int, int]], List[Tuple[int, int]]]:
        """this function is used to "lock lanes" when certain types of travel
        events (delivery second leg and retrieval first leg). This does two
        things: 1) prevents pallet shifting and updates to location caches from
        affecting other travel events and 2) makes the simulation a bit more
        realistic by not allowing agvs to pass over each other in the same lane.

        To lock a lane, the legal actions for both delivery and retrieval orders
         that are contained in the same lane as the given location are added to
         sets self.locked_open_storage and self.locked_occupied_storage. Later,
         when legal actions are needed, the locations in these sets are removed
         before returning the rest of the legal actions.

        """
        # get other locations in lane
        stacks_in_lane = self.lane_manager.lock_lane(location)
        # go through each location and run lock_stack:
        delivery_action_to_return = None
        retrieval_actions_to_return = []
        # go through each stack in the lane
        # TODO: consider using Lane object to eliminate loop
        for stack in stacks_in_lane:
            # go through each level in a stack
            for i in range(self.n_levels):
                position = (stack + (i,))
                # if there is a legal delivery action, lock it
                if ravel(position, self.S.shape) in self.open_storage_locations:
                    delivery_action_to_return = position
                    self.__add_locked_open_location(position)
                    self.__discard_open_location(position)
                elem = self.S[position]
                # if there are pallets, lock them
                if elem != StorageKeys.EMPTY:
                    retrieval_actions_to_return.append(position)
                    self.__add_locked_occupied_location(position)
        return delivery_action_to_return, retrieval_actions_to_return

    def unlock_lane(self, delivery_action: Tuple[int, int, int],
                    retrieval_actions: List[Tuple[int, int, int]]):
        """called when retrieval first leg and delivery second leg travel events
        are handled
        """
        # removed locked delivery action
        self.__discard_locked_open_location(delivery_action)
        # remove lane from locked set
        search_tile = (delivery_action[:2]
                       if delivery_action else retrieval_actions[0][:2])
        self.lane_manager.unlock_lane(delivery_action, search_tile)
        # remove locked retrieval actions
        for act in retrieval_actions:
            self.__discard_locked_occupied_location(act)

    def __update_sku_pick_time(self, sku, time):
        # Todo: Save only for defined time period
        self.sku_pick_time.setdefault(sku, []).append(time)

    def find_open_locations(self) -> Set[int]:
        """
        Creates a list of open locations immediately after simulation
        initialization, when all lanes are empty.

        Since this method is called exactly once, the overhead incurred by
        looping is acceptable.

        :return: A list of open locations in raveled form.
        """
        open_locations = set({})
        for _, lanes in self.lane_manager.lane_clusters.items():
            if lanes[AccessDirection.ABOVE]:
                open_locations.add(ravel(
                    lanes[AccessDirection.ABOVE][0] + (0, ), self.S.shape))
            if lanes[AccessDirection.BELOW]:
                open_locations.add(ravel(
                    lanes[AccessDirection.BELOW][0] + (0, ), self.S.shape))
        return open_locations

    def update_legal_actions_statistics(self):
        self.n_legal_actions_current = len(self.legal_actions)
        self.n_legal_actions_total += self.n_legal_actions_current
        self.n_legal_actions_squared_total += (self.n_legal_actions_current
                                               * self.n_legal_actions_current)
        self.n_actions_taken += 1

    def update_on_retrieval_first_leg(
            self, pallet_position: Tuple[int, int, int], sku: int,
            pallet_cycle_time: float, time: float):
        """
        Updates Trackers after RetrievalFirstLeg events finish.

        :param pallet_position: location at which the pallet was stored
        :param sku: sku of the pallet
        :param pallet_cycle_time: time elapsed between the placement and
            picking of the pallet.
        :param time: time at which the retrieval order was finished.
        :return: the time penalty incurred in shifting the pallets if the
            desired pallet was not taken from the front.
        """
        self.__update_sku_count(pallet_position, sku, retrieval=True)
        self.__update_sku_pick_time(sku, time)
        self.__update_sku_cycle_time(sku, pallet_cycle_time)
        shift_penalty = self.update_location_cache(pallet_position, sku, True)
        self.update_legal_actions_statistics()
        return shift_penalty

    def update_on_delivery_second_leg(self, pallet_position, sku,
                                      time, batch_id, init=False):
        self.__update_sku_count(pallet_position, sku, retrieval=False)
        shift_penalty = self.update_location_cache(pallet_position, sku, False)
        if batch_id in self.batch_arrivals:
            self.batch_arrivals[batch_id] = min(
                time, self.batch_arrivals[batch_id])
        else:
            self.batch_arrivals[batch_id] = time
        if not init:
            self.update_legal_actions_statistics()
        return shift_penalty

    def __update_sku_cycle_time(self, sku, pallet_cycle_time):
        # TODO: What if there is no order for a long time?
        if sku not in self.sku_cycle_time:
            self.sku_cycle_time[sku] = pallet_cycle_time
        else:
            self.sku_cycle_time[sku] = (self.CYCLE_TIME_ALPHA
                                        * self.sku_cycle_time[sku]
                                        + (1 - self.CYCLE_TIME_ALPHA)
                                        * pallet_cycle_time)

    def __update_sku_count(self, storage_position, pallet_sku, retrieval=True):
        ap_pos, ap_dir = self.lane_manager.locate_access_point(
            storage_position[:2])
        i = -1 if retrieval else 1
        self.sku_counts[pallet_sku] += i
        if ap_pos in self.lane_wise_sku_counts:
            if ap_dir in self.lane_wise_sku_counts[ap_pos]:
                if pallet_sku in self.lane_wise_sku_counts[ap_pos][ap_dir]:
                    self.lane_wise_sku_counts[ap_pos][ap_dir][pallet_sku] += i
                else:
                    assert i > 0
                    self.lane_wise_sku_counts[ap_pos][ap_dir][pallet_sku] = i
            else:
                assert i > 0
                self.lane_wise_sku_counts[ap_pos][ap_dir] = {pallet_sku: i}
        else:
            assert i > 0
            self.lane_wise_sku_counts[ap_pos] = {ap_dir: {pallet_sku: i}}
        # self.lane_wise_sku_counts[aisle][direction][pallet_sku] += i
        self.n_open_locations_per_lane[(ap_pos, ap_dir)] -= i
        self.n_open_locations -= i
        self.zone_manager.update_open_locations(ap_pos, -i)

        # Update entropy of the modified lane based on the new counts.
        sku_counts = np.array(
            list(self.lane_wise_sku_counts[ap_pos][ap_dir].values()) +
            [self.n_open_locations_per_lane[(ap_pos, ap_dir)]]
        )
        p_x = (sku_counts[sku_counts != 0]
               / self.n_total_locations_per_lane[(ap_pos, ap_dir)])
        self.lane_wise_entropies[(ap_pos, ap_dir)] = -np.sum(
            p_x * np.log2(p_x))

    def __remove_pallet(
            self, storage_position: Tuple[int, int, int], pallet_sku: int):
        """remove the given SKU from the storage position in
        self.occupied_lanes

        Raises KeyError when trying to remove an inexistent position from the
        cache.
        """
        self.occupied_locations[pallet_sku].remove(
            ravel(storage_position, self.S.shape))
        self.lane_manager.remove_from_occupied_lane(
            storage_position, pallet_sku)
        tgt_lane: Lane = self.lane_manager.get_lane(storage_position)
        tgt_lane.update_sku_border(pallet_sku, storage_position, added=False)


    def assert_storage_and_time_are_similar(self, storage_location):
        if self.S[storage_location] > StorageKeys.EMPTY:  # if storage
            # location is not empty
            assert self.T[storage_location] != -1
        else:  # if storage is empty
            assert self.T[storage_location] == -1,\
                f'storage: {self.S[storage_location]} and ' \
                f'arrival: {self.T[storage_location]}'

    def update_location_cache(self, storage_location: Tuple[int, int, int],
                              pallet_sku: int, freed=False) -> int:
        """
        This functions primarily updates the open_storage_locations set and
        the occupied_storage_location dictionary whenever pallets are delivered
        to or retrieved from the given storage_location. Pallets can be shifted
        if a retrieval creates a hole in a lane/stack. A shift penalty for each
        shift is given and returned.

        :param storage_location:
        :param pallet_sku:
        :param freed:
        :return:
        """
        shift_penalty = 0
        if freed:  # pallet is retrieved
            shift_penalty = self.__update_location_cache_on_retrieval(
                storage_location, pallet_sku)
        else:  # delivering a pallet.
            self.lane_manager.add_lane_assignment(storage_location, pallet_sku)
            self.__add_pallet(storage_location, pallet_sku)
            # self.open_storage_locations.discard(storage_location)
            self.__discard_open_location(storage_location)
            self.__open_next_location(storage_location)
            # self.assert_storage_and_time_are_similar(storage_location)

        return shift_penalty

    def __update_location_cache_on_retrieval(
            self, location: Tuple[int, int, int], sku: int):
        # remove sku from storage_matrix (and arrival time from time matrix)
        self.__remove_pallet(location, sku)
        # self.open_storage_locations.add(storage_location)
        # TODO: what if a shift happened here!!!
        # TODO: what if you retrieve from a locked lane?
        self.__add_open_location(location)
        # if pallets need to be shifted
        shift_needed = self.__shift_needed(location)
        if shift_needed:
            shift_penalty, location = self.__shift_pallets_in_lane(location)
        else:
            shift_penalty = 0
        # self.assert_storage_and_time_are_similar(location)
        # if lane is empty, unassign it
        if self.lane_manager.pure_lanes:
            self.lane_manager.unassign_empty_lane(location, sku)
        return shift_penalty

    def __open_next_location(self, storage_location: Tuple[int, int, int]):
        """
        Called on the delivery of a pallet to add the next location relative to
        the access aisle to open locations.

        :param storage_location: The location where the pallet was delivered.
        :return: None.
        """
        next_position: List[int, int, int] = list(storage_location)
        next_position[2] = next_position[2] + 1
        if next_position[2] > self.n_levels - 1:  # if stack is full
            next_position[2] = 0
            direction = self.lane_manager.get_access_point_direction(
                tuple(storage_location))
            next_position[0] = next_position[0] + -1 * direction

        if self.S[tuple(next_position)] >= StorageKeys.EMPTY:
            next_position_t = cast(Tuple[int, int, int], tuple(next_position))
            self.__add_open_location(next_position_t)

    # <editor-fold desc="SHIFT MECHANISM">
    def __shift_needed(self, storage_position: Tuple[int, int, int]) -> bool:
        """
        Checks if the retrieval of a pallet leads to the formation of a hole in
        the lane (i.e. an unoccupied position surrounded by occupied locations).

        :param storage_position: The position from which a pallet is to be
            retrieved.
        :return: True whether the retrieval creates a hole.
        """
        shift_candidate, _, _ = self.__get_shift_src(storage_position)
        if self.S[tuple(shift_candidate)] \
                not in [StorageKeys.EMPTY, StorageKeys.AISLE]:
            return True
        return False

    def __get_shift_src(self, hole_location: np.array
                        ) -> Tuple[List[int], List[int], int]:
        """
        Return the position of the pallet to be used in pluging a hole.
        The following scheme is used (| marks the aisle, o the hole; the
        representation is a slice of the S matrix along a particular lane,
        so x decreases moving right, y is fixed and z increases moving up):
           | 1 2 3
           | 4 o 5  --> tgt == 2
           | 6 7 8

           | 1 o 2
           | 4 2 5  --> tgt == 6
           | 6 7 8
        If the aisle is on the other side, the x selection is flipped
        (-1 instead of +1).

        :param hole_location: The location of the hole to plug.
        :return: The source position of the palet to plug it with.
        """
        x, y, z = hole_location[0], hole_location[1], hole_location[2]
        shift_tgt = [x, y, z]
        direction = self.lane_manager.get_access_point_direction(
            tuple(shift_tgt)) * -1
        shift_src = [x, y, z + 1]
        if self.lane_manager.level_too_high(shift_src):
            shift_src[2] = 0  # z
            shift_src[0] = shift_src[0] + direction  # x
        return shift_src, shift_tgt, direction

    def __shift_pallets_in_lane(
            self, storage_location) -> (int, Tuple[int, int, int]):
        """
        Goes through each pallet in a lane and moves it one space away from the
        aisle to prevent holes from forming. A penalty proportional to the
        number of shifts required to circumvent the hole formation is returned.

        If the retrieval indicated by the storage_location parameter does not
        create a hole, no action is taken.

        :param storage_location: The location from which a pallet is retrieved.
        :return: A tuple containing the number of shifts and the new position of
            an empty tile.
        """
        self.__discard_open_location(storage_location)
        # shift all pallets that are closer to aisle (or above retrieved pallet)
        shift_src, shift_tgt, direction = self.__get_shift_src(storage_location)
        shift_penalty = 0
        while not (self.lane_manager.is_lane(shift_src)
                   or self.is_empty(tuple(shift_src))):
            # perform shift!
            self.__shift_pallet(shift_src, shift_tgt)
            self.zone_manager.update_any_out_of_zone_sku_locations(
                shift_src, shift_tgt, self.S[tuple(shift_tgt)]
            )
            shift_src, shift_tgt, direction = self.__get_shift_src(shift_src)
            shift_penalty += 1
        released_tile = cast(Tuple[int, int, int], tuple(shift_tgt))
        self.__add_open_location(released_tile)
        return shift_penalty, released_tile

    def __shift_pallet(self, shift_src, shift_tgt):
        """
        Updates the simulation structures reflecting the shift of a pallet from
        shift_src (the next hole) to shift_tgt (the previous hole).

        :param shift_src: The previous position of the pallet in lane
            (now a hole).
        :param shift_tgt: The next position of the pallet to be shifted.
        :return: None.
        """
        next_hole = tuple(shift_src)
        ex_hole = tuple(shift_tgt)
        shift_sku = self.S[next_hole]
        self.S[ex_hole] = shift_sku
        self.T[ex_hole] = self.T[next_hole]
        self.B[ex_hole] = self.B[next_hole]
        self.S[next_hole] = StorageKeys.EMPTY
        self.T[next_hole] = TimeKeys.NAN.value  # -1
        self.B[next_hole] = BatchKeys.NAN.value  # -1
        self.__add_pallet(ex_hole, shift_sku)
        self.__remove_pallet(next_hole, shift_sku)
    # <editor-fold desc="SHIFT MECHANISM">

    def __add_pallet(
            self, storage_location: Tuple[int, int, int], pallet_sku: int):
        """adds storage location tuple for new pallets to add to
         occupied_lanes"""
        self.lane_manager.add_to_occupied_lane(storage_location, pallet_sku)
        tgt_lane: Lane = self.lane_manager.get_lane(storage_location)
        tgt_lane.update_sku_border(pallet_sku, storage_location, added=True)
        if pallet_sku not in self.occupied_locations:
            self.occupied_locations[pallet_sku] = {
                ravel(storage_location, self.S.shape)}
        else:
            self.occupied_locations[pallet_sku].add(
                ravel(storage_location, self.S.shape))

    def is_empty(self, next_position):
        """returns true if storage position is empty"""
        if self.S[next_position] == StorageKeys.EMPTY:
            return True
        else:
            return False

    def __add_open_location(self, position: Tuple[int, int, int]):
        """takes a position tuple (3,1,4) and ravels it into one integer (154)
         based on the dimensions of the warehouse then adds it
         to the open_storage_location set."""
        if position is None:
            return
        self.open_storage_locations.add(ravel(position, self.S.shape))

    def __discard_open_location(self, position):
        if position is None:
            return
        self.open_storage_locations.discard(ravel(position, self.S.shape))

    def __add_locked_open_location(self, position):
        if position is None:
            return
        self.locked_open_storage.add(ravel(position, self.S.shape))

    def __discard_locked_open_location(self, position):
        if position is None:
            return
        self.locked_open_storage.discard(ravel(position, self.S.shape))

    def __add_locked_occupied_location(self, position):
        if position is None:
            return
        self.locked_occupied_storage.add(ravel(position, self.S.shape))

    def __discard_locked_occupied_location(self, position):
        if position is None:
            return
        self.locked_occupied_storage.discard(ravel(position, self.S.shape))

    def get_open_locations(self, sku=None) -> Set[int]:
        """returns list of open storage locations. removes locations that are
        currently locked."""
        if self.lane_manager.pure_lanes and sku:
            open_locations = self.get_open_storage_locations_pure(sku)
        else:
            open_locations = (self.open_storage_locations
                              - self.locked_open_storage)
        if not open_locations:
            open_locations = (self.open_storage_locations
                              - self.locked_open_storage)
        return open_locations

    def get_open_storage_locations_pure(self, sku: int) -> Set[int]:
        """used with a warehouse with pure lane configurations. if the sku
        has been assigned to lanes, it returns the open locations in those
        lanes. If it hasn't been assigned or if there are no more open
        locations in its assigned lanes, it returns all the open locations
        that are in lanes that aren't assigned"""
        if sku in self.lane_manager.sku_lanes:  # if sku is
            legal_actions = (self.open_storage_locations
                             - self.locked_open_storage)
            if self.lane_manager.sku_lanes[sku]:
                lanes_assigned_to_sku = self.lane_manager.sku_lanes[sku]
                locations_in_lanes = self.lane_manager.get_locations_in_lanes(
                    lanes_assigned_to_sku, asint=True)
                possible_actions = set.intersection(
                    legal_actions, locations_in_lanes)
                if possible_actions:
                    legal_actions = possible_actions
                else:  # if there's no space in the sku's assigned lanes,
                    # just get open locations that are in unassigned lanes
                    legal_actions = self.get_unassigned_open_storage_locations()
                    if not legal_actions:
                        legal_actions = (self.open_storage_locations
                                         - self.locked_open_storage)
        else:
            legal_actions = self.get_unassigned_open_storage_locations()
            if not legal_actions:  # check that there are actually no
                # print("no unassigned lanes, placing elsewhere")
                legal_actions = (self.open_storage_locations
                                 - self.locked_open_storage)
        return legal_actions

    def get_unassigned_open_storage_locations(self) -> Set[int]:
        """goes through the lane_assigned dictionary, gets lanes that aren't
        assigned and returns all the storage locations in those lanes"""
        # print("getting unassigned storage locations")
        unassigned_lanes = set()
        for key, value in self.lane_manager.lane_assigned.items():
            if not value:   # if lane not assigned
                unassigned_lanes.add(key)
        unassigned_lane_locations: Set[int] = (
            self.lane_manager.get_locations_in_lanes(
                unassigned_lanes, asint=True))
        return set.intersection(unassigned_lane_locations,
                                set(self.get_open_locations()))

    def get_sku_locations(
            self, sku: int, tes: 'TravelEventTrackers') -> Set[int]:
        """
        returns list of locations with specified SKU. removes locations that
        are currently locked. this function is also used when checking to see if
        a retrieval order should be processed - if sku_locations is empty
        then the order does not get processed and gets queued

        This function is called multiple times from the events module during
        one step. As such, we can decrease its runtime by using an
        occupied_locations_cahe for the duration of one step. The cache must be
        emptied after or before each step.
        @see: invalidate_occupied_locations_cache

        :param tes:
        :param sku:
        :return:
        """
        # assert sku in self.occupied_lanes.keys(), sku
        # if self.pure_lanes:
        #     if (sku in self.occupied_locations_cache
        #             and self.occupied_locations_cache[sku]):
        #         locations = self.occupied_locations_cache[sku]
        #     else:
        #         # the following call also filters locked_occupied_storage
        #         locations = self.get_actions_closest_to_aisle(sku)
        #         self.occupied_locations_cache[sku] = locations
        # else:
        if self.source_location:
            locations = {self.source_location}
            return locations
        if sku in self.occupied_locations and self.occupied_locations[sku]:
            locations = (self.occupied_locations[sku]
                         - self.locked_occupied_storage)
            reserved_locs = self.get_locs_lanes_reserved_for_delivery(tes)
            if reserved_locs:
                # As a minor optimization,
                # only compute intersection if reserved_locs is non-empty
                locations = set.difference(locations, reserved_locs)
            return locations
        else:
            return set([])

    # def update_s_t_b_matrices(self, position: Tuple[int, int, int],
    #                         s_value, t_value, b_value):
    #     """updates storage and arrival matrices within given values"""
    #     p = position
    #     self.S[p[0], p[1], p[2]] = s_value
    #     self.T[p[0], p[1], p[2]] = t_value
    #     self.B[p[0], p[1], p[2]] = b_value

    def __get_first_suitable_tile(
            self, lane: Lane, sku: int) -> Union[int, None]:
        """
        Given a lane and an sku, this method finds the sku position in the lane
        which is closest to the aisle. Raises an error if the SKU is not
        contained in the lane (see get_border_tile in Lane class), the retrieved
        tile is not present in the occupied_locations[sku] or the tile is
        present in the locked storage.

        :param lane: The lane object to retrieve the border sku tile from.
        :param sku: The sku for which to retrieve the border tile.
        :return: The border sku tile in raveled form.
        """
        location = lane.get_border_tile(sku)
        int_loc = ravel(location, self.S.shape)
        assert int_loc in self.occupied_locations[sku]
        assert int_loc not in self.locked_occupied_storage
        return int_loc

    def get_actions_closest_to_aisle(
            self, sku: int) -> Union[Set[int], List[int]]:
        """
        used for retrieval. if lanes are pure, then get_occupied_locations
        only returns the pallet that is closest to the aisle. This means an
        AGV wouldn't need to move any pallets, no pallet shifting needed

        TODO: Finish comment

        :param sku:
        :return:
        """
        closest_pallets = set({})
        if sku not in self.lane_manager.sku_lanes:
            return set({})
        else:
            for lane_ap in self.lane_manager.sku_lanes[sku]:
                direction = (AccessDirection.ABOVE
                             if lane_ap[2] == -1 else AccessDirection.BELOW)
                lane = self.lane_manager.lane_clusters[lane_ap[0:2]][direction]
                int_loc = self.__get_first_suitable_tile(lane, sku)
                if int_loc is not None:
                    closest_pallets.add(int_loc)
            return closest_pallets  # closest_pallets_raveled

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)

    def get_direct_sink_action(self, sku):
        """
        Checks if there is an order requiring the SKU passed as a parameter
        waiting at one of the sink tiles.

        :param sku: The order to check.
        :return: The sink location requiring the sku or None if no such sink
            location exists.
        """
        events = self.events
        sink_location = None
        if events.queued_retrieval_orders.get(sku):
            retrieval_orders = events.queued_retrieval_orders.get(sku)
            if len(retrieval_orders) > 0:
                retrieval_order = retrieval_orders[0]
                self.sink_location = self.raveled_io_locs[retrieval_order.sink]
            else:
                self.sink_location = None
        return sink_location
