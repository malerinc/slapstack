from copy import deepcopy
from typing import Tuple, Dict, List, Set

from slapstack_controls.extensions.control_helpers import (get_oldest_batch,
                                                           count_sku)
from slapstack.core import State
from collections import namedtuple, defaultdict
import heapq as h
import numpy as np

from slapstack.core_events import Order
from slapstack.core_state_route_manager import Route
from slapstack.core_state_location_manager import LocationManager
from slapstack.extensions.c_helpers import c_ravel, c_unravel
from slapstack.helpers import print_3d_np, AccessDirection
from slapstack.interface_templates import StorageStrategy

OpenLocation = namedtuple('OpenLocation', ['distance', 'arrival_time', 'xyz'])


class StorageLane:
    def __init__(self, distance, location):
        self.distance = distance
        self.location = location


SKU = namedtuple('SKU', ['sku', 'popularity', 'maximum_inventory', 'COI'])


def get_distance(src: np.ndarray, dest: Tuple[int, int], state: State):
    if state:
        # sort storage locations by how long it takes to reach it, rather
        # than the euclidean distance. AGVs can't take a straight line to
        # storage locations.
        route = Route(state.routing, state.location_manager.grid, src,
                      dest, through_pallet_weight=100)
        return route.get_duration()


class KeyHeap(object):
    def __init__(self, initial=None, key=lambda x: x):
        self.key = key
        self.index = 0
        if initial:
            self.data = [(key(item), item) for item in initial]
            self.index = len(self.data)
            h.heapify(self.data)
        else:
            self.data = []

    def push(self, item):
        h.heappush(self.data, (self.key(item), item))
        self.index += 1

    def pop(self):
        return h.heappop(self.data)[1]


class StoragePolicy(StorageStrategy):
    def __init__(self, init=False):
        super().__init__('delivery')
        self.name = 'storage_policy'
        self.init = init

    def get_action(self, state: State):
        raise NotImplementedError


class RetrievalPolicy(StorageStrategy):
    def __init__(self):
        super().__init__('retrieval')

    def get_action(self, state: State):
        raise NotImplementedError


class RandomOpenLocation(StoragePolicy):
    def __init__(self):
        super().__init__()

    def get_action(self, state: State):
        assert state.location_manager.get_open_locations(state.current_sku)
        return np.random.choice(state.location_manager.get_open_locations(
            state.current_sku))


class ClosestOpenLocation(StoragePolicy):
    def __init__(self, very_greedy=False):
        super().__init__()
        self.very_greedy = very_greedy
        self.name = f'{"VeryGreedy" if very_greedy else ""} COL'

    def get_action(self, state: State) -> int:  # alex's idea
        src = state.source_positions[state.current_source_sink]
        if self.very_greedy:
            open_locations = state.location_manager.get_open_locations()
        else:
            sku = state.current_sku
            open_locations = state.location_manager.get_open_locations(sku)
        shortest_distance = 99999
        closest_tgt_loc = None
        for loc in open_locations:
            loc_tuple = c_unravel(loc, state.S.shape)
            distance = get_distance(src, loc_tuple[:2], state)
            if distance < shortest_distance:
                shortest_distance = distance
                closest_tgt_loc = loc
        return closest_tgt_loc


class BestLane(StoragePolicy):
    def __init__(self):
        super().__init__()

    def get_action(self, state: State) -> int:
        src = state.source_positions[state.current_source_sink]
        open_locations = state.location_manager.get_open_locations()
        # norm = np.sqrt(np.square(open_locations[0] - src[0]) +
        #                np.square(open_locations[1] - src[1]))
        # distances = []
        shortest_distance = 99999
        closest_tgt_loc = None
        sc = state.location_manager
        for loc in open_locations:
            loc_tuple = c_unravel(loc, state.S.shape)
            ap_pos, _ = sc.lane_manager.locate_access_point(loc_tuple[:2])
            distance = get_distance(src, ap_pos, state)
            if distance < shortest_distance:
                shortest_distance = distance
                closest_tgt_loc = loc
        # closest = np.argmin(np.array(distances))
        return closest_tgt_loc


class BatchLIFO(RetrievalPolicy):
    def __init__(self):
        super().__init__()

    def get_action(self, state: State):
        """
        Chooses an sku position to pick from the warehouse as per the
        following priority scheme:
            1. Out of zone items first.
            2. LIFO on arrival times.
            3. LIFO on batches.

        :param state: The current warehouse state.
        :return: The sku position to be picked (xyz) raveled to a single
            integer.
        """
        ooz_skus = state.location_manager.get_out_of_zone_sku_locations(
            state.current_sku, state.trackers.travel_event_statistics)
        if ooz_skus != set({}):
            sku_pos = ooz_skus
        else:
            sku_pos = state.location_manager.get_sku_locations(
                state.current_sku, state.trackers.travel_event_statistics)
            sku_pos = [(c_unravel(i, state.S.shape)) for i in sku_pos]
        pos = tuple(get_oldest_batch(
            sku_pos, state.B, state.T, state.location_manager.batch_arrivals))
        state.location_manager.zone_manager.remove_out_of_zone_location(
            state.current_sku, pos)
        action = c_ravel(pos, state.S.shape)
        return action

    @staticmethod
    def __get_oldest_batch(sku_pos: List[Tuple[int, int, int]], state: State):
        """
        Pure python implementation of the cythonised function in
        control_helpers. Useful for debugging.


        :param sku_pos: The list of action (sku locations) candidates to chose
            from.
        :param state: The simulation state.
        :return: The location candidate.
        """
        max_batch_arrival = -np.infty
        target_loc_info = (None, -np.infty)
        for sku_location in sku_pos:
            # loc info
            loc = tuple(sku_location)
            batch_id = state.B[loc]
            batch_arrival_time = state.location_manager.batch_arrivals[batch_id]
            arrival_time = state.T[tuple(sku_location)]
            assert arrival_time != -1
            # decide if location is a candidate
            if batch_arrival_time > max_batch_arrival:
                min_batch_arrival = batch_arrival_time
                target_loc_info = (loc, arrival_time)
            elif batch_arrival_time == max_batch_arrival:
                if arrival_time > target_loc_info[1]:
                    target_loc_info = (loc, arrival_time)
            else:
                continue
        return target_loc_info[0]


class ClassBasedStorage(StoragePolicy):
    """
    zones: dict
        data structure where zone locations are stored. the tuples in the
        lists are only the aisle access point for each lane
        zones = {
                    zone: [lanes],
                    0: [(4, 5), (4, 6), ...],
                    1: [(10, 5), ... ] }
    zones_by_location: dict
        similar to the zones dict but instead of the aisle access point in
        the lists, the actual storage locations are used (3D tuples)
        zones_by_location = {
                                zone: [lanes],
                                0: [(4, 5, 0), (3, 5, 1), ...],
                                1: [(10, 5, 0), (9, 5, 1)... ] }
    sku_zones: dict
        data structure where SKU zone assignments are stored
        sku_zones = {
                        sku: zone,
                        20: 0,
                        36: 0,
                        15: 1,
                        23: 1 }
    n_steps: int
        used to keep track of how many times get_action() was called
    recalculation_steps: int
        each time n_steps reaches a multiple of this value, the SKUs assignments
        are recalculated and SKUs may change zones. can lead to
        out_of_zone_items being populated
    zone_distributions: Tuple[float]
        this tuple determines the distribution of zones and SKU assignments.
        for example: 70% of lanes and SKUs are assigned to zone A,
        20% of lanes and SKUs are assigned to zone B, 10% of lanes and SKUs
        are assigned to zone C
    out_of_zone_items: set[Tuple[int, int, int]]
        this set contains the storage locations of any items whose SKU is in
        the wrong zone after SKU reassignment\
    orders_to_assign_skus: list[Order]
        this is a list of orders that are only used to assign skus during
        initialization/adding initial pallets. they are not used for rest of
        the simulatio
    used_for_initialization: boolean
        if this class based strategy is passed in as an
        initial_pallet_storage_strategy, and not as a regular storage strategy
    sku_lanes: dict
        example = {sku: lanes,
                    117: {(4, 5), (4, 6), (4, 7)},
                    121: {(4, 8)},
                    ...}
    lane_assigned:dict
        example = {(4,5): True,
                    (4, 6): True,
                    (4, 10): False,
                    ...}

    """
    def __init__(self, future_counts=None, recalculation_steps=1000,
                 init=False, n_zones=3):
        super().__init__(init)
        self.zones: dict = dict()
        self.zones_by_location = dict()      # {0: {(a, b), (c, d)...}, 1: {...
        self.sku_zones: dict = dict()
        self.n_steps = 0
        self.recalculation_steps = recalculation_steps
        self.n_zones = n_zones
        self.zone_distributions = (np.geomspace(1, 10, n_zones + 1) - 1) / 10
        self.zone_distributions[-1] = 1
        self.out_of_zone_items = set()
        self.future_counts = future_counts
        self.sku_lanes = dict()
        self.sku_period = 0

    def assign_skus_to_zones(self, state: State) -> (
            Dict[int, int], Dict[int, int]):
        """
        Recomputes SKU zone assignment and returns the current and previous
        assignment.

        :param state: The current warehouse state.
        :return: Two dictionaries indexed by integers representing SKUs with
            integer values corresponding to the SKU zone; the first dictionary
            represents the new zone assignment, while the 2nd keeps track of
            the past assignment.
        """
        raise NotImplementedError(f"Instance of {type(self)} cannot be used "
                                  f"directly as it is an abstract class.")

    def assign_sorted_skus_to_zones(self, sorted_skus: List[int],
                                    storage_ratio_per_sku: Dict[int, float],
                                    n_skus: int):
        """
        This function assigns skus to zones based on the provided order.
        The ratio of SKUs assigned to each zone is decided on the basis
        of number of pallets per SKU in the storage. For example, a zone
        that covers 10% of the warehouse will be assigned the SKUs that
        constitute 10% of the total pallets stored in the warehouse.

        :param sorted_skus: a list of SKUs (integers) which are sorted according
            to some external criteria
        :param storage_ratio_per_sku: a dictionary where the key is sku and the
            value is the proportion of storage that SKU needs to be allocated
        :param n_skus: total number of possible SKUs
        :return: A dictionary indexed by integers representing SKUs with
            integer values corresponding to the SKU zone; the dictionary
            represents the new zone assignment.
        """
        # by default, assign all skus to least desirable zone
        sku_zones = defaultdict(int)

        sku_counts = np.array([storage_ratio_per_sku[sku]
                               for sku in sorted_skus])
        sku_count_ratio_cumsum = np.cumsum(sku_counts) / np.sum(sku_counts)

        # Computes the index start for each zone distribution.
        # For example: for zone distribution 0.1 it gives us the index in sku_
        # count_ratio_cumsum which partitions the array into numbers smaller
        # than 0.1 and larger than 0.1
        indices = np.searchsorted(sku_count_ratio_cumsum,
                                  self.zone_distributions)
        for i in range(self.n_zones):
            skus_in_one_zone = sorted_skus[indices[i]:indices[i + 1] + 1]
            for sku in skus_in_one_zone:
                sku_zones[sku] = i

        # assert len(sku_zones) == n_skus; unknown skus could arrive ;)
        return sku_zones

    # <editor-fold desc="GET_ACTION AND GET_ACTION HELPERS">
    def __process_zones(self, s: State):
        """
        Initializes zones and sku assignments.  After n_steps_to_reassign_skus,
        SKUs are reassigned and any pallets of SKUs that are reassigned to a new
        zone are designated as out_of_zone_items that are given priority for
        next retrieval.

        :param s: The current warehouse state.
        :return: None.
        """
        if self.n_steps == 0:
            self.zones = self.calculate_zones(s)
            self.zones_by_location = self.create_zones_by_locations(s)
            self.sku_zones, previous_sku_zones = self.assign_skus_to_zones(
                s)
            s.location_manager.zone_manager.update_zone_assignments(
                self.zones,
                s.location_manager.n_open_locations_per_lane,
                s.location_manager.n_total_locations_per_lane
            )
        elif ((not self.future_counts
              and self.recalculation_steps % self.n_steps == 0)
              or (self.future_counts
                  and self.sku_period != s.params.sku_period)):
            self.sku_zones, previous_sku_zones = self.assign_skus_to_zones(s)
            self.check_out_of_zone_items(s, previous_sku_zones)
            self.sku_period = s.params.sku_period
            s.location_manager.zone_manager.update_zone_assignments(
                self.zones,
                s.location_manager.n_open_locations_per_lane,
                s.location_manager.n_total_locations_per_lane
            )
        self.n_steps += 1

    def get_action(self, state: State):
        """
        Returns an action based on the current sku to zone distribution. First
        the zone associated with the current sku is identified. If there are
        open locations within the sku zone, the one closest to the source is
        selected from them. If no locations in the zone are available, the sku
        will be placed at the closest open location outside the zone.

        If needed, this function also recomputes the sku to zone assignment.

        :param state: The current warehouse state.
        :return: The closest open location in the sku target zone, or the
            ooz closest open location if the zone is full.
        """
        # zone updates, if necessary
        self.__process_zones(state)
        sku = state.current_sku
        # all open location and zone open location retrieval
        zone_to_deliver = self.__get_target_zone(sku)
        if self.init:
            open_locations = state.location_manager.get_open_locations(sku)
        else:
            open_locations = state.location_manager.legal_actions
        locations_in_zone = self.zones_by_location[zone_to_deliver]
        iz_min_dist, ooz_min_dist = np.infty, np.infty
        iz_action, ooz_action = None, None
        src = np.array(state.source_positions[state.current_source_sink])
        for location in open_locations:
            tgt_loc = c_unravel(location, state.S.shape)
            if tgt_loc in locations_in_zone:
                d = get_distance(src, tgt_loc[:2], state)
                if d < iz_min_dist:
                    iz_action = tgt_loc
            elif iz_action is None and tgt_loc:
                # gets skipped as soon as an in zone location gets found ;)
                d = get_distance(src, tgt_loc[:2], state)
                if d < ooz_min_dist:
                    ooz_action = tgt_loc
        # update ooz in state cache, if necessary, and return
        if iz_action is not None:
            return c_ravel(iz_action, state.S.shape)
        else:
            state.location_manager.zone_manager.add_out_of_zone_sku(
                ooz_action, sku, buffer=True)
            try:
                assert ooz_action
            except AssertionError:
                print('herehere')
            return c_ravel(ooz_action, state.S.shape)

    @staticmethod
    def get_location_min_entropy(state, open_locations) -> int:
        min_entropy = -np.infty
        lowest_entropy_location = None
        sc = state.state_cache
        for location in open_locations:
            unravel_location = c_unravel(location, state.S.shape)
            ap_pos, ap_dir = sc.lane_manger.locate_access_point(
                unravel_location[:2])
            entropy = state.state_cache.lane_wise_entropies[
                (ap_pos, ap_dir)]
            if entropy > min_entropy:
                lowest_entropy_location = location
                min_entropy = entropy
        return lowest_entropy_location

    @staticmethod
    def get_locations_max_entropy_max_pallets(
            state: State, open_locations) -> Set[int]:
        sc: LocationManager = state.location_manager
        lane_wise_max_sku = 0
        max_entropy = -np.infty
        highest_entropy_locations = set()
        other_max_sku_locations = set()
        no_sku_aisle_direction_locations = set()
        for location in open_locations:
            unravel_location = c_unravel(location, state.S.shape)
            ap_pos, ap_dir = sc.lane_manager.locate_access_point(
                unravel_location[:2])
            entropy = sc.lane_wise_entropies[(ap_pos, ap_dir)]
            if entropy > max_entropy:
                highest_entropy_locations = set()
                highest_entropy_locations.add(location)
                max_entropy = entropy
            elif entropy == max_entropy:
                highest_entropy_locations.add(location)
            if (sc.lane_wise_sku_counts.get(ap_pos)
                    and sc.lane_wise_sku_counts[ap_pos].get(ap_dir)):
                total_skus = list(
                    sc.lane_wise_sku_counts[ap_pos][ap_dir].values())
                if len(total_skus) > lane_wise_max_sku:
                    other_max_sku_locations = set()
                    other_max_sku_locations.add(location)
                    lane_wise_max_sku = len(total_skus)
                elif len(total_skus) == lane_wise_max_sku:
                    other_max_sku_locations.add(location)
            else:
                no_sku_aisle_direction_locations.add(location)
        return other_max_sku_locations.union(highest_entropy_locations,
                                             no_sku_aisle_direction_locations)

    def __get_target_zone(self, sku_to_deliver: int) -> int:
        """
        get_action helper function.

        Returns the zone index associated with  the sku_to_deliver parameter.
        If the sku is not yet associated with any zone, the first zone is
        chosen and the sku is added to the sku_zones dictionary.

        :param sku_to_deliver: The sku of the pallet to be placed in the
            warehouse.
        :return: The zone associated with the current sku.
        """
        if sku_to_deliver in self.sku_zones:
            zone_to_deliver = self.sku_zones[sku_to_deliver]
        else:
            zone_to_deliver = 0
            self.sku_zones[sku_to_deliver] = zone_to_deliver
        return zone_to_deliver

    @staticmethod
    def __check_action_integrity(action: Tuple[int, int, int], state: State):
        """
        Checks whether the action is not None, has a length of 2 and that the
        (x, y) coordinates of the action correspond to an aisle.

        :param action: The action to be checked.
        :param state: The warehouse state.
        :return: None.
        """
        assert action
        assert len(action) == 3
        if action[0:2] not in state.location_manager.lane_manager.closest_aisle:
            print_3d_np(state.S)
            print(action)
            print()
        assert action[0:2] in state.location_manager.lane_manager.closest_aisle
        return c_ravel(action, state.S.shape)

    @staticmethod
    def get_average_distance_to_access_points(state, lane_aisle_access):
        """used to find most desirable locations. gets the average distance
        from the middle aisle lane access point to all source tiles (up to
        10) and all sink tiles (up to 4)"""
        distances = []
        sc = state.state_cache
        for source in state.source_positions:
            distances.append(get_distance(source, lane_aisle_access, state))
        for sink in state.sink_positions:
            distances.append(get_distance(sink, lane_aisle_access, state))
        return sum(distances)/len(distances)

    def calculate_zones(self, state: State):
        """
        This function is only executed once. Lanes are taken from the
        state and are assigned to zones based on their distance to the
        entrance.

        :param state: The current warehouse state.
        :return: The zone to lane dictionary.
        """
        lanes: Dict[Tuple[int, int], Dict[str, List[Tuple[int, int]]]]
        lanes = state.location_manager.lane_manager.lane_clusters
        middle_source = int(len(state.source_positions)/2)
        # measure distance from this source tile to each lane aisle access tile
        src = state.source_positions[middle_source]
        lane_heap = KeyHeap([], key=lambda x: x.distance)
        for lane_aisle_access in lanes:
            lane = StorageLane(
                ClassBasedStorage.__get_average_distance_to_access_points(
                    state, lane_aisle_access), lane_aisle_access)
            lane_heap.push(lane)
        zones = defaultdict(list)
        lane_list = lane_heap.data
        # assign lane access points to zones
        zone_indices = (self.zone_distributions * len(lane_list)).astype('int')
        for i in range(self.n_zones):
            lane_section = lane_list[zone_indices[i]: zone_indices[i + 1]]
            for lane in lane_section:
                zones[i].append(lane[1].location)
        return zones

    @staticmethod
    def __get_average_distance_to_access_points(
            state: State, lane_aisle_access):
        """
        calculate_zones helper.

        Used to find most desirable locations. gets the average distance
        from the middle aisle lane access point to all source tiles (up to
        10) and all sink tiles (up to 4).

        :param state: The current warehouse state.
        :param lane_aisle_access:
        :return:
        """
        distances = []
        sum_d = 0
        n_pts = 0
        for sink in state.sink_positions:
            sum_d += get_distance(sink, lane_aisle_access, state)
            n_pts += 1
        for src in state.source_positions:
            sum_d += get_distance(src, lane_aisle_access, state)
            n_pts += 1
        return sum_d / n_pts

    def draw_zones(self, state):
        """
        UNUSED FUNCTION!

        Used for visualization, creates a string of a matrix with the same
        size as the warehouse to show what zone each storage location is in.

        :param state:
        :return:
        """
        zone_matrix = np.full((state.S.shape[0], state.S.shape[1]), -1)
        for zone, locations_in_zone in self.zones_by_location.items():
            for loc in locations_in_zone:
                zone_matrix[(loc[0], loc[1])] = zone
        output_string = ""
        for col in range(0, zone_matrix.shape[1]):
            output_string += "{}".format(col).rjust(3)
        output_string += "\n"
        row = 0
        for s in zone_matrix:
            output_string += "{}".format(row).rjust(3)
            for elem in s:
                if elem == -1:
                    elem = "-"
                output_string += "{}".format(elem).rjust(3)
            row += 1
            output_string += "\n"
        return output_string

    def create_zones_by_locations(self, state: State):
        """creates zones_by_locations"""
        zones_by_location = dict()
        lane_clusters = state.location_manager.lane_manager.lane_clusters
        for zone, lanes in self.zones.items():
            locations_without_level = []
            for lane in lanes:
                locations_without_level.extend(
                    lane_clusters[lane][AccessDirection.ABOVE])
                locations_without_level.extend(
                    lane_clusters[lane][AccessDirection.BELOW])
            # add z coordinate to locations
            locations = set({})
            n_levels = state.location_manager.n_levels
            for loc in locations_without_level:
                for i in range(n_levels):
                    locations.add((loc + (i,)))
            zones_by_location[zone] = locations
        return zones_by_location

    def check_out_of_zone_items(self, state: State,
                                previous_sku_zones: Dict[int, int]):
        """
        Interacts with the location_manager to update ana out of zone SKUs after
        a zone recomputation.

        If the zone that SKUs are assigned to has changed, then any items
        that are now in the wrong zone are logged to the out_of_zone_sku data
        structure in the location_manager.

        :param state: The current warehouse state.
        :param previous_sku_zones: A dictionary mapping SKU to zone before the
            zone recomputation.
        :return: None
        """
        if previous_sku_zones == self.sku_zones:
            return
        else:
            sc = state.location_manager
            for sku in range(1, state.params.n_skus + 1):
                if previous_sku_zones[sku] != self.sku_zones[sku]:
                    zone = previous_sku_zones[sku]
                    new_zone = self.sku_zones[sku]
                    loc_in_old_zone = self.zones_by_location[zone]
                    loc_in_new_zone = self.zones_by_location[new_zone]
                    # get all sku locations in the warehouse
                    sku_locs = ClassBasedStorage.__get_sku_locations(sku, sc)
                    # get locations of skus that are in the old zone
                    old_zone_sku_locs = set.intersection(
                        sku_locs, loc_in_old_zone)
                    new_zone_sku_locs = set.intersection(
                        sku_locs, loc_in_new_zone)
                    # add the newly out of zone sku locations to state cache's
                    # out of zone sku locations
                    ooz_skus = sc.zone_manager.out_of_zone_skus[sku]
                    ooz_skus_new = set.difference(
                        set.union(ooz_skus, old_zone_sku_locs),
                        new_zone_sku_locs)
                    sc.zone_manager.set_out_of_zone_locations(sku, ooz_skus_new)

    @staticmethod
    def __get_sku_locations(sku: int, sc: LocationManager):
        """
        Extracts all sku locations in the warehouse. To that end, first the
        lane clusters where the passed sku is present are extracted from the
        location_manager. Then the locations are extracted from each lane in the
        group (can be at most two, above or below the aisle indexing the
        "lane_clusters" structure).

        :param sku: The sku to retrieva locations for.
        :param sc: The current state cache.
        :return: The set of all sku locations.
        """
        if sku in sc.lane_manager.occupied_lanes:
            lane_clusters = (sc.lane_manager.occupied_lanes[sku].values())
            all_skus_locations = {
                loc
                for lane_cluster in lane_clusters
                for direction in lane_cluster.values()
                for loc in direction
            }
            return all_skus_locations
        else:
            return set()


class ClassBasedPopularity(ClassBasedStorage):
    """this class inherits from ClassBasedStorage because it has a specific
    assign_skus_to_zones functions"""
    def __init__(self, retrieval_orders_only=False,
                 future_counts=None, n_orders=1000, recalculation_steps=1000,
                 name='ClassBasedPopularity', init=False, n_zones=3):
        super().__init__(future_counts, recalculation_steps, init,
                         n_zones=n_zones)
        self.n_orders = n_orders
        self.retrieval_orders_only = retrieval_orders_only
        self.name = name

    def compute_orders_per_sku(
            self, order_list: List[Order]):
        order_counts = dict()
        order_counts = count_sku(order_list, self.retrieval_orders_only)
        # for order in order_list:
        #     if self.retrieval_orders_only and order.type != 'retrieval':
        #         continue
        #     sku = order.SKU
        #     if sku in order_counts:
        #         order_counts[sku] += 1
        #     else:
        #         order_counts[sku] = 1
        return order_counts

    def assign_skus_to_zones(self, state: State) -> (
            Dict[int, int], Dict[int, int]):
        """
        Implementation of the abstract method in the ClassBasedStorage
        superclass.

        This function looks at future or past orders (depending on if the
        strategy is initialized or not), ranks SKUs by popularity (number of
        orders containing SKU), and assigns them one of the available zones.
        can also return the previous_sku_zones for when zones are reassigned.
        Note that with small numbers of SKUs, some zones may not be assigned
        any SKUs.

        :param state: The current warehouse state.
        :return: Two dictionaries indexed by integers representing SKUs with
            integer values corresponding to the SKU zone; the first dictionary
            represents the new zone assignment, while the 2nd keeps track of
            the past assignment.
        """
        previous_sku_zones = deepcopy(self.sku_zones)
        order_counts = defaultdict(int)  # {sku: number of orders, ...}
        # if initial fill, use the orders that are only for assigning skus
        if self.future_counts:
            assert self.future_counts
            # "retrieval"/"delivery", sku, time, sink/source, batch
            if self.retrieval_orders_only:
                for sku in state.params.all_skus:
                    if sku in state.params.n_skus_out[state.params.sku_period]:
                        order_counts[sku] += state.params.n_skus_out[
                            state.params.sku_period][sku]
                    if sku in state.n_skus_inout_now:
                        order_counts[sku] += state.n_skus_inout_now[sku]
            else:
                for sku in state.params.all_skus:
                    if sku in state.params.n_skus_in[state.params.sku_period]:
                        order_counts[sku] += state.params.n_skus_in[
                            state.params.sku_period][sku]
                    if sku in state.params.n_skus_out[state.params.sku_period]:
                        order_counts[sku] += state.params.n_skus_out[
                            state.params.sku_period][sku]
                    if sku in state.n_skus_inout_now:
                        order_counts[sku] += state.n_skus_inout_now[sku]
        # if assigning skus during simulation, calculate based off of history
        # (finished_orders), but only a certain amount
        else:
            nfo = len(state.trackers.finished_orders)
            first_order = 0 if self.n_orders > nfo else -self.n_orders
            count_base = state.trackers.finished_orders[first_order:]
            order_counts = self.compute_orders_per_sku(count_base)
        popularity_sorted_skus = sorted(list(order_counts.keys()),
                                        key=lambda x: -order_counts[x])
        sku_zones = self.assign_sorted_skus_to_zones(
            sorted_skus=popularity_sorted_skus,
            storage_ratio_per_sku=order_counts,
            n_skus=state.params.n_skus
        )
        return sku_zones, previous_sku_zones


class ClassBasedCycleTime(ClassBasedStorage):
    """this class inherits from ClassBasedStorage because it has a specific
    assign_skus_to_zones functions"""
    def __init__(self, n_orders=3000, recalculation_steps=1000, n_zones=3):
        super().__init__(None, recalculation_steps, n_zones=n_zones)
        self.n_orders = n_orders
        self.name = f"classBasedCycleTime_z{n_zones}"

    def assign_skus_to_zones(self, state: State) -> (
            Dict[int, int], Dict[int, int]):
        """
        Implementation of the abstract method in the ClassBasedStorage
        superclass.

        This function ranks SKUs by exponential moving average of cycle times
        (time between placement and removal of a pallet). Top ranked skus are
        assigned to the zone A and so on. The ratio of SKUs assigned to each
        zone is decided on the basis of current pallets per sku in the storage.
        Note that with small numbers of SKUs, some zones may not be assigned any
        SKUs.

        :param state: The current warehouse state.
        :return: Two dictionaries indexed by integers representing SKUs with
            integer values corresponding to the SKU zone; the first dictionary
            represents the new zone assignment, while the 2nd keeps track of
            the past assignment.
        """
        previous_sku_zones = deepcopy(self.sku_zones)

        sku_cycle_time = state.location_manager.sku_cycle_time
        popularity_sorted_skus = sorted(list(sku_cycle_time.keys()),
                                        key=lambda x: sku_cycle_time[x])

        sku_zones = self.assign_sorted_skus_to_zones(
            popularity_sorted_skus,
            state.location_manager.sku_counts,
            state.params.n_skus
        )

        return sku_zones, previous_sku_zones
