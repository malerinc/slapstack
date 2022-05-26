from collections import defaultdict
from typing import Tuple, Dict, List, Set


class ZoneManager:
    def __init__(self):
        # see class based storage strategy
        self.is_initialized = False
        self.out_of_zone_skus = defaultdict(set)
        self.pending_out_of_zone_skus = defaultdict(set)
        self.zone_to_lanes = dict()
        self.lane_to_zones = dict()
        self.n_open_locations_per_zone = dict()
        self.n_total_locations_per_zone = dict()

    def update_open_locations(self, aisle: Tuple[int, int], step: int):
        if self.is_initialized:
            self.n_open_locations_per_zone[self.lane_to_zones[aisle]] += step

    def update_zone_assignments(
            self,
            zone_to_lanes: Dict[int, List[Tuple[int, int]]],
            n_open_locations_per_lane: Dict[Tuple[Tuple[int, int], str], int],
            n_total_locations_per_lane: Dict[Tuple[Tuple[int, int], str], int]
    ):
        """
        Re-calculates the number of open locations and total locations in each
        zone, when the zone assignments are changed. This function is called
        from ClassBasedStorage.
        """
        self.is_initialized = True
        self.zone_to_lanes = zone_to_lanes
        self.lane_to_zones = {
            lane: zone
            for zone, lanes in zone_to_lanes.items()
            for lane in lanes
        }

        for zone in zone_to_lanes.keys():
            self.n_total_locations_per_zone[zone] = 0
            self.n_open_locations_per_zone[zone] = 0

        for (aisle, direction) in n_open_locations_per_lane:
            zone = self.lane_to_zones[aisle]
            self.n_total_locations_per_zone[zone] += \
                n_total_locations_per_lane[(aisle, direction)]
            self.n_open_locations_per_zone[zone] += \
                n_open_locations_per_lane[(aisle, direction)]

    def set_out_of_zone_locations(
            self, sku: int, out_of_zone_locations: Set[Tuple[int, int]]):
        """
        Replaces the out of zone locations for the sku parameter with the
        passed out_of_zone_locations set.

        This function should called on zone recomputation within a class
        based storage strategy.

        :param sku: The index of the out of zone locations to update.
        :param out_of_zone_locations: The new out of zone locations.
        :return: None.
        """
        self.out_of_zone_skus[sku] = out_of_zone_locations

    def remove_out_of_zone_location(self, sku: int,
                                    sku_location: Tuple[int, int]):
        """
        Checks whether a particular sku location is out of zone or not. If out
        of zone, the location will be removed from the set indexed by the sku
        parameter. If the location removal yields an empty out of zone set for
        the passed sku, then the key is removed from the out_of_zone_sku map.

        This function should be called by retrieval strategies.

        :param sku: The sku associated with a location to remove.
        :param sku_location: The sku location.
        :return: None.
        """
        if sku_location in self.out_of_zone_skus[sku]:
            self.out_of_zone_skus[sku].remove(sku_location)
            if self.out_of_zone_skus[sku] == set({}):
                del self.out_of_zone_skus[sku]

    def update_any_out_of_zone_sku_locations(self, current_hole, ex_hole, sku):
        pos_src = tuple(current_hole)
        pos_tgt = tuple(ex_hole)
        # sku = self.S[tuple(ex_hole)]
        assert sku != 0
        if pos_src in self.out_of_zone_skus[sku]:
            self.out_of_zone_skus[sku].remove(pos_src)
            self.out_of_zone_skus[sku].add(pos_tgt)

    def add_out_of_zone_sku(self, location: Tuple[int, int, int], sku: int,
                            buffer: bool = False):
        """
        Adds a sku location to either the out of zone (ooz) location buffer
        (pending_out_of_zone_skus) or the actual ooz list (out_of_zone_sku).

        The buffer should be updated on action selection, while the ooz
        finalization should occur on DeliverySecondLeg event handling. The
        buffering mechanism ensures that ooz locations are only available for
        retrieval after their delivery has been finalized.

        :param location: The location of the sku to be delivered.
        :param sku: The sku to add to the warehouse.
        :param buffer: True if the sku delivery is locked (at DeliverySecondLeg
            event creation) but has not occurred yet (at DeliverySecondLeg event
            handling)
        :return: None.
        """
        if buffer:
            self.pending_out_of_zone_skus[sku].add(location)
        else:
            if location in self.pending_out_of_zone_skus[sku]:
                self.pending_out_of_zone_skus[sku].remove(location)
                self.out_of_zone_skus[sku].add(location)
