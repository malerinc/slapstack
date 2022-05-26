import math
from collections import deque

import numpy as np
import heapq as heap

from slapstack.core_state_route_manager import Route
from slapstack.core_state import State
from slapstack.core_state_agv_manager import AGV
from slapstack.core_state_location_manager import LocationManager
from slapstack.helpers import faster_deepcopy, StorageKeys, VehicleKeys, \
    BatchKeys, TimeKeys, TravelEventKeys
from typing import Tuple, MutableSequence, Set, Dict, Deque, Any


class EventHandleInfo:
    def __init__(
            self,
            action_needed: bool,
            event_to_add,
            travel_event_to_add,
            queued_retrieval_order_to_add,
            queued_delivery_order_to_add
    ):
        self.action_needed = action_needed
        self.event_to_add = event_to_add
        self.travel_event_to_add = travel_event_to_add
        self.queued_retrieval_order_to_add = queued_retrieval_order_to_add
        self.queued_delivery_order_to_add = queued_delivery_order_to_add


class Event:
    """this class contains all the different types of events in the simulation
    that are the foundation of the processes in the storage location allocation
    problem
    """
    def __init__(self, time: float, verbose: bool):
        """

        time: float
            The time at which travel events should end and the time at which
            orders arrive
        verbose: bool
            debugging, slapstack_controls print statements
        """
        self.time = time
        self.verbose = verbose

    def __eq__(self, other):
        """used for sorting events in heaps"""
        return self.time == other.time

    def __le__(self, other):
        return self.time <= other.time

    def __lt__(self, other):
        return self.time < other.time

    def __ge__(self, other):
        return self.time >= other.time

    def __gt__(self, other):
        return self.time > other.time

    def handle(self, state: State):
        raise NotImplementedError


class Order(Event):
    def __init__(self, time: float, SKU: int, order_number: int, batch_id: int,
                 verbose: bool, io_type=None, period=None):
        super().__init__(time, verbose)
        self.SKU = SKU
        self.batch_id = batch_id
        self.order_number = order_number
        self.period = period
        self.completion_time = -1
        self.type = io_type

    def __hash__(self):
        return hash(str(self.order_number))

    def __eq__(self, other):
        """used for sorting events in heaps"""
        if isinstance(other, Order):
            return self.order_number == other.order_number
        else:
            return False

    def handle(self, state: State):
        state.n_skus_inout_now[self.SKU] += 1
        if self.period != state.params.sku_period:
            state.params.sku_period = self.period

    def set_completion_time(self, time: float):
        self.completion_time = time


class Delivery(Order):
    """delivery order contains a SKU number and source position. seizes a free
    AGV, moves it to source to pick up a pallet, then moves it to a desired
    position in the warehouse and drops off the pallet.

    """
    def __init__(self, time: float, SKU: int, order_number: int, verbose: bool,
                 source: int, batch_id: int = 0, period=None):
        super().__init__(time, SKU, order_number, batch_id, verbose, 'delivery',
                         period)
        self.source = source
        if self.verbose:
            print("event created: ", self)

    def handle(self, state: State) -> EventHandleInfo:
        """creates a delivery first leg travel event if there are free AGVs,
        updates agv cache. if there are no free AGVs, the event gets added
        to queued_delivery_orders"""
        super().handle(state)
        sc: LocationManager = state.location_manager
        if state.agv_manager.agv_available() and state.delivery_possible():
            agv: AGV = state.agv_manager.book_agv(
                state.source_positions[self.source], state.time)
            state.agv_manager.update_v_matrix(agv.position, None)
            travel_event = DeliveryFirstLeg(
                state=state, start_point=agv.position,
                end_point=state.source_positions[self.source],
                travel_type="delivery_first_leg",
                level=0, source=self.source, order=self, agv_id=agv.id)
            state.add_travel_event(travel_event)
            return EventHandleInfo(
                False, travel_event, travel_event, None, None)
        else:
            if self.verbose:
                print("no delivery AGV available, adding to queued events")
            return EventHandleInfo(False, None, None, None, self)

    def __str__(self):
        return (f'delivery order #{self.order_number} for SKU {self.SKU} that '
                f'arrives at {self.time} at source {self.source}')

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)


class Retrieval(Order):
    """retrieval order contains a SKU number and sink position. seizes a free
    AGV, moves it to the pallet, picks it up, then moves it to the appropriate
    sink position in the warehouse and drops off the pallet.

    """
    def __init__(self, time: float, SKU: int, order_number: int, verbose: bool,
                 sink: int, batch_id: int = 0, period=None):
        super().__init__(time, SKU, order_number, batch_id, verbose,
                         "retrieval", period)
        self.sink = sink
        if self.verbose:
            print("event created: ", self)

    def handle(self, state: State):
        """3 different scenarios here.
        1) if the SKU number for the retrieval order is not in the warehouse,
        the retrieval order gets added to queued retrieval orders
        2) if the SKU is serviceable and there are free AGVs, action_needed
        returns True so that the agent can create a retrieval order with the
        specific pallet location coming from an agent or retrieval policy
        3) if there are no free AGVs, the retrieval order gets queued and is
        added to the queued_retrieval_orders dictionary
        """
        super().handle(state)
        lm: LocationManager = state.location_manager
        if self.SKU not in lm.lane_manager.occupied_lanes:
            if self.verbose:
                print("SKU not available.")
            return EventHandleInfo(False, None, None, self, None)
        if not lm.get_sku_locations(
                self.SKU, state.trackers.travel_event_statistics):
            if self.verbose:
                print("SKU available but is currently locked")
            return EventHandleInfo(False, None, None, self, None)

        # let step() and agent create the retrieval travel event
        if state.agv_manager.agv_available():
            return EventHandleInfo(True, None, None, None, None)
        else:
            if self.verbose:
                print("no retrieval AGV available, adding to queued events")
            return EventHandleInfo(False, None, None, self, None)

    def __str__(self):
        return (f'retrieval order #{self.order_number} for SKU {self.SKU} '
                f'arrives at {self.time} at sink {self.sink}')

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)


class Travel(Event):
    """travel events always have a route and describe what tiles an AGV should
    go through in order to reach a destination
    """
    def __init__(self, state: State, start_point: Tuple[int, int],
                 end_point: Tuple[int, int], travel_type: str,
                 level: int, order: Order, key: TravelEventKeys, agv_id: int):
        """
        state: State
        start_point: Tuple[int, int]
            first node/tile in the route
        end_point: Tuple[int, int]
            last node/tile in the route
        SKU: int
            the sku number for the order that the travel event is completing
        travel_type: str
            there are four different types of travel. each is its own object
            but is also described here. values can be: retrieval_first_leg,
            retrieval_second_leg, delivery_first_leg, or delivery_second_leg
        order_number: int
            the order number for the order that the travel event is completing
        verbose: bool
            debugging, slapstack_controls print statements
        level: int
            routes only contain 2D coordinates, so level is saved here.
            for retrieval travel events, defines the level that a pallet is
            being retrieved from. for delivery travel events, defines the
            level where a pallet should be placed.

        """
        self.key = key
        self.agv_id = agv_id
        route = Route(state.routing, state.location_manager.grid,
                      start_point, end_point)
        super().__init__(state.time + route.get_duration(), order.verbose)
        self.route = route
        state.add_route(route.get_indices())
        self.travel_type = travel_type
        self.level = level
        if self.verbose:
            print("event created: ", self)
            print("route created:")
            print(self.route.plot_route())
        # convenience variables set during handle
        self.first_node = self.route.get_first_node()
        self.last_node = self.route.get_last_node()
        self.order = order
        state.update_on_travel_event_creation(self.key)

    def __hash__(self):
        return hash(str(self.order.order_number) + self.travel_type)

    def __eq__(self, other):
        """used for sorting events in heaps"""
        return (self.order.order_number == other.order.order_number
                and self.travel_type == other.travel_type)

    def handle(self, state: State):
        """executed for all types of travel events - correct nodes are set
        based on current route, travel time and distance traveled are calculated
        for trackers"""
        tk = self.key
        d = self.route.get_total_distance()
        t = self.route.get_duration()
        state.update_on_travel_event_completion(tk, t, d)

    def partial_step_handle(self, state: State, elapsed_time: float):
        previous_first_node = self.route.get_first_node()
        state.remove_route(self.route.get_indices())
        n_tiles = len(self.route.midpoints)
        i = 0
        while i < n_tiles:
            if self.route.midpoints[i] <= elapsed_time:
                i += 1
            else:
                # pointless to keep iterating since midpoint times are
                # monotonously increasing ;)
                break
        for j in range(i):
            del self.route.indices[0]
            del self.route.midpoints[0]
        state.add_route(self.route.get_indices())
        cur_total_time = self.route.get_duration()
        new_first_node = self.route.get_indices()[0]
        state.agv_manager.update_v_matrix(
            previous_first_node, new_first_node)

    def __str__(self):
        return (f'{self.travel_type} travel with SKU {self.order.SKU} finishes '
                f'at {self.time} and takes route {self.route} with duration '
                f'{self.route.get_duration()} to level {self.level}')


class RetrievalFirstLeg(Travel):
    """first leg of a travel event used to complete a retrieval order. a free
     AGV is seized and moves from its current position to the tile from where
    it is picking up a pallet
    """
    def __init__(self, state: State, start_point: Tuple[int, int],
                 end_point: Tuple[int, int], travel_type: str,
                 level: int, sink: int,
                 order: Order, agv_id: int):
        super().__init__(state, start_point, end_point, travel_type,
                         level, order, TravelEventKeys.RETRIEVAL_1STLEG, agv_id)
        self.sink = sink
        self.delivery_action, self.retrieval_actions = (state.location_manager
                                                        .lock_lane(end_point))

    def handle(self, state: State):
        """updates agv position in vehicle matrix, removes sku from storage
        and arrival time matrix, removes SKU from occupied locations in cache,
        removes route, removes travel event, creates retrieval second leg travel
        event.

        returns:
        action_needed = False
        event_to_add = travel_event
        travel_event_to_add = travel_event
        queued_retrieval_order_to_add = None
        queued_delivery_order_to_add = None
        """
        super().handle(state)
        pallet_position = self.last_node + (self.level,)
        pallet_cycle_time = state.time - state.T[pallet_position]
        state.update_s_t_b_matrices(pallet_position, StorageKeys.EMPTY.value,
                                    TimeKeys.NAN.value, BatchKeys.NAN.value)
        state.location_manager.unlock_lane(self.delivery_action,
                                           self.retrieval_actions)
        assert len(pallet_position) == 3

        state.agv_manager.update_v_matrix(self.first_node, self.last_node)
        state.remove_route(self.route.get_indices())
        state.remove_travel_event(self.order.order_number)

        shift_penalty = state.location_manager.update_on_retrieval_first_leg(
            pallet_position, self.order.SKU, pallet_cycle_time, self.time)
        state.trackers.number_of_pallet_shifts += shift_penalty
        state.n_skus_inout_now[self.order.SKU] -= 1
        travel_event = RetrievalSecondLeg(
            state=state, start_point=self.last_node,
            end_point=state.sink_positions[self.sink],
            travel_type="retrieval_second_leg",
            level=0, sink=self.sink,
            n_shifts=shift_penalty, order=self.order, agv_id=self.agv_id)
        state.add_travel_event(travel_event)
        return EventHandleInfo(False, travel_event, travel_event, None, None)

    def __str__(self):
        return super().__str__()

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)


class RetrievalSecondLeg(Travel):
    """second leg of a travel event used to complete a retrieval order. a busy
     AGV moves from its current position (where it just picked up a pallet) to
     the sink tile where it will drop off the pallet
    """
    def __init__(self, state: State, start_point: Tuple[int, int],
                 end_point: Tuple[int, int], travel_type: str,
                 level: int, sink: int,
                 n_shifts: int, order: Order, agv_id: int):
        super().__init__(state, start_point, end_point, travel_type, level,
                         order, TravelEventKeys.RETRIEVAL_2ND_LEG, agv_id)
        self.sink = sink
        self.time += n_shifts * state.params.shift_penalty

    def handle(self, state: State):
        """updates agv position in vehicle matrix, removes route, removes travel
        event.
        Since multiple AGVs can end up at the same spot (sink tile), this
        function tries to make sure that they are placed on different tiles in
        the vehicle matrix

        returns:
        action_needed = False
        event_to_add = None
        travel_event_to_add = None
        queued_retrieval_order_to_add = None
        queued_delivery_order_to_add = None
        """
        super().handle(state)
        # if the AGV has not reached the sink tile yet or if it's already
        # there (possibly from a simulate_travel_events() advancing it
        # to the last node in its route
        tile_found = False
        offset = 0
        state.remove_route(self.route.get_indices())
        while not tile_found:
            vehicle_position = (self.last_node[0] + offset, self.last_node[1])
            value_at_sink = state.V[vehicle_position]
            if (value_at_sink == VehicleKeys.N_AGV or
                    value_at_sink == VehicleKeys.BUSY):
                last_node = (self.last_node[0] + offset, self.last_node[1])
                state.agv_manager.update_v_matrix(
                    self.first_node, self.last_node, True)
                state.agv_manager.release_agv(
                    last_node, state.time, self.agv_id)
                state.remove_route(self.route.get_indices())
                state.remove_order(self.order.order_number)
                state.remove_travel_event(self.order.order_number)
                tile_found = True
            else:
                if self.verbose:
                    print("already an AGV at this tile, "
                          "checking next tile for free space")
                offset += 1
        if self.verbose:
            print(f"finished retrieval order #{self.order.order_number}")
        self.order.set_completion_time(state.time)
        state.trackers.update_on_order_completion(self.order)
        return EventHandleInfo(False, None, None, None, None)

    def __str__(self):
        return super().__str__()

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)


class DeliveryFirstLeg(Travel):
    """first leg of a travel event used to complete a delivery order. a free
     AGV is seized and moves from its current position to the source tile where
    it will pick up a pallet
    """
    def __init__(self, state: State, start_point: Tuple[int, int],
                 end_point: Tuple[int, int], travel_type: str,
                 level: int, source: int, order: Order, agv_id: int):
        super().__init__(state, start_point, end_point, travel_type,
                         level, order, TravelEventKeys.DELIVERY_1ST_LEG, agv_id)
        self.source = source

    def handle(self, state: State):
        """vehicle position is updated. since delivery second leg needs an
        action from the agent or policy, it is not created here.

        returns:
        action_needed = True
        event_to_add = None
        travel_event_to_add = None
        queued_retrieval_order_to_add = None
        queued_delivery_order_to_add = None
        """
        super().handle(state)
        state.remove_travel_event(self.order.order_number)
        state.agv_manager.update_v_matrix(
            self.first_node, self.last_node, False)
        state.remove_route(self.route.get_indices())
        return EventHandleInfo(True, None, None, None, None)

    def __str__(self):
        return super().__str__()

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)


class DeliverySecondLeg(Travel):
    """second leg of a travel event used to complete a retrieval order. a busy
     AGV moves from its current position (where it just picked up a pallet) to
     the sink tile where it will drop off the pallet
    """
    def __init__(self, state: State, start_point: Tuple[int, int],
                 end_point: Tuple[int, int], travel_type: str,
                 level: int, source: int, order, agv_id: int):
        super().__init__(state, start_point, end_point, travel_type,
                         level, order, TravelEventKeys.DELIVERY_2ND_LEG, agv_id)
        self.source = source
        self.delivery_action, self.retrieval_actions = (state.location_manager
                                                        .lock_lane(end_point))

    def handle(self, state: State):
        """updates agv position in vehicle matrix, removes route, removes travel
        event. adds correct SKU number to storage matrix and the current time
        to the arrival time matrix

        returns:
        action_needed = False
        event_to_add = None
        travel_event_to_add = None
        queued_retrieval_order_to_add = None
        queued_delivery_order_to_add = None
        """
        super().handle(state)
        state.remove_travel_event(self.order.order_number)
        state.agv_manager.update_v_matrix(self.first_node, self.last_node, True)
        state.remove_route(self.route.get_indices())
        pallet_position = self.last_node + (self.level,)
        state.location_manager.unlock_lane(self.delivery_action,
                                           self.retrieval_actions)
        state.update_s_t_b_matrices(pallet_position, self.order.SKU, state.time,
                                    self.order.batch_id)
        storage_location = self.last_node + (self.level,)
        state.location_manager.zone_manager.add_out_of_zone_sku(
            pallet_position, self.order.SKU)
        assert len(storage_location) == 3

        state.remove_order(self.order.order_number)
        # no shift penalty can occur at this point, at least not for the
        # "pure lane" simulation mode
        state.n_skus_inout_now[self.order.SKU] += 1
        state.agv_manager.release_agv(
            pallet_position[:-1], self.time, self.agv_id)
        _ = state.location_manager.update_on_delivery_second_leg(
            pallet_position, self.order.SKU, self.time, self.order.batch_id)

        if self.verbose:
            print("finished delivery order #{0}".format(
                self.order.order_number))
        self.order.set_completion_time(state.time)
        state.trackers.update_on_order_completion(self.order)
        return EventHandleInfo(False, None, None, None, None)

    def __str__(self):
        return super().__str__()

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)


class EventManager:
    def __init__(self):
        self.running: MutableSequence[Event] = []
        self.queued_delivery_orders: Deque[Delivery] = deque([])
        self.queued_retrieval_orders: Dict[int, Deque[Retrieval]] = {}
        self.current_travel: Set[Travel] = set({})
        self.__verbose = False
        self.__retrieval_possible = False
        self.__delivery_possible = False
        self.__state_changed_retrieval = True
        self.__state_changed_delivery = True
        self.__n_queued_retrieval_orders = 0
        self.__n_queued_delivery_orders = 0
        self.__earliest_delivery_order = None
        self.__earliest_retrieval_order = None

    @property
    def n_queued_retrieval_orders(self):
        return self.__n_queued_retrieval_orders

    @property
    def n_queued_delivery_orders(self):
        return self.__n_queued_delivery_orders

    def add_future_event(self, event: Event):
        self.__state_changed_retrieval = True
        self.__state_changed_delivery = True
        heap.heappush(self.running, event)

    def add_current_events(
            self, event_queueing_info: EventHandleInfo):
        """this function takes the events that were returned from handling an
        event and adds them to their appropriate data structures. For example,
        if a travel event was created when handling a delivery order, it will
        be added to self.events.current_travel.
        """
        if event_queueing_info.event_to_add:
            self.add_future_event(event_queueing_info.event_to_add)
        if event_queueing_info.travel_event_to_add:
            self.__state_changed_retrieval = True
            self.__state_changed_delivery = True
            initial_len = len(self.current_travel)
            self.current_travel.add(event_queueing_info.travel_event_to_add)
            assert initial_len + 1 == len(self.current_travel)
        if event_queueing_info.queued_retrieval_order_to_add:
            self.__print(f"added retrieval order to queue: "
                         f"{event_queueing_info.queued_retrieval_order_to_add}")
            self.__queue_retrieval_order(
                event_queueing_info.queued_retrieval_order_to_add)
        if event_queueing_info.queued_delivery_order_to_add:
            self.__print(f"added delivery order to queue: "
                         f"{event_queueing_info.queued_delivery_order_to_add}")
            self.__queue_delivery_order(
                event_queueing_info.queued_delivery_order_to_add)

    def __queue_delivery_order(self, order: Delivery):
        self.__state_changed_delivery = True
        self.__n_queued_delivery_orders += 1
        self.queued_delivery_orders.append(order)

    def __queue_retrieval_order(self, order: Retrieval):
        self.__n_queued_retrieval_orders += 1
        self.__state_changed_retrieval = True
        sku = order.SKU
        if sku in self.queued_retrieval_orders:
            self.queued_retrieval_orders[sku].append(order)
        else:
            self.queued_retrieval_orders[sku] = deque([order])

    def pop_future_event(self):
        self.__state_changed_retrieval = True
        self.__state_changed_delivery = True
        return heap.heappop(self.running)

    def pop_queued_event(self, state: State) -> Event:
        """this method is only executed if there are queued orders that can be
        handled and there are free AGVs. It is a bit hairy because of many
        specific if statements but these are the three conditions below. Note
        that retrieval orders have higher priority than delivery orders.
        1) if there is at least one serviceable retrieval order, handle
        whichever order one is oldest (i.e. queued first)
        2) if there are no queued retrieval orders, but there are queued
        delivery orders that can be serviced (i.e. there is space in the
        warehouse/ there are legal actions), handle the oldest delivery order
        3) if there are both serviceable queued retrieval orders and queued
        delivery orders, handle whichever one is oldest.
        Finally, since the queued events time are in the past, they are updated
        to be the current state time.
        return the event to be handled
        """
        next_event = None
        if self.__state_changed_retrieval:
            self.available_retrieval(state)
        if self.__state_changed_delivery:
            self.available_delivery(state)
        self.__print("picking from queued events")
        if self.__retrieval_possible and not self.__delivery_possible:
            next_event = self.pop_queued_retrieval_order()
        elif self.__delivery_possible and not self.__retrieval_possible:
            next_event = self.pop_queued_delivery_order()
        elif self.__retrieval_possible and self.__delivery_possible:
            if (self.__earliest_retrieval_order.time
                    <= self.__earliest_delivery_order.time):
                next_event = self.pop_queued_retrieval_order()
            else:
                next_event = self.pop_queued_delivery_order()
        # the queued order's time has already passed so it must be
        # updated to current time
        # TODO: Fix service times?
        # if next_event:
        #     next_event.time = state.time
        self.__state_changed_retrieval = True
        return next_event

    def add_travel_event(self, event: Travel):
        """
        Adds the travel event to both the future events queue and the running
        travel events queue.

        :param event: The new travel event wich is to finish at some time in the
            future.
        :return: None.
        """
        self.current_travel.add(event)
        self.add_future_event(event)

    def remove_travel_event(self, next_event: Travel):
        self.__state_changed_retrieval = True
        self.__state_changed_delivery = True
        initial_len = len(self.current_travel)
        self.current_travel.remove(next_event)
        assert initial_len - 1 == len(self.current_travel)

    def available_retrieval(self, state: State):
        if not self.__state_changed_retrieval:
            return self.__retrieval_possible
        query_result = False
        if self.__n_queued_retrieval_orders > 0:   # there are queued orders
            minimum_time = math.inf
            oldest_order_possible = None
            for sku, retrieval_orders in self.queued_retrieval_orders.items():
                first_order_time = retrieval_orders[0].time
                if state.retrieval_possible(sku):
                    if first_order_time < minimum_time:
                        minimum_time = first_order_time
                        oldest_order_possible = retrieval_orders[0]
            if oldest_order_possible:
                self.__earliest_retrieval_order = oldest_order_possible
                query_result = True
        self.__state_changed_retrieval = False
        self.__retrieval_possible = query_result
        return query_result

    def available_delivery(self, state: State):
        """

        :param state:
        :return:
        """
        if not self.__state_changed_delivery:
            return self.__delivery_possible
        if not self.queued_delivery_orders:
            query_result = False
        else:
            query_result = state.delivery_possible()
            self.__earliest_delivery_order = self.queued_delivery_orders[0]
        self.__delivery_possible = query_result
        self.__state_changed_delivery = False
        return query_result

    def all_orders_complete(self):
        if (not self.running
                and self.__n_queued_retrieval_orders == 0
                and self.__n_queued_delivery_orders == 0):
            return True
        else:
            return False

    def pop_queued_delivery_order(self) -> Delivery:
        """
        Returns the oldest serviceable queued retrieval order.

        :return: The oldest order.
        """
        self.__state_changed_retrieval = True
        self.__n_queued_delivery_orders -= 1
        return self.queued_delivery_orders.popleft()

    def pop_queued_retrieval_order(self) -> Retrieval:
        """
        Return the oldest serviceable retrieval order and removes it from the
        queue. Since the queued retrieval orders are stored in a sku indexed
        dictionary of deques, and orders are queued by appending to the right,
        we only need to look at all the deque heads to find the oldest queued
        order.

        :return: The oldest serviceable retrieval order.
        """
        assert not self.__state_changed_retrieval
        self.__state_changed_delivery = True
        self.__n_queued_retrieval_orders -= 1
        target_sku = self.__earliest_retrieval_order.SKU
        order = self.queued_retrieval_orders[target_sku].popleft()
        if not self.queued_retrieval_orders[target_sku]:
            del self.queued_retrieval_orders[target_sku]
        return order

    def get_min_retrival_order_time(self):
        """
        Computes the minimum retrieval order time.

        :return: The time of the oldest retrieval order.
        """
        min_time = np.inf
        for orders in self.queued_retrieval_orders.values():
            if orders[0].time < min_time:
                min_time = orders[0].time
        return min_time

    def __print(self, string: Any):
        """this function can be used instead of the python default print(). It
        allows all print statements to be turned on/off with one parameter:
        __verbose
        """
        if self.__verbose:
            print(string)

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)
