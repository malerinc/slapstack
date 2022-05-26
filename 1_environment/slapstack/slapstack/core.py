import time
from typing import Tuple, List, Any

import gym
import numpy as np
from slapstack.core_events import (Retrieval, RetrievalFirstLeg,
                                   DeliverySecondLeg, Travel, Delivery, Event,
                                   DeliveryFirstLeg, RetrievalSecondLeg)
from slapstack.core_state import State
from slapstack.core_state_agv_manager import AGV
from slapstack.core_state_lane_manager import LaneManager
from slapstack.core_state_location_manager import LocationManager
from slapstack.extensions import c_unravel
from slapstack.helpers import faster_deepcopy
from slapstack.core_logger import MatrixLogger
from slapstack.core_events import EventManager, EventHandleInfo
from slapstack.interface_input import Input
from slapstack.interface_templates import SimulationParameters, SlapLogger


# from line_profiler_pycharm import profile


class SlapCore(gym.Env):
    """

    """
    rng: np.random.default_rng = np.random.default_rng()

    def render(self, mode='human'):
        pass

    def __init__(self, usr_inpt: Input, logger: SlapLogger):
        """
        initializes an environment that has a size of
        n_rows x n_columns x n_levels.
        the slap instance is represented by three n_rows x n_columns x n_levels
        matrices.
        inpt: dict
            dictionary of values that are used to create SlapCore and
            its object inpt (SlapWarehouse, SlapOrders, etc.). Keys
            include inpt such as number of rows, number of columns,
            initial pallet inpt (storage strategy if any, sku counts if
            any)
        seeds: dict
            random seeds that are used to control stochasticity. seeds are
            used in numpy and random modules
        warehouse: SlapWarehouse
        S: np.array
            storage matrix - numpy array where values represent types of tiles
            (walls, aisles, source, sink, and storage)
        V: np.array
            vehicle matrix - numpy array where values represent AGVs and their
            status (busy, free, etc.)
        T: np.array
            arrival time matrix - numpy array where values represent at what
            time an sku was placed on that tile
        state: State
        orders: SlapOrders
        resetting: bool
           parameter used for debugging - becomes true once reset()
            is executed
        __verbose: bool
            parameter used for debugging - if set to true, then all
            print() statements are activated
        SKU_counts: dict
            dictionary where keys are SKU numbers and values are
            how many pallets with that SKU are in the warehouse - at times, may
            not be 100% accurate, i.e. when a retrieval order is placed, this
            gets updated, and not exactly when the pallet gets picked up
        events: SlapEvents
        previous_event: Event
            saves the previous event so its inpt can be
            accessed in future
        decision_mode: str
            used to keep track of what type of event needs to be
            created next and what type of actions are legal -
            can be either delivery or retrieval
        legal_actions: list of 3D tuples
            tuples represent one location in the numpy
            matrices. can represent what pallets can be retrieved during
            retrieval orders, or what storage locations are empty during
            delivery orders
        state_stack_size: int
            how many states should be stored in state_stack
        state_stack: deque
            saves concatenated state and kicks out
            past states if it doesn't fit
            saves each state's storage matrix into a list
        initial_state: np.array
            first value in state_stack
        logger: Logger
        """
        self.inpt = usr_inpt
        self.rng = None
        self.set_seed(usr_inpt.seed)
        self.events = EventManager()
        self.state = State(usr_inpt.params, self.events, self.rng)
        self.orders = SlapOrders(usr_inpt.params,
                                 self.state.get_n_storage_locations())
        self.resetting = usr_inpt.params.resetting
        self.verbose = usr_inpt.params.verbose
        self.SKU_counts = {i: 0 for i in range(1, self.orders.n_SKUs + 1)}
        self.previous_event = None
        self.decision_mode = "delivery"
        self.legal_actions = None
        # includes state after reset(), step_no_action(), and step()
        if isinstance(logger, str):
            self.logger = MatrixLogger(logger)
        elif isinstance(logger, SlapLogger):
            self.logger = logger
        else:
            raise ValueError("Unsupported logger type. Pass either a relative "
                             "directory path or an object of type 'SlapLogger' "
                             "as defined by the interface_templates.py module. "
                             "To deactivate logging pass the empty string.")

    # def __init_legal_actions(self):
    #     sc = self.state.sc
    #     ual = sc.get_unassigned_open_storage_locations()
    #     for sku in range(1, self.state.n_skus + 1):
    #         if sku in sc.occupied_lanes.keys():
    #             sc.legal_retrieval_actions[sku] = (
    #                 self.state.sc.get_sku_locations(sku))
    #         if sku not in sc.sku_lanes:
    #             sc.legal_delivery_actions[-1] = ual
    #             assert sc.legal_delivery_actions[-1]
    #             sc.legal_delivery_actions[sku] = sc.legal_delivery_actions[-1]
    #         else:
    #             sc.legal_delivery_actions[sku] = sc.get_open_locations(sku)

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def create_orders_from_distribution(self):
        """creates delivery and retrieval orders automatically based on input
        inpt. SKUs are selected randomly and times are created
        with a distribution based on number of storage locations
        """
        self.events.running = []
        sim_time = 0

        # np.random.seed(1)

        def random_sku():
            return int(SlapCore.rng.integers(1, self.orders.n_SKUs + 1))
        # random_sku = lambda: random.randint(1, self.n_SKUs)

        # make a copy of SKU_counts because the self variable will be used
        # to keep track of stock during simulation but copy is just used to
        # create orders and make sure retrieval orders aren't created
        # when no stock would be available and delivery orders aren't created
        # when no spaces would be available
        available_source_tiles = [i for i in
                                  range(0, self.inpt.params.n_sources)]
        available_sink_tiles = [i for i in range(0, self.inpt.params.n_sinks)]
        SKU_counts = self.SKU_counts.copy()
        for i in range(1, self.orders.n_orders):
            total_pallets = sum(SKU_counts.values())
            order_time = self.get_order_time()
            order_type = self.get_order_type(total_pallets)
            if order_type == "delivery":  # delivery order
                source_index = SlapCore.rng.choice(available_source_tiles)
                self.create_delivery_order(SKU_counts, i, random_sku,
                                           sim_time, source_index)
            elif order_type == "retrieval":  # retrieval order
                sink_index = SlapCore.rng.choice(available_sink_tiles)
                self.create_retrieval_order(SKU_counts, i,
                                            sim_time, sink_index)
            sim_time += order_time
        self.print_events()

    def print_events(self):
        """just prints first 5 orders when calling reset() and after orders
        are added to heap
        """
        if self.resetting and self.verbose:
            self.print("first 5 events")
            for i in range(0, 5):
                self.print(self.events.running[i])

    def get_order_type(self, total_pallets: int) -> str:
        """determines order type based on how full warehouse is at that point.
        intended to keep number of pallets between a certain range. if it's
        already between that range, then it's a random choice between
        delivery and retrieval
        """
        order_type = SlapCore.rng.choice(["delivery", "retrieval"])
        # if too many pallets in warehouse,
        # make next order a retrieval order
        if total_pallets > 1.2 * self.orders.average_n_pallets:
            order_type = "retrieval"
        # if too few pallets in warehouse, make next order a delivery order
        if total_pallets < 0.8 * self.orders.average_n_pallets:
            order_type = "delivery"
        return order_type

    def get_order_time(self) -> int:
        """gets arrival time of retrieval and delivery orders. based on a
        normal distribution with set mean and standard deviation. intended to
        create some overlap of orders and travel times but not every single one.
        """
        mean_order_time = self.state.get_n_storage_locations() * 0.4
        std_order_time = self.state.get_n_storage_locations() * 0.1
        order_time = SlapCore.rng.normal(mean_order_time, std_order_time, 1)[0]
        return order_time

    def create_delivery_order(self, sku_counts: dict, i: int, random_sku,
                              sim_time: int, source: int):
        """get random sku, create delivery order, push to running event heap,
        and update sku_counts dictionary
        """
        sku = random_sku()
        self.events.add_future_event(
                      Delivery(sim_time, sku, i, self.verbose, source))
        sku_counts[sku] += 1

    def create_retrieval_order(self, sku_counts: dict, i: int, sim_time: int,
                               sink: int):
        """get random feasible sku, create retrieval order, push to running
        event heap, and update sku_counts dictionary
        """
        # if there are no pallets for a specific SKU available,
        # do not make a retrieval order for it
        possible_skus = [sku for sku in range(1, self.orders.n_SKUs + 1)]
        for j in possible_skus:
            if sku_counts[j] == 0:
                possible_skus.remove(j)
        sku = SlapCore.rng.choice(possible_skus)
        self.events.add_future_event(
            Retrieval(sim_time, sku, i, self.verbose, sink))
        sku_counts[sku] -= 1

    # def add_silent_storage_state(self):
    #     """add storage matrix np array to storage_matrix_history during
    #     step_no_action()
    #     """
    #     self.storage_matrix_history.append(self.state.S)

    def reset(self):
        """this function should be called to initialize and/or reset the
        slap environment to its initial state. It initializes inpt,
        adds initial pallets, creates orders, logs states. Lastly, it executes
        step_no_action() so that the steps that don't require actions are
        executed and environment is ready to accept an action afterwards"""
        self.print("~" * 150 + "\n" + "reset\n" + "~" * 150)
        self.__init__(self.inpt, self.logger)
        self._assert_orders()
        if not self.orders.generate_orders:
            self.create_orders_from_list()
        else:
            self.create_orders_from_distribution()
        self.state.add_orders(self.events.running)
        self._add_initial_pallets()
        assert self.events.running
        self.logger.log_state(self.state)
        # self.add_silent_storage_state()
        # self.__init_legal_actions()
        self.step_no_action()

    # @profile
    def step_no_action(self):
        """
        this function is the second most important one in the simulation.
        it executes all of the events in the simulation that do not require an
        action, logs each state, and sets legal actions for the method that
        will be called: step()

        If all of the future events have been handled or deferred and none of
        the queued retrieval/delivery orders cannot be serviced, this function
        will indicate simulation termination by returning True.

        :return: True if the simulation has ended and False otherwise.
        """
        s = self.state
        action_needed, e, sc = False, self.events, s.location_manager
        sc.invalidate_sku_location_cache()
        # loop through simulation until an action
        # is needed (for delivery second leg event)
        # or until there are no more events or queued orders to process
        retrieval_ok = e.available_retrieval(s)
        delivery_ok = e.available_delivery(s)
        while (not action_needed
               and (e.running or retrieval_ok or delivery_ok)):
            self.print("~" * 150 + "\n" + "step no action \n" + "~" * 150)
            next_event = None
            # if there are serviceable queued events, take care of them first.
            if (retrieval_ok or delivery_ok) and s.agv_manager.agv_available():
                next_event = e.pop_queued_event(s)
            if next_event is None:
                if not e.running:
                    return True
                next_event = e.pop_future_event()
            action_needed = self.handle_event_and_update_env(next_event)
            sc.invalidate_sku_location_cache()
            retrieval_ok = e.available_retrieval(s)
            delivery_ok = e.available_delivery(s)
        if not e.all_orders_complete():
            # update legal actions for next step()
            self.legal_actions = self.get_legal_actions()
            self.print("legal actions for " + self.decision_mode +
                       " order: " + str(self.legal_actions))
        return False

    # @profile
    def handle_event_and_update_env(self, next_event: Event) -> bool:
        """
        This function takes care of the bookkeeping, from handling an event -
        (manually) removes any travel events, calculates time between handled
        and previous event, updates state time, adds future events to
        appropriate data structures, logs states, and returns a boolean that
        decides if another event should be handled or if it should move on to
        step().

        The simulation time is updated using the event occurrence time
        maintained by the event object. However, if the event was previously
        queued order, its actual occurence time lies in the past. In such
        cases, the simulation time will not be updated.

        :param next_event: An Event object popped from either the future events
            or queued_delivery/retrieval_orders in the EventManager.
        :return: True if the Event handled requires simulation control action
            (storage/retrieval decisions)
        """
        if next_event in self.events.current_travel:
            # noinspection PyTypeChecker
            next_event: Travel
            self.events.remove_travel_event(next_event)
        self.print("current time: " + str(next_event.time))
        self.print("popped event: " + str(next_event))
        elapsed_time = round(next_event.time -
                             self.state.time, 2)
        self.state.trackers.update_time(elapsed_time)
        #  travel events need to be updated by time_to_simulate
        self.process_travel_events(elapsed_time)
        if next_event.time > self.state.time:
            self.state.time = next_event.time
        # handle event and see if an action is needed and what data
        # structures should the next (or same) event be added to
        event_queueing_info: EventHandleInfo = next_event.handle(self.state)
        self.events.add_current_events(event_queueing_info)
        self.update_prev_event_and_curr_order(
            next_event, event_queueing_info.queued_retrieval_order_to_add)
        self.print(self.state)
        self.print("SKU counts: " + str(self.SKU_counts))
        if self.verbose:
            self.print_any_events()
        self.state.n_silent_steps += 1
        self.logger.log_state(self.state)
        # self.add_silent_storage_state()
        return event_queueing_info.action_needed

    def __print_debug_info(self, action: Tuple[int, int, int]):
        """just prints time and action taken"""
        self.print("~" * 150 + "\n" + "step with action \n" + "~" * 150)
        self.print("time: " + str(self.state.time))
        self.print("given action: " + str(action))

    # @profile
    def step(self, action: int) -> Tuple[State, bool]:
        """key function that takes an action from an agent, randomly, or from
        a storage or retrieval strategy. first action is converted from an int
        to a 3D tuple that fits the warehouse shape, then depending on if the
        current order is a delivery or retrieval order, it creates a
        DeliverySecondLeg or RetrievalFirstLeg event, respectively. Updates
        event data structures, and executes step_no_action(). If the simulation
        is done afterwards, it can be ended here."""
        action = c_unravel(action, self.inpt.params.shape)
        self.state.n_steps += 1
        self.__print_debug_info(tuple(action))
        travel_event = None
        if self.decision_mode == "delivery":
            travel_event = self.__create_event_on_delivery(action)
        elif self.decision_mode == "retrieval":
            travel_event = self.__create_event_on_retrieval(action)
        self.state.add_travel_event(travel_event)
        self.events.add_travel_event(travel_event)
        self.logger.log_state(self.state)
        done_prematurely = self.step_no_action()
        return self.__has_ended(done_prematurely)

    def __has_ended(self, done_prematurely: bool) -> (State, bool):
        # sc, e = self.state.location_manager, self.events
        # free_agvs = sum([v for _, v in sc.free_agv_positions.items()])
        # busy_agvs = len(self.events.current_travel)
        # sc.n_visible_agvs = free_agvs + busy_agvs
        # assert sc.n_visible_agvs == sc.n_agvs - 1
        if self.events.all_orders_complete():
            self.state.done = True
            return self.state, True  # sim done
        elif done_prematurely:
            print("WARNING: Simulation ended due to an overfull warehouse or "
                  "possibly because of inconsitent i/o order definition.")
            self.state.done = True
            return self.state, True
        else:
            return self.state, False,  # state, done

    def __create_event_on_delivery(
            self, action: Tuple[int, int, int]) -> DeliverySecondLeg:
        lm: LaneManager = self.state.location_manager.lane_manager
        prev_e: DeliveryFirstLeg = self.previous_event
        if lm.pure_lanes:
            lm.add_lane_assignment(action, prev_e.order.SKU)
        travel_event = DeliverySecondLeg(
            state=self.state,
            start_point=self.state.source_positions[prev_e.source],
            end_point=action[0:2],
            travel_type="delivery_second_leg",
            level=int(action[2]),
            source=prev_e.source,
            order=prev_e.order,
            agv_id=prev_e.agv_id
        )
        return travel_event

    def __create_event_on_retrieval(
            self, action: Tuple[int, int, int]) -> RetrievalFirstLeg:
        sc: LocationManager = self.state.location_manager
        prev_e: Retrieval = self.previous_event
        agv: AGV = self.state.agv_manager.book_agv(action[0:2], self.state.time)
        self.state.agv_manager.update_v_matrix(agv.position, None)
        travel_event = RetrievalFirstLeg(
            state=self.state, start_point=agv.position,
            end_point=action[0:2],
            travel_type="retrieval_first_leg",
            level=int(action[2]),
            sink=self.previous_event.sink,
            order=prev_e, agv_id=agv.id)
        return travel_event

    def update_prev_event_and_curr_order(
            self, this_event: Event, queued_retrieval_order_to_add: Retrieval):
        """
        Updates the decision_mode and previous_event fields in the case of a
        blocking event.

        The previous_event field is used to associate order information with 
        travel events, on decisions. It only gets instantiated if the current_
        event is blocking, i.e. DeliveryFirstLeg or RetrievalOrder. After the
        decision, the order information saved will be used to generate 
        DeliverySecondLeg and RetreivalFirstLeg information events.
         
        :param this_event: The event to be saved if a decision is to be made in 
            the next step.
        :param queued_retrieval_order_to_add: The Retrieval that was added to
            the queue of currently unseviceable orders (event was unblocking and
            these updates will be overriden) or None if the oreder can be
            serviced.
        :return: None
        """
        # self.previous_event = this_event
        # TODO check that previous event isn't getting a different order
        if (isinstance(this_event, Delivery)
                or isinstance(this_event, DeliveryFirstLeg)
                or isinstance(this_event, DeliverySecondLeg)):
            self.decision_mode = "delivery"
            if isinstance(this_event, DeliveryFirstLeg):
                self.previous_event = this_event
            if isinstance(this_event, DeliverySecondLeg):
                self.SKU_counts[this_event.order.SKU] += 1
        if (isinstance(this_event, Retrieval)
                or isinstance(this_event, RetrievalFirstLeg)
                or isinstance(this_event, RetrievalSecondLeg)):
            self.decision_mode = "retrieval"
            if (isinstance(this_event, Retrieval)
                    and not queued_retrieval_order_to_add):
                self.previous_event = this_event
                self.SKU_counts[this_event.SKU] -= 1
                self.print("SKU counts: " + str(self.SKU_counts))

    def process_travel_events(self, elapsed_time: float):
        """ if time has elapsed, update any currently active travel events"""
        if self.events.current_travel and elapsed_time > 0:
            self.print("simulating travel events by " + str(elapsed_time))
            self.simulate_travel_events(elapsed_time)

    def print_any_events(self):
        """debugging purposes - print any event data structures that aren't
        empty
        """
        sc = self.state.location_manager
        if self.events.current_travel:
            self.print("currently active travel events: ")
            for i in self.events.current_travel:
                self.print("active: " + str(i))
        if self.events.n_queued_retrieval_orders():
            self.print("currently queued retrieval orders: ")
            for i in self.events.queued_retrieval_orders.values():
                for j in i:
                    self.print("queued: " + str(j))

        if self.events.queued_delivery_orders:
            self.print("currently queued delivery orders: ")
            for i in self.events.queued_delivery_orders:
                self.print("queued: " + str(i))

    def _add_initial_pallets(self):
        """ this function adds pallets/SKUs to the initial storage and arrival
        time matrices. It is versatile and can add a random quantity of SKU
        numbers or it can read from a dictionary from self.orders. The location
        of the initial pallets can either be random free locations, or they can
        follow a storage strategy (i.e. closest to source, furthest from source)
        If needed in the future, both SKU number and location can be directly
        given as an input of type numpy array
        """
        n_initial_pallets = 0
        SKU_choices = []
        # first calculate how many initial pallets there are
        # if it is already known how many initial pallets per sku there are,
        # make a list of sku_choices
        if not self.orders.read_sku_from_dict():
            n_initial_pallets = self.orders.average_n_pallets
        else:
            n_initial_pallets = sum(
                self.orders.initial_pallets_sku_counts.values())
            self.state.trackers.n_storage_locations = (
                self.orders.n_storage_locations)
            self.state.trackers.n_pallets_in_storage = n_initial_pallets
            for sku, count in self.orders.initial_pallets_sku_counts.items():
                for i in range(count):
                    SKU_choices.append(sku)
            self.SKU_counts = self.orders.initial_pallets_sku_counts
        self.print("")

        # loop for each initial pallet, get sku number to add to storage matrix.
        # either random integer or by reading from sku choices.
        # if a storage strategy is being used for the locations, it is done
        # here, otherwise location is random.
        # SKU_choices = SlapCore.rng.choice(SKU_choices, size=len(SKU_choices),
        #                                   replace=False)
        start = time.time()
        batch_id = np.float32(0)
        timestamp = np.float32(0)
        sc = self.state.location_manager
        for i in range(0, n_initial_pallets):
            if i % 1000 == 0:
                print(time.time() - start)
                print(f"[ {i} / {n_initial_pallets} ]")
            if not self.orders.read_sku_from_dict():
                chosen_sku = SlapCore.rng.integers(1, self.orders.n_SKUs + 1)
                self.SKU_counts[chosen_sku] = self.SKU_counts[
                                                  chosen_sku] + 1
            else:
                chosen_sku = int(SKU_choices[i])
            self.state.current_sku = chosen_sku
            if self.orders.follow_storage_strategy():
                index = self.orders.initial_pallets_storage_strategy.get_action(
                    self.state)
                index = c_unravel(index, self.inpt.params.shape)
            else:
                possible_locations = sc.get_open_locations(chosen_sku)
                for agv in self.state.agv_manager.get_agv_locations():
                    if agv in possible_locations:
                        possible_locations.remove(agv)
                assert len(possible_locations), 'warehouse full'
                index = SlapCore.rng.integers(0, len(possible_locations))
                index = list(possible_locations)[index]
                index = c_unravel(index, self.inpt.params.shape)
            assert len(index) == 3
            # TODO: Add randomization for initial pallets
            batch_id += np.float32(0.0001)
            timestamp += np.float32(0.0001)
            sc.update_on_delivery_second_leg(
                index, chosen_sku, timestamp, batch_id, init=True
            )
            self.state.n_skus_inout_now[chosen_sku] += 1
            # self.state.trackers.add_pallet()
            self.state.update_s_t_b_matrices(
                index, chosen_sku, timestamp, batch_id)
            sc.unlock_lane(index, [])
        self.print(str(n_initial_pallets) + " initial pallets added")
        self.print(self.SKU_counts)
        self.print(self.state)
        self.print("")

    def get_legal_actions(self) -> List[int]:
        """returns legal actions depending on what the current order is.
        actions are returned as integers as agents are better at picking 1D
        tuples than 3D
        """
        if self.decision_mode == "delivery":
            legal_actions = self.get_legal_delivery_actions()
            assert legal_actions
            return legal_actions
        elif self.decision_mode == "retrieval":
            legal_actions = self.get_legal_retrieval_actions()
            assert legal_actions
            return legal_actions

    def get_legal_retrieval_actions(self) -> List[int]:
        """returns legal actions that are all of the storage locations with
        the desired SKU number
        """
        legal_actions = self.state.location_manager.get_sku_locations(
            self.previous_event.SKU,
            self.state.trackers.travel_event_statistics)
        self.state.set_current_order(self.decision_mode)
        if self.previous_event:  # RetrievalOrder is the event type
            self.state.set_current_sku(self.previous_event.SKU)
            self.state.set_current_source_sink(self.previous_event.sink)
        self.state.set_legal_actions(legal_actions)
        return list(legal_actions)

    def get_legal_delivery_actions(self) -> List[int]:
        """returns legal actions that are all empty locations in the warehouse.
        If there are multiple levels, then it only returns the lowest space in
        the stack (of pallets)
        """
        if self.previous_event:  # DeliveryFirstLeg is the event type
            prev_e: DeliveryFirstLeg = self.previous_event
            self.state.set_current_sku(prev_e.order.SKU)
            self.state.set_current_source_sink(prev_e.source)
            self.state.set_current_order_arrival_time(prev_e.order.time)
        legal_actions = self.state.location_manager.get_open_locations(
            self.state.current_sku)
        self.state.set_current_order(self.decision_mode)
        self.state.set_legal_actions(legal_actions)
        # converts tuple legal actions to linear index
        # legal_actions = [int(np.ravel_multi_index(i, self.warehouse.shape))
        #                  for i in legal_actions]
        if not legal_actions:
            print('herehere!')
        return list(legal_actions)

    def simulate_travel_events(self, elapsed_time: float):
        # TODO move to handle?
        """if there are any currently active travel events, they are updated.
        the updated routes are also used for tracking statistics.
        """
        for travel_event in self.events.current_travel:
            self.print("before update: " + str(travel_event))
            travel_event.partial_step_handle(self.state, elapsed_time)
            self.print("after update: " + str(travel_event))

        self.print("")

    def create_orders_from_list(self):
        """takes a list of orders as an input and creates events from it
        instead of using a random distribution. An order should be represented
        as a tuple with inpt
        (order_type: str, sku: int, arrival time: int),
        such as ("delivery", 2, 300)
        """
        order_number = 1
        for order in self.orders.order_list:
            if len(order) == 4:
                order_type, sku, arrival_time, source_sink = order
                batch = 1
                source_sink -= 1
                period = None
            elif len(order) == 6:  # coming from use case
                order_type, sku, arrival_time, source_sink, batch = order[:-1]
                period = order[-1]
                source_sink -= 1
            else:
                raise ValueError("Unknown Order Structure!")
            if order_type == "retrieval":
                self.events.add_future_event(
                    Retrieval(arrival_time, sku, order_number,
                              self.verbose, source_sink, batch, period))
            elif order_type == "delivery":
                self.events.add_future_event(
                    Delivery(arrival_time, sku, order_number,
                             self.verbose, source_sink, batch, period))
            order_number += 1
        self.events.order_times = [i.time for i in self.events.running]
        self.print("")

    def print(self, string: Any):
        """this function can be used instead of the python default print(). It
        allows all print statements to be turned on/off with one parameter:
        verbose
        """
        if self.verbose:
            print(string)

    def _assert_orders(self):
        """asserts that the correct configuration of n_orders, generate_orders,
        and order_list is used"""
        if self.orders.order_list is False:
            assert self.orders.generate_orders
            assert self.orders.n_orders
        if self.orders.generate_orders is False:
            assert self.orders.order_list

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)


class SlapOrders:
    """this class groups together inpt that have to do with skus, orders,
    initial, pallets"""
    def __init__(self, params: SimulationParameters,
                 n_storage_locations: int):
        """
        n_SKUs: int
            number of stock keeping units (SKU)/unique part numbers
        n_orders: int
            the number of retrieval and delivery orders (combined) that
            are added to the initial event heap queue (if an order list is not
            given). This is the main factor in the length of a simulation.
        generate_orders: bool
            determines if the user wants the simulation to generate its own
            orders from a distribution. should be set to false if user wants to
            give a list of orders
        desired_fill_level: float
            determines what percent of warehouse the should be fill level should
            fluctuate around. represents a percentage but should be inputted as
            a number between 0.0 and 1.0. If there are 100 storage locations
            and the fill level is 0.5, the simulation will try to keep around
            an average of 50 pallets in the warehouse at all times.
        order_list: list of tuples
            format of tuple: (delivery type: str, sku: int, arrival time: int)
        initial_pallets_sku_counts: dictionary
            user can provide a dictionary to decide how many pallets for each
            sku number should be added as initial pallets instead of generating
            automatically from a distribution. format {sku: quantity}. for
            example, {1: 5, 2: 10, 3: 7} would create five pallets of sku 1,
            ten pallets of sku 2, and seven pallets of sku 3.
        initial_pallets_storage_strategy: StoragePolicy
            user can provide a storage policy to determine how initial pallets
            are placed in the warehouse. if no policy is provided, their
            locations are just selected randomly.
        average_n_pallets: int
            average number of pallets that should be in the warehouse at one
            time. can fluctuate around this number.
        finished_orders: list of orders that were successfully completed.
        """
        self.n_SKUs = params.n_skus
        self.n_orders = params.n_orders
        self.generate_orders = params.generate_orders
        if params.desired_fill_level is not None:
            assert 0.0 <= params.desired_fill_level <= 0.3
            self.desired_fill_level = params.desired_fill_level
            self.average_n_pallets = int(self.desired_fill_level
                                         * n_storage_locations)
        else:
            self.desired_fill_level = None
            self.average_n_pallets = None
        self.order_list = params.order_list
        self.initial_pallets_sku_counts = params.initial_pallets_sku_counts
        self.initial_pallets_storage_strategy = (
            params.initial_pallets_storage_strategy)
        self.n_storage_locations = n_storage_locations

    def create_times_for_histogram_with_limit(self, limit):
        delivery_times = []
        retrieval_times = []
        for order in self.order_list:
            if order[0] == "retrieval" and order[2] < limit:
                retrieval_times.append(order[2])
            elif order[0] == "delivery" and order[2] < limit:
                delivery_times.append(order[2])
        return retrieval_times, delivery_times

    def read_sku_from_dict(self) -> bool:
        """returns true if a dictionary was passed into the
        initial_pallets_sku_counts parameter
        """
        if isinstance(self.initial_pallets_sku_counts, dict):
            return True
        else:
            return False

    def follow_storage_strategy(self) -> bool:
        """returns true if a storage policy was passed into the
        initial_pallets_storage_strategy parameter"""
        if self.initial_pallets_storage_strategy is not None:
            return True
        else:
            return False

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)
