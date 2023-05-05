import numpy as np

from slapstack.core_state_route_manager import RouteManager
from slapstack.core_state_agv_manager import AgvManager
from slapstack.core_state_lane_manager import LaneManager
from slapstack.core_state_location_manager import LocationManager

from typing import List, Union, TYPE_CHECKING, Tuple
import sys

from collections import defaultdict
from slapstack.helpers import faster_deepcopy, StorageKeys, TravelEventKeys
from slapstack.interface_templates import SimulationParameters

if TYPE_CHECKING:
    from slapstack.core_events import EventManager, Order, DeliveryFirstLeg
    from slapstack.core_events import RetrievalSecondLeg, RetrievalFirstLeg


class TravelEventTrackers:
    def __init__(self):
        self.n_finished_retrieval_1st_leg = 0
        self.n_finished_retrieval_2nd_leg = 0
        self.n_finished_delivery_1st_leg = 0
        self.n_finished_delivery_2nd_leg = 0
        self.n_running_retrieval_1st_leg = 0
        self.n_running_retrieval_2nd_leg = 0
        self.n_running_delivery_1st_leg = 0
        self.n_running_delivery_2nd_leg = 0
        self.d_delivery_1st_leg = 0
        self.d_delivery_2nd_leg = 0
        self.d_retrieval_1st_leg = 0
        self.d_retrieval_2nd_leg = 0
        self.total_shift_distance = 0
        self.t_delivery_1st_leg = 0
        self.t_delivery_2nd_leg = 0
        self.t_retrieval_1st_leg = 0
        self.t_retrieval_2nd_leg = 0
        self.total_finished_travel = 0
        self.total_travel_time = 0
        self.total_distance_traveled = 0
        self.last_travel_distance = 0
        self.avg_dy = 0
        self.avg_dx = 0
        self.avg_count = 0

    @staticmethod
    def __get_suffix(event_key: TravelEventKeys):
        if event_key == TravelEventKeys.RETRIEVAL_1STLEG:
            attr_suffix = 'retrieval_1st_leg'
        elif event_key == TravelEventKeys.RETRIEVAL_2ND_LEG:
            attr_suffix = 'retrieval_2nd_leg'
        elif event_key == TravelEventKeys.DELIVERY_1ST_LEG:
            attr_suffix = 'delivery_1st_leg'
        elif event_key == TravelEventKeys.DELIVERY_2ND_LEG:
            attr_suffix = 'delivery_2nd_leg'
        else:
            raise ValueError('Unknown TravelEventKey')
        return attr_suffix

    def __update_trackers(
            self, distance: float, duration: float, shift_distance: float,
            attr_suffix: str, mark_complete):
        self.total_distance_traveled += distance
        self.total_travel_time += duration
        if not mark_complete:  # event was just initialized
            assert duration == 0.
            setattr(self, f'n_running_{attr_suffix}',
                    getattr(self, f'n_running_{attr_suffix}') + 1)
        else:  # eent has ended
            self.total_finished_travel += 1
            self.last_travel_distance = distance
            self.total_shift_distance += shift_distance
            setattr(self, f'n_running_{attr_suffix}',
                    getattr(self, f'n_running_{attr_suffix}') - 1)
            setattr(self, f'd_{attr_suffix}',
                    getattr(self, f'd_{attr_suffix}') + distance)
            setattr(self, f't_{attr_suffix}',
                    getattr(self, f't_{attr_suffix}') + duration)
            setattr(self, f'n_finished_{attr_suffix}',
                    getattr(self, f'n_finished_{attr_suffix}') + 1)

    def update_travel_events(self, event: TravelEventKeys,
                             duration: float = 0, distance: float = 0,
                             shift_distance: float = 0,
                             is_completion: bool = False):
        attr_suffix = self.__get_suffix(event)
        self.__update_trackers(
            distance=distance, duration=duration, attr_suffix=attr_suffix,
            shift_distance=shift_distance, mark_complete=is_completion)

    def get_unbound_travel_events(self):
        return self.n_running_delivery_1st_leg

    def average_travel_time(self, travel_type: TravelEventKeys = None):
        if travel_type is None:
            if self.total_finished_travel > 0:
                return self.total_travel_time / self.total_finished_travel
            else:
                return 0
        else:
            suffix = TravelEventTrackers.__get_suffix(travel_type)
            t = getattr(self, f't_{suffix}')
            n = getattr(self, f'n_finished_{suffix}')
            return t / n if n > 0 else 0

    def get_average_travel_time_retrieval(self):
        return (self.average_travel_time(TravelEventKeys.RETRIEVAL_2ND_LEG)
                + self.average_travel_time(TravelEventKeys.RETRIEVAL_1STLEG))

    def average_travel_distance(self, travel_type: TravelEventKeys = None):
        if travel_type is None:
            if self.total_finished_travel > 0:
                return self.total_distance_traveled / self.total_finished_travel
            else:
                return 0
        else:
            suffix = TravelEventTrackers.__get_suffix(travel_type)
            d = getattr(self, f'd_{suffix}')
            n = getattr(self, f'n_finished_{suffix}')
            return d / n if n > 0 else 0

    def get_average_travel_distance_retrieval(self):
        return (self.average_travel_distance(TravelEventKeys.RETRIEVAL_2ND_LEG)
                + self.average_travel_distance(TravelEventKeys.RETRIEVAL_1STLEG)
                )


class Trackers:
    """this class is used to calculate desired statistics that can be later
    used to calculate rewards in reinforcement learning loop
    finished_orders: list
        list of orders that were successfully completed.
        """

    def __init__(self, events: 'EventManager', compute_feature_trackers=True):
        # travel trackers
        self.travel_event_statistics = TravelEventTrackers()
        # time trackers
        self.time = 0
        # event trackers
        self.n_finished_travel_events = 0
        self.__queued_delivery_orders = events.queued_delivery_orders
        self.__queued_retrieval_orders = events.queued_retrieval_orders
        self.finished_orders: List[Order] = []
        # compound trackers
        self.average_service_time = 0
        self.total_service_time = 0
        # other
        self.n_storage_locations = 0
        self.n_pallets_in_storage = 0
        self.number_of_pallet_shifts = 0
        # flag
        self.compute_feature_trackers: bool = compute_feature_trackers

    def update_on_order_completion(self, order: 'Order', distance_penalty=0.0):
        self.finished_orders.append(order)
        service_time = order.completion_time - order.time
        self.total_service_time += service_time
        self.average_service_time = (self.total_service_time /
                                     len(self.finished_orders))
        if order.type == 'retrieval':
            self.n_pallets_in_storage -= 1
        elif order.type == 'delivery':
            self.n_pallets_in_storage += 1

    def add_pallet(self):
        self.n_pallets_in_storage += 1

    def update_on_travel_event_creation(self, travel_type: TravelEventKeys):
        self.travel_event_statistics.update_travel_events(travel_type)

    def update_on_travel_event_completion(
            self, travel_type: TravelEventKeys, duration, distance,
            shift_distance):
        self.travel_event_statistics.update_travel_events(
            travel_type, duration=duration, distance=distance,
            shift_distance=shift_distance, is_completion=True)

    def get_fill_level(self):
        return self.n_pallets_in_storage / self.n_storage_locations

    @property
    def n_queued_retrieval_orders(self):
        return sum([
            len(q) for sku, q in self.__queued_retrieval_orders.items()
        ])

    @property
    def n_queued_delivery_orders(self):
        return len(self.__queued_delivery_orders)

    def update_time(self, time_elapsed):
        """adds time elapsed"""
        if not self.compute_feature_trackers:
            return
        self.time += time_elapsed

    def __str__(self):
        return (f"makespan: {self.time} seconds, "
                f"total distance traveled: "
                f"{self.travel_event_statistics.total_distance_traveled}")

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)


class State:
    """
    This class contains all the variables that define the state of the warehouse
    including storage/vehicle matrices, time, current order, complete/incomplete
    orders and more.
    """
    def __init__(
            self, params: SimulationParameters, events: 'EventManager', rng):
        """
        time: float
            simulation time
        n_steps: int
            counts how many steps with action have been taken (used for logging)
        n_silent_steps: int
            counts how many steps without action have been taken (used for
            logging)
        source_positions: set of Tuple[int, int]
            index of tile where delivery orders come in from, i.e. AGVs must
            come here first in order to pick up a pallet. source_position is
            connected to an order
        sink_positions: set of Tuple[int, int]
            index of tile where retrieval orders should finish, i.e. AGVs must
            come here after picking up a pallet. sink_position is connected
            to an order
        location_manager: LocationManager
        trackers: Trackers
        routes: set of lists of Tuple[int, int]
            this is a set where current travel routes are saved. it is used in
            visualization and is updated when routes are added, completed, or
            updated. TODO not needed
        legal_actions: list
            used in visualization
        decision_mode: str
            used to determine what actions are legal
        current_sku: int
            used for storage policies
        incomplete_orders: dictionary of orders
            used in visualization to keep track of what orders still have to be
            completed
        travel_events: dictionary of travel events
            used in visualization

        """
        # simulation inpt
        self.params: SimulationParameters = params
        # matrices
        self.S = State.__init_storage_matrix(params)
        self.T = State.__init_arrival_time_matrix(params)
        self.B = State.__init_batch_id_matrix(params)
        # state managers
        self.agv_manager = AgvManager(params, self.S, rng)
        self.V = self.agv_manager.V  # Location of Vehicles
        lane_manager = LaneManager(self.S, params)
        self.routing = RouteManager(
            lane_manager, self.S, params.agv_speed, params.unit_distance,
            params.use_case_name)
        self.location_manager = LocationManager(
            self.S, self.T, self.B, lane_manager, events)
        # KPIs and trackers
        self.trackers = Trackers(events, params.compute_feature_trackers)
        # simple progression trackers
        self.time = 0
        self.n_steps = 0
        self.n_silent_steps = 0
        self.n_skus_inout_now = defaultdict(int)
        # paths relevant variables
        self.I_O_positions = State.get_io_locations(self.S)
        self.routes = set()
        # past event links
        self.current_order: Union[str, None] = None
        self.current_source_sink = 0
        self.current_sku = None
        self.order_arrival_time = None
        # debugging information
        self.incomplete_orders = {}
        self.travel_events = {}
        self.done = False
        self.current_destination = None
        self.door_to_door = params.door_to_door

    def get_mid_aisles(self):
        mid_aisles = [tuple(i[0:2]) for i in np.argwhere(self.S[:, :, 0] ==
                                                         StorageKeys.MID_AISLE)]
        return mid_aisles

    @staticmethod
    def __init_storage_matrix(p: SimulationParameters) -> np.ndarray:
        """
        Creates the initially empty storage matrix either by following a
        simple pattern or by reading the layout parameter.

        :param p: The simulation inpt provided by the user.
        :return: The storage matrix.
        """
        if p.layout is None:
            S = State.__create_storage_matrix(
                p.n_rows, p.n_columns, p.n_levels, p.n_sources, p.n_sinks)
        else:
            S = np.zeros((p.n_rows, p.n_columns, p.n_levels), dtype='int16')
            S[:, :, :] = p.layout[:, :, np.newaxis]
        return S

    @staticmethod
    def __init_arrival_time_matrix(p: SimulationParameters) -> np.ndarray:
        """
        Creates np array full of -1 values representing the absence of any
        pallets.

        :param p: The simulation inpt provided by the user.
        :return: The pallet arrival time matrix.
        """
        t = np.full((p.n_rows, p.n_columns, p.n_levels), -1.0, dtype='float32')
        return t

    @staticmethod
    def __init_batch_id_matrix(p: SimulationParameters) -> np.ndarray:
        """
        Creates np array full of -1 values representing the initial absence of
        pallets.

        :param p: The simulation inpt provided by the user.
        """
        t = np.full((p.n_rows, p.n_columns, p.n_levels), -1, dtype='float32')
        return t

    @staticmethod
    def __create_storage_matrix(n_rows: int, n_columns: int,
                                n_levels: int,
                                n_sources: int, n_sinks: int) -> np.ndarray:
        """
        Creates np array using keys that represent aisles, walls, etc.

        :param n_rows: The number of rows in the warehouse grid.
        :param n_columns: The number of columns in the warehouse grid.
        :param n_levels: The maximum levels in the warehouse.
        :param n_sources: The number of input docks.
        :param n_sinks: The number of output docks.
        :return: The storage location matrix.
        """
        s = np.zeros((n_rows, n_columns, n_levels), dtype='int16')
        src_rows = [int(n_rows / 2)]
        sink_rows = src_rows.copy()
        # 3 vertical aisles at left of warehouse
        s[:, 1:4, :] = StorageKeys.AISLE.value
        # 3 vertical aisles at right of warehouse
        s[:, (n_columns - 4): (n_columns - 1), :] = StorageKeys.AISLE.value
        # 3 horizontal aisles in middle of warehouse
        s[(src_rows[0] - 1):
          (src_rows[0] + 2), 0:n_columns - 1, :] = StorageKeys.AISLE.value
        # middle vertical aisle at right of warehouse
        s[:, n_columns - 3, :] = StorageKeys.MID_AISLE.value
        # middle vertical aisle at right of warehouse
        s[:, 2, :] = StorageKeys.MID_AISLE.value
        # traversable aisle tile in front of access tiles
        s[src_rows[0], 0:n_columns - 1, :] = StorageKeys.MID_AISLE.value
        s[:, 0, :] = StorageKeys.WALL.value
        s[:, n_columns - 1, :] = StorageKeys.WALL.value
        s[0, :, :] = StorageKeys.WALL.value
        s[n_rows - 1, :, :] = StorageKeys.WALL.value

        s[src_rows[0], 0, :] = StorageKeys.SOURCE.value
        counter = 0
        direction = 1
        scale = 1
        for i in range(0, n_sources - 1):
            if i == 2 * scale:
                scale += 1
            mid_row = src_rows[0]
            src_rows.append(mid_row + 2 * direction * scale)
            s[mid_row + 2 * direction * scale, 0, :] = StorageKeys.SOURCE.value
            s[mid_row
              + 2 * direction * scale, 1, :] = StorageKeys.MID_AISLE.value
            direction = direction * -1
            counter += 1

        s[sink_rows[0], n_columns - 1, :] = StorageKeys.SINK.value
        counter = 0
        direction = 1
        scale = 1
        for i in range(0, n_sinks - 1):
            if i == 2 * scale:
                scale += 1
            mid_row = sink_rows[0]
            sink_rows.append(mid_row + 2 * direction * scale)
            s[mid_row + 2
              * direction * scale, n_columns - 1, :] = StorageKeys.SINK.value
            s[mid_row + 2 * direction
              * scale, n_columns - 2, :] = StorageKeys.MID_AISLE.value
            direction = direction * -1
            counter += 1
        return s.astype(int)

    def get_average_travel_time(self):
        tes = self.trackers.travel_event_statistics
        return tes.average_travel_time() / 3600

    def get_average_service_time(self):
        return self.trackers.average_service_time / 60

    def add_travel_event(
            self, travel_event: Union['DeliveryFirstLeg', 'RetrievalFirstLeg',
                                      'RetrievalSecondLeg', None]):
        """adds a Travel event to self.travel_events"""
        if travel_event:
            t = travel_event
            order_type, leg, _ = t.travel_type.split('_')
            parameters = {"SKU": int(t.order.SKU),
                          "type": order_type,
                          "leg": leg,
                          "time": t.time,
                          "path": [list(([int(i) for i in node]))
                                   for node in t.route.get_indices()]}

            self.travel_events[t.order.order_number] = parameters

    def delivery_possible(self, agv_pos: Tuple[int, int] = None,
                          index: int = None) -> bool:
        sc, tes = self.location_manager, self.trackers.travel_event_statistics
        n_open_lanes = (sc.lane_manager.n_lanes
                        - len(sc.lane_manager.locked_lanes)
                        - len(sc.lane_manager.full_lanes))
        n_unbound_delivery = tes.get_unbound_travel_events()
        if agv_pos is not None and index is not None:
            n_forks = self.agv_manager.free_agv_positions[agv_pos][index].forks
            return n_open_lanes - (n_forks - 1) >\
                n_unbound_delivery
        else:
            return n_open_lanes - (self.agv_manager.maximum_forks - 1) >\
                n_unbound_delivery

    def retrieval_possible(self, sku: int):
        """returns true if the given SKU is serviceable"""
        sc, tes = self.location_manager, self.trackers.travel_event_statistics
        locations = sc.get_sku_locations(sku, tes)
        if locations:
            return True
        else:
            return False

    def remove_travel_event(self, order_number):
        self.travel_events.pop(order_number)

    def add_orders(self, orders):
        """called during reset() adds all """
        for o in orders:
            parameters = {"SKU": int(o.SKU),
                          "type": type(o).__name__,
                          "time": o.time}
            self.incomplete_orders[o.order_number] = parameters

    def remove_order(self, order_number):
        """remove order_number from self.incomplete_orders"""
        try:
            self.incomplete_orders.pop(order_number)
        except KeyError:
            print("A minor error has occured in remove_order...")

    def add_route(self, new_route):
        """adds route to self.routes"""
        self.routes.add(tuple(new_route))

    def remove_route(self, existing_route):
        """removes route from self.routes"""
        self.routes.discard(tuple(existing_route))

    def update_s_t_b_matrices(self, position, s_value, t_value, b_value):
        """
        Sets the the values of the Storage, Time and Batch matrices at the
        positions specified by this function's first parameter to the values
        specified by s_value, t_values and b_value respectively.


        :param position: The position within the warehouse.
        :param s_value: The SKU of the pallet or a valid StorageKey constant.
        :param t_value: The time of the last modification.
        :param b_value: The production batch of the pallet.
        :return: None.
        """
        p = position
        self.S[p[0], p[1], p[2]] = s_value
        self.T[p[0], p[1], p[2]] = t_value
        self.B[p[0], p[1], p[2]] = b_value

    @staticmethod
    def get_source_locations(storage_matrix: np.ndarray):
        """called once. gets location of tiles marked as source"""
        return [tuple(i[0:2]) for i in
                np.argwhere(storage_matrix[:, :, 0] == StorageKeys.SOURCE)]

    @staticmethod
    def get_sink_locations(storage_matrix: np.ndarray):
        """called once. gets location of tiles marked as sink"""
        return [tuple(i[0:2]) for i in
                np.argwhere(storage_matrix[:, :, 0] == StorageKeys.SINK)]

    @staticmethod
    def get_io_locations(storage_matrix: np.ndarray):
        """called once. gets location of tiles marked as source and sink"""
        source_positions = [tuple(i[0:2]) for i in
                            np.argwhere(storage_matrix[:, :, 0] ==
                                        StorageKeys.SOURCE)]

        sink_positions = [tuple(i[0:2]) for i in
                          np.argwhere(storage_matrix[:, :, 0] ==
                                      StorageKeys.SINK)]

        return source_positions + sink_positions

    def concatenate(self):
        """"reshapes multi-level/3D storage and time matrices, then concatenates
        them along with the vehicle matrix
        """
        n_rows = self.S.shape[0]
        s = np.copy(self.S).transpose((1, 0, 2)).reshape(n_rows, -1)
        t = np.copy(self.T).transpose((1, 0, 2)).reshape(n_rows, -1)
        return np.concatenate([s, self.V, t], axis=1)

    def get_n_storage_locations(self):
        """called """
        # return len(np.argwhere(self.S == StorageKeys.EMPTY))
        return (self.location_manager.n_open_locations
                - len(self.location_manager.locked_open_storage))

    def __str__(self):
        """converts all matrices into readable, printable strings, also adds
        some state values like open storage locations, occupied storage
        locations, etc.
        """
        output = "storage matrix: \n"
        n_levels = self.S.shape[2]
        for i in range(0, n_levels):
            output += "level {}\n".format(i+1)
            output += "   "
            for col in range(0, self.S.shape[1]):
                output += "{}".format(col).rjust(3)
            output += "\n"
            row = 0
            for s in self.S[:, :, i]:
                output += "{}".format(row).rjust(3)
                for elem in s:
                    if elem == -1:
                        elem = "w"
                    if elem == -2:
                        elem = "a"
                    if elem == 0:
                        elem = "-"
                    if elem == -3:
                        elem = "so"
                    if elem == -4:
                        elem = "si"
                    if elem == -5:
                        elem = "m"
                    output += "{}".format(elem).rjust(3)
                row += 1
                output += "\n"
            output += "\n"
        output += "vehicle matrix: \n"
        for v in self.V:
            for elem in v:
                if elem == -1:
                    elem = "-"
                if elem == 0:
                    elem = "f"
                if elem == 1:
                    elem = "b"
                output += "{}".format(elem).rjust(3)
            output += "\n"

        output += "arrival \n"
        for i in range(0, n_levels):
            output += "level {}\n".format(i + 1)
            for t in self.T[:, :, i]:
                for elem in t:
                    if elem == -1:
                        elem = "-"
                    else:
                        elem = round(elem, 1)
                    output += "{}".format(elem).rjust(3)
                output += "\n"
        output += "all open locations: \n"
        output += str(self.location_manager.open_storage_locations)
        output += "\noccupied storage locations: "
        output += str(self.location_manager.lane_manager.occupied_lanes)
        output += "\nlegal delivery locations: \n"
        output += str(self.location_manager.get_open_locations())
        output += "\nlocked open storage: "
        output += str(self.location_manager.locked_open_storage)
        output += "\nlocked occupied storage: "
        output += str(self.location_manager.locked_occupied_storage)
        output += "\nfree agvs: \n"
        output += str(self.agv_manager.get_agv_locations())
        return output

    @staticmethod
    def __fill_matrix_metadata(matrix: np.ndarray,
                               label_x='x',
                               label_y='y',
                               nfo_type='jobs'):
        """
        Converts a particular state matrix to a list and ads metadata
        pertaining
        to the contained values, matrix structure and index names as specified
        by the function inpt. The list representation together with the
        metadata fields is stored within a dictionary which is returned.

        to_dict() helper method.

        :param matrix: The state matrix to add metadata to.
        :param label_x: The name of the column index.
        :param label_y: The name of the row index.
        :param nfo_type: The matrix category; can be either "jobs", "machines"
            or "tracking".
        :return: The dictionary containing the list representation and metadata
            for the matrix parameter.
        """
        if np.inf in matrix:
            matrix[matrix == np.inf] = 9999
        return {
            'x_label': label_x,
            'y_label': label_y,
            'data': matrix.astype(str).tolist(),
            'min_value': int(matrix.min(initial=sys.maxsize)),
            'max_value': int(matrix.max(initial=-sys.maxsize)),
            'n_rows': matrix.shape[0],
            'n_cols': matrix.shape[1],
            'nfo_type': nfo_type
        }

    def to_dict(self):
        """
        Creates a serializable dictionary representation of the state matrices.

        :return: The dictionary representation of the state matrices.
        """
        dictionary = {}
        n_levels = self.S.shape[2]
        dictionary['storage_matrix'] = State.__fill_matrix_metadata(
            self.S)
        dictionary['vehicle_matrix'] = State.__fill_matrix_metadata(
            self.V)
        # t = np.copy(self.T)
        # t = t.astype(str)
        dictionary['arrival_time_matrix'] = State.__fill_matrix_metadata(self.T)
        legal_actions = [list(int(j) for j in c_unravel(i, self.S.shape))
                         for i in self.location_manager.legal_actions]
        # legal_actions = list(self.legal_actions)
        # legal_actions = [list(([int(i) for i in action]))
        #                  for action in self.legal_actions]
        routes = [list(list(([int(i) for i in node])) for node in route)
                  for route in self.routes]
        dictionary['paths'] = routes
        dictionary['legal_actions'] = legal_actions
        dictionary['decision_mode'] = self.current_order
        # for key, value in self.incomplete_orders.items():
        #     value['time'] = str(value['time'])
        dictionary['orders'] = self.incomplete_orders
        # for key, value in self.travel_events.items():
        #     value['time'] = str(value['time'])
        dictionary['travel_events'] = self.travel_events
        dictionary['n_lvls'] = n_levels
        return dictionary

    """
    def print(self):
        print("storage matrix")
        for s in self.S:
            for elem in s:
                print("{}".format(elem).rjust(3), end="")
            print(end="\n")
        print("vehicle matrix")
        for v in self.V:
            for elem in v:
                print("{}".format(elem).rjust(3), end="")
            print(end="\n")
        
        print("arrival")
        for t in self.T:
            for elem in t:
                print("{}".format(elem).rjust(3), end="")
            print(end="\n")
        """

    def set_legal_actions(self, legal_actions):
        self.location_manager.legal_actions = legal_actions

    def set_current_order(self, current_order):
        self.current_order = current_order

    def set_current_sku(self, current_sku):
        self.current_sku = current_sku

    def set_current_source_sink(self, current_source_sink):
        self.current_source_sink = current_source_sink

    def set_current_order_arrival_time(self, time):
        self.order_arrival_time = time

    def update_on_travel_event_creation(self, travel_type: TravelEventKeys):
        self.trackers.update_on_travel_event_creation(travel_type)

    def update_on_travel_event_completion(
            self, travel_type: TravelEventKeys, duration, distance: float,
            penalty: float):
        self.trackers.update_on_travel_event_completion(
            travel_type, duration, distance, penalty)

    def update_when_vehicle_free(self, end_position: Tuple[int, int]):
        """
        Updates the vehicle position running averages.

        :param end_position: The position of the freed vehicle.
        :return: None.
        """
        tes = self.trackers.travel_event_statistics
        avg = tes.avg_dx
        n = tes.avg_count + 1
        tes.avg_dx = (avg * (n - 1)) / n + end_position[0] / n
        avg = tes.avg_dy
        tes.avg_dy = (avg * (n - 1)) / n + end_position[1] / n
        tes.avg_count += 1

    def set_current_destination(self, destination):
            self.current_destination = destination

    def __deepcopy__(self, memo):
        return faster_deepcopy(self, memo)
