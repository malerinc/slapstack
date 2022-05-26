from math import inf, hypot
from typing import Tuple, Dict, List, Union

import numpy as np

from slapstack.helpers import VehicleKeys, StorageKeys
from slapstack.interface_templates import SimulationParameters


class AGV:
    def __init__(self, transport_id: int, pos: Tuple[int, int]):
        """
        AGV object constructor. Instances of this class represent the warehouse
        transports.

        :param transport_id: The unique transport id.
        :param pos: The current AGV position.
        """
        self.id = transport_id
        self.position = pos
        self.free = True
        self.booking_time = -1
        self.utilization = 0

    def log_booking(self, booking_time: float):
        """
        Marks the AGV as busy and notes down the time at which the booking
        occurred.

        :param booking_time: The time at which the AGV was selected for a job.
        :return: None.
        """
        self.booking_time = booking_time
        self.free = False

    def log_release(self, release_time: float, position: Tuple[int, int]):
        """
        Marks the AGV as free and computes the utilization time using the
        previously set booking time and updates its position.

        :param release_time: The time at which the AGV finished its job.
        :param position: The new AGV position.
        :return: None.
        """
        self.position = position
        self.free = True
        self.utilization += release_time - self.booking_time


class AgvManager:
    def __init__(
            self, p: SimulationParameters, storage_matrix: np.ndarray, rng):
        """
        AgvManager Constructor. This object deals with information related to
        AGVs through the warehouse. Most importantly it tracks the position
        of the free AGVs and the AGV utilization time.

        The free_agv_positions property indexes free ASVs (@see the AGV object)
        by their position in the warehouse. The agv_index property indexes the
        *same* AGV objects by their id.

        Additionally the class maintains several counters such as the current
        number of free or busy AGVs.

        :param storage_matrix: The vehicle matrix from which to extract the
            initial AGV positions.
        """
        self.free_agv_positions: Dict[Tuple[int, int], List[AGV]] = {}
        self.agv_index: Dict[int, AGV] = {}
        self.V = np.full((p.n_rows, p.n_columns),
                         VehicleKeys.N_AGV.value, dtype='int8')
        self.__initialize_agvs(storage_matrix, rng, p.n_agvs)
        self.n_free_agvs = len(self.free_agv_positions)
        self.n_busy_agvs = 0
        self.n_agvs = p.n_agvs
        self.n_visible_agvs = p.n_agvs

    def __initialize_agvs(self, S: np.ndarray, rng, n_agvs):
        """
        Initializes the simulation AGVs by assigning initial positions (aisle
        only!) and populating the agv_index and free_agv_positions fields.

        :return: None.
        """
        # agv_counter = 2  # initial position (column) of AGVs
        storage_locations = np.argwhere(S[:, :, 0] == StorageKeys.MID_AISLE)
        transport_id = 0
        for i in range(0, n_agvs):
            index = rng.integers(0, len(storage_locations))
            pos_t = tuple(storage_locations[index])
            if pos_t in self.free_agv_positions:
                continue
            new_agv = AGV(transport_id, pos_t)
            self.free_agv_positions[pos_t] = [new_agv]
            self.agv_index[transport_id] = new_agv
            transport_id += 1
            self.V[pos_t] = VehicleKeys.FREE.value

    def update_v_matrix(self, first_position: Tuple[int, int],
                        second_position: Union[Tuple[int, int], None],
                        release: bool = False):
        """
        Updates the vehicle matrix as AGVs move across tiles. The following
        cases are distinguished:
        1. When an AGV gets selected to service an order (delivery or retrieval)
        it will be marked as busy. The service booking is indicated by the None
        value of the second_position parameter.
        2. When an AGV simply moves without having finished its order
        (release == False), its position is updated without changing the
        markings status (AGV stays busy).
        3. Whenever an order is finished, the AGVs position is updated (at sink
        for delivery orders, in the corresponding lane for retrieval orders) and
        the vehicle is marked as free.

        :param first_position: The AGV position before the update.
        :param second_position: The AGV position after the update; if None, then
            the AGV has not moved.
        :param release: Whether the AGV became free or not.
        :return: None.
        """
        # when creating an order
        src_x, src_y = first_position[0], first_position[1]
        if not second_position:
            self.V[src_x, src_y] = VehicleKeys.BUSY.value
        else:
            tgt_x, tgt_y = second_position[0], second_position[1]
            if release:
                self.V[src_x, src_y] = VehicleKeys.N_AGV.value
                self.V[tgt_x, tgt_y] = VehicleKeys.FREE.value
            else:
                self.V[src_x, src_y] = VehicleKeys.N_AGV.value
                self.V[tgt_x, tgt_y] = VehicleKeys.BUSY.value

    def update_on_retrieval_second_leg(
            self, position: Tuple[int, int], system_time: float, agv_id: int):
        self.release_agv(position, system_time, agv_id)

    def agv_available(self) -> bool:
        """
        Checks if there are any available AGVs to use for transport.

        :return: True if the free_agv_positions dictionary is not empty and
            false otherwise.
        """
        return bool(self.free_agv_positions)

    def book_agv(self, position: Tuple[int, int], system_time: float):
        """
        Called whenever a fist leg transport event starts. Depending on the
        event trigger position (source for delivery first leg or chosen pallet
        in the case of retrieval first leg) the closest agv from the
        free_agv_positions is selected for booking. The euclidean distance is
        used for distance comparison.

        An AGV object located at the chosen position is removed from
        free_agv_positions and its booking time is marked by calling the
        log_booking method. Finally, if the chosen AGV was the last located at
        the given position, the position is removed from the index entirely.

        :param position: The position of the triggering event (either source or
            chosen sku).
        :param system_time: The current simulation time.
        :return: The AGV booked AGV.
        """
        selected_agv_pos = self.__get_close_agv(position)
        agv = self.free_agv_positions[selected_agv_pos].pop()
        agv.log_booking(system_time)
        self.n_busy_agvs += 1
        self.n_free_agvs -= 1
        if not self.free_agv_positions[selected_agv_pos]:
            del self.free_agv_positions[selected_agv_pos]
        return agv

    def release_agv(self, position, system_time: float, agv_id: int):
        """
        Called on occurrence of second leg transport events to release the
        associated AGV.

        The released AGV is selected from the agv_index by means of the passed
        id. The the release is loged within the AGV object which in particular
        results in the update of the corresponding utilization time. The AGV
        is then added to the free_agv_positions index and the class counters
        are updated.

        :param position: The AGV position after the finished second leg
            transport.
        :param system_time: The current simulation time.
        :param agv_id: The id of the release AGV; note that the AGV ids are set
            on first leg Transport event creation and then passed to the
            corresponding second leg transport.
        :return: None.
        """
        released_agv = self.agv_index[agv_id]
        released_agv.log_release(system_time, position)
        if position in self.free_agv_positions:
            self.free_agv_positions[position].append(released_agv)
        else:
            self.free_agv_positions[position] = [released_agv]
        self.n_busy_agvs -= 1
        self.n_free_agvs += 1

    def __get_close_agv(self, position: Tuple[int, int]) -> Tuple[int, int]:
        """
        Iterates over the free AGVs in the free_agv_positions and selects the
        one closest to the passed position with respect to euclidean distance.

        :param position: The position relative to which the closest AGV is to
            be selected.
        :return: The closest AGV position.
        """
        selected_agv = None
        min_distance = inf
        for agv in self.free_agv_positions:
            distance = hypot(position[0]-agv[0], position[1]-agv[1])
            if distance < min_distance:
                selected_agv = agv
                min_distance = distance
        # assert selected_agv in self.free_agv_positions
        return selected_agv

    def get_agv_locations(self) -> Dict[Tuple[int, int], List[AGV]]:
        """
        Returns the free_agv_positions index.

        :return: The dictionary mapping free agv positions to lists of AGV
            objects.
        """
        return self.free_agv_positions

    def get_average_utilization(self):
        """
        Iterates over the agv_index and extracts the average utilization time.

        :return: The average utilization time.
        """
        utl_sum = 0
        for agv_id, agv in self.agv_index.items():
            utl_sum += agv.utilization
        return utl_sum / len(self.agv_index)
