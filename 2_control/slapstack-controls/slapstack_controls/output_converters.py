from math import sqrt

import numpy as np

from slapstack.core_state import State
from slapstack.interface_templates import OutputConverter


class LegacyOutputConverter:
    def __init__(self, reward_type="average_travel_length", state_modifier=None):
        self.previous_average_travel_length = 0
        self.previous_average_service_time = 0
        self.reward_type = reward_type
        self.state_modifier = state_modifier
        self.n_steps = 0
        self.reward_interval = 5

    def modify_state(self, state: State) -> np.ndarray:
        if self.state_modifier == "storage_matrix_only":
            return state.S.flatten()
        if self.state_modifier == "storage_locations_only":
            return state.S[state.S >= 0]
        if self.state_modifier == "lane_free_space":
            return self.calculate_free_spaces_per_lane(state)
        if self.state_modifier == "free_entropy_dominant":
            return self.get_free_space_entropy_and_dominant(state)
        else:
            return state.concatenate().flatten()

    def calculate_free_spaces_per_lane(self, state):
        free_spaces = []
        for lane, lane_info in \
                state.state_cache.lanes.items():
            free_spaces.append(lane_info['n_free_spaces'])
        return np.array(free_spaces)

    def calculate_reward(self, state: State, action: int, legal_actions: list) \
            -> float:
        reward = 0.0
        if self.reward_type == 'average_travel_length':
            reward = self.calculate_average_travel_length_reward(
                action, legal_actions, state)
        elif self.reward_type == 'average_service_time':
            reward = self.calculate_average_service_time_reward(
                action, legal_actions, state)
        elif self.reward_type == 'distance_traveled_shift_penalty':
            reward = self.calculate_distance_traveled(
                action, legal_actions, state)
        reward = reward + 5.1
        return reward

    def calculate_distance_traveled(self, action, legal_actions, state):
        """this reward takes the distance traveled by Travel events and takes a
        percentage of it as a negative reward. It also takes pallet shifts as
        negative reward since they take a while."""
        warehouse_shape = state.S[:, :, 0].shape
        scale = 1/sqrt(warehouse_shape[0]*warehouse_shape[1])
        reward = 0
        if action not in set(legal_actions):
            reward = -500
        else:
            if self.n_steps % self.reward_interval == 0 and self.n_steps:
                reward = -1 * state.trackers.last_travel_distance * scale
                reward -= state.trackers.number_of_pallet_shifts
                if reward < -5:
                    reward = -5
        self.n_steps += 1
        return reward

    def calculate_average_travel_length_reward(self, action, legal_actions,
                                               state):
        average_travel_length = state.trackers.average_travel_length
        difference = self.previous_average_travel_length - average_travel_length
        # if action not in set(legal_actions):
        #     reward = -500
        # else:
        if self.n_steps % self.reward_interval == 0 and self.n_steps:
            # if measurement got worse
            if average_travel_length > self.previous_average_travel_length:
                if difference < -1:
                    reward = -1
                else:
                    reward = difference
            else:  # if measurement got better
                if difference > 1:
                    reward = 1
                else:
                    reward = difference
        else:
            reward = 0.0
        self.n_steps += 1
        self.previous_average_travel_length = average_travel_length
        if reward < -100:
            print()
        return reward

    def calculate_average_service_time_reward(self, action, legal_actions, state):
        average_service_time = state.trackers.average_service_time
        difference = self.previous_average_service_time - average_service_time
        # if action not in set(legal_actions):
        #     reward = -500
        # else:
        if self.n_steps % self.reward_interval == 0 and self.n_steps:
            if average_service_time > self.previous_average_service_time:  # if measurement got worse
                if difference < -1:
                    reward = -1
                else:
                    reward = difference
            else:  # if measurement got better
                if difference > 1:
                    reward = 1
                else:
                    reward = difference
        else:
            reward = 0.0
        self.n_steps += 1
        self.previous_average_service_time = average_service_time
        return reward

    def get_free_space_entropy_and_dominant(self, state):
        entropy = []
        free_spaces = []
        dominant_skus = []
        for lane, lane_info in \
                state.state_cache.lanes.items():
            entropy.append(lane_info['entropy'])
            free_spaces.append(lane_info['n_free_spaces'])
            dominant_skus.append(lane_info['dominant_sku'])
        return np.array(entropy+free_spaces+dominant_skus)


class FeatureConverter(OutputConverter):
    def __init__(self, feature_list):
        self.flattened_entropies = None
        self.fill_level_per_lane = None
        self.fill_level_per_zone = None
        self.sku_counts = None
        self.feature_list = feature_list

    def init_fill_level_per_lane(self, state: State):
        open_locations = np.array(list(state.location_manager.n_open_locations_per_lane.values()))
        total_locations = np.array(list(state.location_manager.n_total_locations_per_lane.values()))
        self.fill_level_per_lane = 1 - open_locations / total_locations

    def f_get_lanewise_entropy_avg(self, state: State):
        if self.flattened_entropies is None:
            self.flattened_entropies = np.array(list(state.location_manager.lane_wise_entropies.values()))
        return np.average(self.flattened_entropies)

    def f_get_lanewise_entropy_std(self, state: State):
        if self.flattened_entropies is None:
            self.flattened_entropies = np.array(list(state.location_manager.lane_wise_entropies.values()))
        return np.std(self.flattened_entropies)

    def f_get_global_entropy(self, state: State):
        sku_counts = np.array(list(state.location_manager.sku_counts.values()))
        p_x = sku_counts[sku_counts != 0] / np.sum(sku_counts)
        return - np.sum(p_x * np.log2(p_x))

    def f_get_lane_fill_level_avg(self, state: State):
        if self.fill_level_per_lane is None:
            self.init_fill_level_per_lane(state)
        return np.average(self.fill_level_per_lane)

    def f_get_lane_fill_level_std(self, state: State):
        if self.fill_level_per_lane is None:
            self.init_fill_level_per_lane(state)
        return np.std(self.fill_level_per_lane)

    def f_get_global_fill_level(self, state: State):
        return 1 - state.location_manager.n_open_locations / state.location_manager.n_total_locations

    def f_get_lane_occupancy(self, state: State):
        if self.fill_level_per_lane is None:
            self.init_fill_level_per_lane(state)
        return np.average(self.fill_level_per_lane < 1)

    def f_get_n_sku_items_avg(self, state: State):
        if self.sku_counts is None:
            if not state.location_manager.sku_counts:
                self.sku_counts = np.array([1])
            else:
                self.sku_counts = np.array(list(state.location_manager.sku_counts.values()))
        # Normalized by max number of occupied locations by any SKU
        return np.average(self.sku_counts) / np.max(self.sku_counts)

    def f_get_n_sku_items_std(self, state: State):
        if self.sku_counts is None:
            if not state.location_manager.sku_counts:
                self.sku_counts = np.array([1])
            else:
                self.sku_counts = np.array(list(state.location_manager.sku_counts.values()))
        # Normalized by max number of occupied locations by any SKU
        return np.std(self.sku_counts) / np.max(self.sku_counts)

    def f_get_n_sku(self, state: State):
        return state.n_skus

    def f_get_total_pallet_shifts(self, state: State):
        # TODO: Normalize by total time steps.
        return state.trackers.number_of_pallet_shifts

    @staticmethod
    def f_get_queue_len_retrieval_orders(state: State):
        # TODO: Normalize by either n_agvs or n_sinks
        return len(state.trackers.queued_retrieval_orders)

    @staticmethod
    def f_get_queue_len_delivery_orders(self, state: State):
        # TODO: Normalize by either n_agvs or n_sources
        return len(state.trackers.queued_delivery_orders)

    def f_get_n_agvs(self, state: State):
        return state.location_manager.n_agvs

    def f_get_free_agv_ratio(self, state: State):
        # TODO: Should we use an if or add a stabilizing term in denominator?
        return len(state.location_manager.free_agv_positions) / (1e-6 + state.location_manager.n_agvs)

    def f_get_legal_actions_avg(self, state: State):
        # Done: Normalize with total number of storage locations.
        return state.location_manager.n_legal_actions_total / (
                (1e-6 + state.location_manager.n_actions_taken) * state.location_manager.n_total_locations
        )

    def init_fill_level_per_zone(self, state: State):
        zm: 'ZoneManager' = state.location_manager.zone_manager
        open_locations = np.array(list(zm.n_open_locations_per_zone.values()))
        total_locations = np.array(list(zm.n_total_locations_per_zone.values()))
        self.fill_level_per_zone = 1 - open_locations / total_locations

    def f_get_zone_fill_level_avg(self, state: State):
        if self.fill_level_per_zone is None:
            self.init_fill_level_per_zone(state)
        return np.average(self.fill_level_per_zone)

    def f_get_zone_fill_level_std(self, state: State):
        if self.fill_level_per_zone is None:
            self.init_fill_level_per_zone(state)
        return np.std(self.fill_level_per_zone)

    def f_get_legal_actions_std(self, state: State):
        e_x2 = state.location_manager.n_legal_actions_squared_total / (1e-6 + state.location_manager.n_actions_taken)
        e_x_2 = state.location_manager.n_legal_actions_total / (1e-6 + state.location_manager.n_actions_taken)
        return (e_x2 - (e_x_2 ** 2)) / state.location_manager.n_total_locations

    def f_get_legal_actions_current(self, state: State):
        return state.location_manager.n_legal_actions_current / (
                (1e-6 + state.location_manager.n_actions_taken) * state.location_manager.n_total_locations
        )

    def modify_state(self, state: State) -> np.ndarray:
        self.__init__(self.feature_list)

        features = []
        for feature_name in self.feature_list:
            features.append(getattr(self, f'f_get_{feature_name}')(state))

        return np.array(features)

        # Average lane entropy
        # Std dev lane entropy
        # Global entropy
        # Avg lane fill level
        # Std dev lane fill level
        # Global fill level
        # Lane occupancy percentage
        # Avg/Std dev number of items for each SKU / total.
        # Number of SKUs.
        # Global queue length of pending retrieval and delivery orders.
        # Number of each type of AGV.
        # AGV occupancy percentage for each type and over all types.
        # Avg/Stddev number of legal actions
        # Current number of legal actions

        # Total and/or rolling window pallet shifts

        # Avg distance traveled (global or per SKU or/and per AGV)
        # Avg service time (global or per SKU or/and per AGV)
        # Std dev distance traveled

        # Avg number of batches per SKU in the warehouse: FIFO inter-batches, BatchFIFO intra-batches
        # Entrance and exit occupancies: Needs to be implemented in simulation
        # Avg/Stddev age for each SKU
        # Avg/Stddev round trip time for each SKU
        # Loaded and unloaded travel distances (global or/and per AGV)
        # Entropy over agreement of strategies on current action. (possibly)
        # Editing distance of lanes as compared to a baseline.
        # Std dev of storage matrix.
        pass

    def calculate_reward(self, state: State, action: int, legal_actions: list) -> float:
        return 0

