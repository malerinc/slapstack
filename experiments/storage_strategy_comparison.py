from typing import List

import pandas as pd

import numpy as np
from tqdm import tqdm
import time

from slapstack.core_state import State
from slapstack.core_state_location_manager import LocationManager
from slapstack.interface import SlapEnv
from slapstack.helpers import create_folders, TravelEventKeys
# from slapstack.helpers import parallelize_heterogeneously
from slapstack.interface_templates import SimulationParameters, SlapLogger
from slapstack_controls.storage_policies import (ClassBasedPopularity,
                                                 ClassBasedCycleTime,
                                                 ClosestOpenLocation,
                                                 BatchLIFO, StoragePolicy)


class ExperimentLogger(SlapLogger):
    def __init__(self, filepath: str, logfile_name: str = 'experiment_data',
                 n_steps_between_saves=10000, nr_zones=3):
        super().__init__(filepath)
        self.n_steps_between_saves = n_steps_between_saves
        self.log_dir = filepath
        create_folders(f'{self.log_dir}/dummy')
        self.logfile_name = logfile_name
        self.log_data = []
        self.prev_n_orders = 0
        self.n_zones = nr_zones

    def set_logfile_name(self, logfile: str):
        self.logfile_name = logfile

    def log_state(self, s: State):
        first_step = len(self.log_data) == 0
        save_logs = len(self.log_data) % self.n_steps_between_saves == 0
        n_orders = len(s.trackers.finished_orders)
        if n_orders != self.prev_n_orders:
            self.prev_n_orders = n_orders
            self.log_data.append(ExperimentLogger.__get_row(s))
            if (not first_step and save_logs) or s.done:
                cols = self.__get_header()
                df = pd.DataFrame(data=self.log_data,
                                  columns=cols)
                df.to_csv(f'{self.log_dir}/{self.logfile_name}_{n_orders}.csv')
                self.log_data = []

    def __get_header(self):
        return [
            # Travel Info
            'total_distance',
            'average_distance',
            'travel_time_retrieval_ave',
            'distance_retrieval_ave',
            'utilization_time',
            # Order Info
            'n_queued_retrieval_orders',
            'n_queued_delivery_orders',
            'n_finished_orders',
            # KPIs
            'kpi__throughput',
            'kpi__makespan',
            'kpi__average_service_time',
            'kpi__cycle_time',
            # Broad Trackers
            'runtime',
            'n_free_agvs',
            'n_pallet_shifts',
            'n_steps',
            'n_decision_steps',
            'fill_level',
            'entropy'
        ] + [f'fill_zone_{z}' for z in range(self.n_zones)]

    @staticmethod
    def __get_row(s: State):
        zm = s.location_manager.zone_manager
        tes = s.trackers.travel_event_statistics
        t = s.trackers
        sc = s.location_manager
        am = s.agv_manager
        n_orders = len(t.finished_orders)
        row = (
            # Travel Info:
            tes.total_distance_traveled,
            tes.average_travel_distance(),
            tes.get_average_travel_time_retrieval(),
            tes.get_average_travel_distance_retrieval(),
            am.get_average_utilization() / s.time if s.time != 0 else 0,
            # Order Info:
            t.n_queued_retrieval_orders,
            t.n_queued_delivery_orders,
            len(t.finished_orders),
            # KPIs
            ExperimentLogger.__get_throughput(s),  # throughput
            s.time,  # makespan
            t.average_service_time,
            ExperimentLogger.__get_cycle_time(sc),
            # Broad Trackers
            time.time(),
            am.n_free_agvs,
            t.number_of_pallet_shifts,
            s.n_steps + s.n_silent_steps,
            s.n_steps,
            t.get_fill_level(),
            ExperimentLogger.__get_lane_entropy(sc)
        )
        fill_level_per_zone = tuple(
                1 - np.array(list(zm.n_open_locations_per_zone.values())) /
                np.array(list(zm.n_total_locations_per_zone.values()))
        )
        return row + fill_level_per_zone

    @staticmethod
    def __get_cycle_time(sc: LocationManager):
        sku_cycle_times = sc.sku_cycle_time
        sum_cycle_times = 0
        if len(sku_cycle_times) != 0:
            for sku, cycle_time in sku_cycle_times.items():
                sum_cycle_times += cycle_time
            return sum_cycle_times / len(sku_cycle_times)
        else:
            return 0

    @staticmethod
    def __get_lane_entropy(sc: LocationManager):
        lane_entropies = sc.lane_wise_entropies
        average_entropy = 0
        for lane, entropy in lane_entropies.items():
            average_entropy += entropy
        return average_entropy / len(lane_entropies)

    @staticmethod
    def __get_throughput(s: State):
        t = s.trackers
        return len(t.finished_orders) / s.time if s.time != 0 else 0

    @staticmethod
    def print_episode_info(strategy_name: str, episode_start_time: float,
                           episode_decisions: int, end_state: State,
                           ep_nr: int):
        zm = end_state.location_manager.zone_manager
        tes = end_state.trackers.travel_event_statistics
        t = end_state.trackers
        fill_level_per_zone = \
            1 - np.array(list(zm.n_open_locations_per_zone.values())) / \
            np.array(list(zm.n_total_locations_per_zone.values()))

        sc = end_state.location_manager
        es = end_state
        print(f"Episode {ep_nr + 1} with storage strategy "
              f"{strategy_name} ended after "
              f"{time.time() - episode_start_time} seconds:")
        print(f"\tBroad Trackers:")
        print(f"\t\tNumber of decisions: {episode_decisions}")
        print(f"\t\tNumber of pallet shifts: {t.number_of_pallet_shifts}")
        print(f"\t\tFill Level: {t.get_fill_level()}")
        print(f"\t\tFill level per zone: {fill_level_per_zone}")
        print(f'\t\tAverage Lane Entropy: '
              f'{ExperimentLogger.__get_lane_entropy(sc)}')
        print(f'\tKPI:')
        print(f'\t\tThroughput: {ExperimentLogger.__get_throughput(es)}')
        print(f"\t\tMakespan: {es.time}")
        print(f"\t\tMean service time: {t.average_service_time}")
        print(f'\t\tCycle Time: {ExperimentLogger.__get_cycle_time(sc)}')
        print("\tTravel Info:")
        print(f"\t\tTotal travel distance: {tes.total_distance_traveled}")
        print(f"\t\tMean travel distance: {tes.average_travel_distance()}")
        print(f"\t\tMean travel time: {tes.average_travel_time()}")
        td_rl1 = tes.average_travel_distance(TravelEventKeys.RETRIEVAL_1STLEG)
        td_rl2 = tes.average_travel_distance(TravelEventKeys.RETRIEVAL_2ND_LEG)
        mean_dist_ret = (td_rl1 + td_rl2) / 2
        print(f"\t\tMean travel distance retrieval: {mean_dist_ret}")
        tt_rl1 = tes.average_travel_time(TravelEventKeys.RETRIEVAL_2ND_LEG)
        tt_rl2 = tes.average_travel_time(TravelEventKeys.RETRIEVAL_1STLEG)
        mean_time_ret = (tt_rl1 + tt_rl2) / 2
        print(f"\t\tMean travel time retrieval: "
              f"{mean_time_ret}")
        print(f'\t\tAverage AGV utilization: '
              f'{es.agv_manager.get_average_utilization() / es.time}')
        print(f"\tOrder Info:")
        print(f"\t\tPending Retrieval Orders: {t.n_queued_retrieval_orders}")
        print(f"\t\tPending Delivery Orders: {t.n_queued_delivery_orders}")
        print(f"\t\tNumber of orders completed: {len(t.finished_orders)}")
        print(f"\t\tNumber of Visible AGVs: {es.agv_manager.n_visible_agvs}")


def get_episode_env(n_partitions: int, use_case_partition_nr: int,
                    log_frequency: int, nr_zones: int):
    params = SimulationParameters(
        use_case_root_dir="WEPAStacks",
        use_case_n_partitions=n_partitions,
        use_case_partition_to_use=use_case_partition_nr,
        n_agvs=40,
        generate_orders=False,
        verbose=False,
        resetting=False,
        initial_pallets_storage_strategy=ClassBasedPopularity(
            retrieval_orders_only=False,
            future_counts=True,
            init=True,
            n_zones=nr_zones
        ),
        pure_lanes=True,
        # https://logisticsinside.eu/speed-of-warehouse-trucks/
        agv_speed=2,
        unit_distance=1.4,
        pallet_shift_penalty_factor=20,  # in seconds
        compute_feature_trackers=True
    )
    seeds = [56513]
    return SlapEnv(
        params, seeds,
        logger=ExperimentLogger(
            './result_data/', n_steps_between_saves=log_frequency,
            nr_zones=nr_zones),
        action_converters=[BatchLIFO()])


def run_episode(storage_strategy: StoragePolicy,
                ep_nr: int, n_progress_bar_orders=0, print_freq=0):
    # env_cp, done, n_decisions = deepcopy(environment), False, 0
    if n_progress_bar_orders:
        pbar = tqdm(total=n_progress_bar_orders)
    if hasattr(storage_strategy, 'n_zones'):
        environment: SlapEnv = get_episode_env(
            1, ep_nr, log_frequency=1000,
            nr_zones=storage_strategy.n_zones)
    else:
        environment: SlapEnv = get_episode_env(
            1, ep_nr, log_frequency=1000, nr_zones=3)
    done, n_decisions = False, 0
    state = environment.core_env.state
    # state.location_manager.perform_sanity_check()
    environment.core_env.logger.set_logfile_name(
        f'ep{ep_nr}_{storage_strategy.name}')
    start = time.time()
    while not done:
        step_ts = time.time()
        # if state.decision_mode == "delivery":
        #     assert len(state.location_manager.legal_actions) <= len(
        #         state.location_manager.open_storage_locations)
        if len(state.trackers.finished_orders) < 1000:
            # warm start ;)
            action = ClosestOpenLocation().get_action(state)
        else:
            action = storage_strategy.get_action(state)
        as_time = time.time()
        state: State
        output, reward, done, info = environment.step(action)
        step_time = time.time()
        if print_freq and n_decisions % print_freq == 0:
            ExperimentLogger.print_episode_info(
                storage_strategy.name, start, n_decisions,
                environment.core_env.state, ep_nr)
            # state.location_manager.perform_sanity_check()
        n_decisions += 1
        if n_progress_bar_orders:
            # noinspection PyUnboundLocalVariable
            pbar.update(1)
        # if n_decisions > 10000:
        #     break
    ExperimentLogger.print_episode_info(
        storage_strategy.name, start, n_decisions,
        environment.core_env.state, ep_nr)


def get_storage_strategies(nr_zones: List[int]):
    storage_strategies = []
    for n_zone in nr_zones:
        storage_strategies += [
            ClassBasedCycleTime(
                n_orders=10000, recalculation_steps=1000, n_zones=n_zone),
            ClassBasedPopularity(
                retrieval_orders_only=False, n_zones=n_zone,
                future_counts=True,
                name=f'allOrdersPopularity_future_z{n_zone}'),
            ClassBasedPopularity(
                retrieval_orders_only=True, n_zones=n_zone,
                future_counts=True,
                name=f'retrievalPopularity_future_z{n_zone}'),
            ClassBasedPopularity(
                retrieval_orders_only=False, n_zones=n_zone,
                future_counts=False, n_orders=10000, recalculation_steps=1000,
                name=f'allOrdersPopularity_past_z{n_zone}'),
            ClassBasedPopularity(
                retrieval_orders_only=True, n_zones=n_zone,
                future_counts=False, n_orders=10000, recalculation_steps=1000,
                name=f'retrievalPopularity_past_z{n_zone}')
        ]
    storage_strategies += [
        ClosestOpenLocation(very_greedy=True),
        ClosestOpenLocation(very_greedy=False),
    ]
    return storage_strategies


# TODO. think about and (re)program visualization
storage_policies = get_storage_strategies([2])
partitions = 1

if __name__ == '__main__':
    for j in range(0, partitions):
        n_strategies = len(storage_policies)
        for i in range(0, n_strategies):
            run_episode(storage_policies[i], j, 205000, 1000)

        # parallelize_heterogeneously(
        #     [run_episode] * n_strategies,
        #     list(zip(storage_policies,
        #              [j] * n_strategies)))
