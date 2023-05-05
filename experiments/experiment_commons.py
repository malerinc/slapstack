import pickle
import time
from os.path import exists

import numpy as np
import pandas as pd
from tqdm import tqdm

from slapstack import SlapEnv

from slapstack.core_state import State, Trackers
from slapstack.core_state_location_manager import LocationManager
from slapstack.helpers import create_folders, TravelEventKeys
from slapstack.interface_templates import SlapLogger, SimulationParameters
from slapstack_controls.storage_policies import (
    StoragePolicy, ClosestOpenLocation, ClosestToNextRetrieval, ShortestLeg,
    BatchFIFO, ClassBasedPopularity, ShortestLeg)


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
        self.t_s = time.time()

    def set_logfile_name(self, logfile: str):
        self.logfile_name = logfile

    def log_state(self):
        s = self.slap_state
        first_step = len(self.log_data) == 0
        save_logs = len(self.log_data) % self.n_steps_between_saves == 0
        n_orders = len(s.trackers.finished_orders)
        if n_orders != self.prev_n_orders:
            self.prev_n_orders = n_orders
            self.log_data.append(ExperimentLogger.__get_row(s, self.t_s))
            if (not first_step and save_logs) or s.done:
                self.write_logs()

    def write_logs(self):
        n_orders = len(self.slap_state.trackers.finished_orders)
        cols = self.__get_header(self.slap_state)
        df = pd.DataFrame(data=self.log_data,
                          columns=cols)
        df.to_csv(f'{self.log_dir}/{self.logfile_name}_{n_orders}.csv')
        self.log_data = []

    @staticmethod
    def __get_header(state: State):
        zm = state.location_manager.zone_manager
        header = [
            # Travel Info
            'total_distance',
            'average_distance',
            'travel_time_retrieval_ave',
            'distance_retrieval_ave',
            'total_shift_distance',
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
        ]
        if len(zm.n_open_locations_per_zone) != 0:
            header += [f'fill_zone_{i}'
                       for i in range(len(zm.n_open_locations_per_zone.keys()))]
        return header

    @staticmethod
    def __get_row(s: State, t_s=0.0):
        zm = s.location_manager.zone_manager
        tes = s.trackers.travel_event_statistics
        t = s.trackers
        sc = s.location_manager
        am = s.agv_manager
        row = (
            # Travel Info:
            tes.total_distance_traveled,
            tes.average_travel_distance(),
            tes.get_average_travel_time_retrieval(),
            tes.get_average_travel_distance_retrieval(),
            tes.total_shift_distance,
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
            time.time() - t_s,
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
                           episode_decisions: int, end_state: State):
        zm = end_state.location_manager.zone_manager
        tes = end_state.trackers.travel_event_statistics
        t = end_state.trackers
        fill_level_per_zone = \
            1 - np.array(list(zm.n_open_locations_per_zone.values())) / \
            np.array(list(zm.n_total_locations_per_zone.values()))

        sc = end_state.location_manager
        es = end_state
        print(f"Episode with storage strategy "
              f"{strategy_name} ended after "
              f"{time.time() - episode_start_time} seconds:")
        print(f"\tBroad Trackers:")
        print(f"\t\tNumber of decisions: {episode_decisions}")
        print(f"\t\tNumber of pallet shifts: {t.number_of_pallet_shifts}")
        print(f"\t\tShift distance penalty: {tes.total_shift_distance}")
        print(f"\t\tFill Level: {t.get_fill_level()}")
        print(f"\t\tFill level per zone: {fill_level_per_zone}")
        print(f'\t\tAverage Lane Entropy: '
              f'{ExperimentLogger.__get_lane_entropy(sc)}')
        print(f'\tKPI:')
        print(f'\t\tThroughput: {ExperimentLogger.__get_throughput(end_state)}')
        print(f"\t\tMakespan: {end_state.time}")
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
        print(f"\t\tMean travel time retrieval: {mean_time_ret}")
        print(f'\t\tAverage AGV utilization: '
              f'{end_state.agv_manager.get_average_utilization() / es.time}')
        print(f"\tOrder Info:")
        print(f"\t\tPending Retrieval Orders: {t.n_queued_retrieval_orders}")
        print(f"\t\tPending Delivery Orders: {t.n_queued_delivery_orders}")
        print(f"\t\tNumber of orders completed: {len(t.finished_orders)}")
        print(f"\t\tNumber of Visible AGVs: {es.agv_manager.n_visible_agvs}")


class LoopControl:
    def __init__(self, env: SlapEnv, pbar_on=True):
        self.done = False
        self.n_decisions = 0
        if pbar_on:
            total_orders = env.core_env.orders.n_orders
            finished_orders = len(env.core_env.state.trackers.finished_orders)
            remaining_orders = total_orders - finished_orders
            self.pbar = tqdm(total=int(remaining_orders / 2))
        else:
            self.pbar = None
        self.state: State = env.core_env.state
        self.trackers: Trackers = env.core_env.state.trackers

    def stop_prematurely(self):
        t = self.state.trackers
        if (t.average_service_time > 1800
                or t.n_queued_delivery_orders > 627
                or t.n_queued_retrieval_orders > 693):
            return True
        return False


def _init_run_loop(simulation_parameters, storage_strategy, log_dir):
    if hasattr(storage_strategy, 'n_zones'):
        environment: SlapEnv = get_episode_env(
            sim_parameters=simulation_parameters,
            log_frequency=1000,
            nr_zones=storage_strategy.n_zones, log_dir=log_dir)
    else:
        environment: SlapEnv = get_episode_env(
            sim_parameters=simulation_parameters,
            log_frequency=1000, nr_zones=3, log_dir=log_dir)
    loop_controls = LoopControl(environment)
    # state.state_cache.perform_sanity_check()
    environment.core_env.logger.set_logfile_name(
        f'{storage_strategy.name}_n{simulation_parameters.n_agvs}')
    return environment, loop_controls


def run_episode(simulation_parameters: SimulationParameters,
                storage_strategy: StoragePolicy, print_freq=0,
                warm_start=False, log_dir='./result_data/',
                stop_condition=False, pickle_at_decisions=np.infty):
    pickle_path = (f'end_env_{storage_strategy.name}_'
                   f'{pickle_at_decisions}.pickle')
    env, loop_controls = _init_run_loop(
        simulation_parameters, storage_strategy, log_dir)
    parametrization_failure = False
    start = time.time()
    if exists(pickle_path):
        env = pickle.load(open(pickle_path, 'rb'))
        loop_controls = LoopControl(env, pbar_on=True)
    while not loop_controls.done:
        if warm_start and len(loop_controls.trackers.finished_orders) < 1000:
            action = ClosestOpenLocation().get_action(loop_controls.state)
        elif (isinstance(storage_strategy, ClosestToNextRetrieval)
              or isinstance(storage_strategy, ShortestLeg)):
            action = storage_strategy.get_action(
                loop_controls.state, env.core_env)
        else:
            action = storage_strategy.get_action(loop_controls.state)
        output, reward, loop_controls.done, info = env.step(action)
        if print_freq and loop_controls.n_decisions % print_freq == 0:
            if loop_controls.n_decisions > pickle_at_decisions:
                pickle.dump(env, open(pickle_path, 'wb'))
            ExperimentLogger.print_episode_info(
                storage_strategy.name, start, loop_controls.n_decisions,
                loop_controls.state)
            # state.state_cache.perform_sanity_check()
        loop_controls.n_decisions += 1
        if loop_controls.pbar is not None:
            loop_controls.pbar.update(1)
        if not loop_controls.done and stop_condition:
            # will set the done control to true is stop criteria is met
            loop_controls.done = loop_controls.stop_prematurely()
            if loop_controls.done:
                parametrization_failure = True
                env.core_env.logger.write_logs()
    ExperimentLogger.print_episode_info(
        storage_strategy.name, start, loop_controls.n_decisions,
        loop_controls.state)
    return parametrization_failure


def get_episode_env(sim_parameters: SimulationParameters,
                    log_frequency: int, nr_zones: int,
                    log_dir='./result_data/'):
    seeds = [56513]
    if isinstance(sim_parameters.initial_pallets_storage_strategy,
                  ClassBasedPopularity):
        sim_parameters.initial_pallets_storage_strategy = ClassBasedPopularity(
            retrieval_orders_only=False,
            future_counts=True,
            init=True,
            n_zones=nr_zones
        )
    return SlapEnv(
        sim_parameters, seeds,
        logger=ExperimentLogger(
            filepath=log_dir,
            n_steps_between_saves=log_frequency,
            nr_zones=nr_zones),
        action_converters=[BatchFIFO()])
