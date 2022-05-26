from typing import Tuple, TYPE_CHECKING, Union, Dict, Set
from slapstack.helpers import StorageKeys
from slapstack.use_case import partition_use_case

import numpy as np


if TYPE_CHECKING:
    from slapstack.core_state import State


class SlapLogger:
    def __init__(self, dirpath: str):
        self.log_dir = dirpath

    def log_state(self, state: 'State'):
        raise NotImplementedError


class StorageStrategy:
    def __init__(self, strategy_type):
        assert strategy_type in ['retrieval', 'delivery']
        self.type = strategy_type

    def get_action(self, state: 'State') -> Tuple[int, int, int]:
        pass

    def update(self, action: Tuple[int, int, int]):
        pass


class OutputConverter:
    def modify_state(self, state: 'State') -> np.ndarray:
        pass

    def calculate_reward(self, state: 'State', action: int,
                         legal_actions: list):
        pass


class SimulationParameters:
    def __init__(self,
                 n_agvs: int,
                 pure_lanes: bool,
                 generate_orders: bool,
                 initial_pallets_storage_strategy: StorageStrategy,
                 resetting,
                 desired_fill_level: float = None,
                 n_skus: int = None,
                 all_skus: Set[int] = None,
                 n_orders: Union[int, None] = None,
                 order_list=None,
                 initial_pallets_sku_counts: Dict[int, int] = None,
                 verbose: bool = False,
                 agv_speed: float = 2.0,
                 unit_distance: float = 1.4,
                 pallet_shift_penalty_factor: int = 10,
                 n_rows: int = None,
                 n_columns: int = None,
                 n_levels: int = None,
                 n_sources: int = None,
                 n_sinks: int = None,
                 layout: Union[None, np.ndarray] = None,
                 start_period: int = None,
                 n_sku_in: Union[None, Dict[int, Dict[int, int]]] = None,
                 n_sku_out: Union[None, Dict[int, Dict[int, int]]] = None,
                 compute_feature_trackers: bool = True,
                 use_case_root_dir: str = None,
                 use_case_n_partitions: int = None,
                 use_case_partition_to_use: int = None,
                 ):

        # The inpt that are not required when usecase is provided.
        optionals = [
            n_rows, n_columns, n_levels, n_skus, all_skus, n_orders, order_list,
            initial_pallets_sku_counts, n_sources, n_sinks, layout, n_sku_in,
            n_sku_out, start_period, desired_fill_level
        ]

        if use_case_root_dir is not None:
            assert use_case_partition_to_use is not None
            assert use_case_n_partitions is not None

            use_case_partitions = partition_use_case(use_case_root_dir,
                                                     use_case_n_partitions)
            use_case = use_case_partitions[use_case_partition_to_use]
            self.n_rows = use_case.layout.shape[0]
            self.n_columns = use_case.layout.shape[1]
            self.n_levels = use_case.n_levels
            self.n_skus = len(use_case.distinct_skus)
            self.all_skus = use_case.distinct_skus
            self.n_orders = len(use_case.order_list)
            self.order_list = use_case.order_list
            self.initial_pallets_sku_counts = use_case.initial_skus[1]
            self.n_sources = len(
                np.argwhere(use_case.layout == StorageKeys.SOURCE))
            self.n_sinks = len(np.argwhere(use_case.layout == StorageKeys.SINK))
            self.layout = use_case.layout
            self.n_skus_in = use_case.sku_in_counts
            self.n_skus_out = use_case.sku_out_counts
            self.sku_period = use_case.current_week
            self.desired_fill_level = None
            self.shape = use_case.layout.shape + (use_case.n_levels,)

            if any(optionals):
                print("WARNING: some of the following options were passed to "
                      "SimulationParameters, but have been overridden due to "
                      "the passed usecase:")
                print(
                    "\tn_rows\n"
                    "\tn_columns\n"
                    "\tn_levels\n"
                    "\tn_skus\n"
                    "\tall_skus\n"
                    "\tn_orders\n"
                    "\torder_list\n"
                    "\tinitial_pallets_sku_counts\n"
                    "\tn_sources\n"
                    "\tn_sinks\n"
                    "\tlayout\n"
                    "\tn_sku_in\n"
                    "\tn_sku_out\n"
                    "\tstart_period\n"
                    "\tdesired_fill_level\n"
                )
        # elif all(optionals):
        else:
            self.n_rows = n_rows
            self.n_columns = n_columns
            self.n_levels = n_levels
            self.n_skus = n_skus
            self.all_skus = all_skus
            self.n_orders = n_orders
            self.order_list = order_list
            self.initial_pallets_sku_counts = initial_pallets_sku_counts
            self.n_sources = n_sources
            self.n_sinks = n_sinks
            self.layout = layout
            self.n_sku_in = n_sku_in
            self.n_sku_out = n_sku_out
            self.sku_period = start_period
            self.desired_fill_level = desired_fill_level
            self.shape = (n_rows, n_columns, n_levels)

        self.n_agvs = n_agvs
        self.pure_lanes = pure_lanes
        self.generate_orders = generate_orders
        self.initial_pallets_storage_strategy = initial_pallets_storage_strategy
        self.resetting = resetting
        self.verbose = verbose
        self.agv_speed = agv_speed
        self.unit_distance = unit_distance
        self.shift_penalty = pallet_shift_penalty_factor
        self.compute_feature_trackers = compute_feature_trackers
