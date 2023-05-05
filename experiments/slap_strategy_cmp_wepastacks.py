from typing import List

from experiment_commons import run_episode
from slapstack.interface_templates import SimulationParameters
from slapstack_controls.storage_policies import (ClassBasedPopularity,
                                                 ClassBasedCycleTime,
                                                 ClosestOpenLocation)


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


storage_policies = get_storage_strategies([2, 3, 5])

params = SimulationParameters(
    use_case="wepastacks",
    use_case_n_partitions=1,
    use_case_partition_to_use=0,
    n_agvs=40,
    generate_orders=False,
    verbose=False,
    resetting=False,
    initial_pallets_storage_strategy=ClassBasedPopularity(
        retrieval_orders_only=False,
        future_counts=True,
        init=True,
        # n_zones changes dynamically based on the slap strategy
        # in get_episode_env
        n_zones=2
    ),
    pure_lanes=True,
    n_levels=3,
    # https://logisticsinside.eu/speed-of-warehouse-trucks/
    agv_speed=2,
    unit_distance=1.4,
    pallet_shift_penalty_factor=20,  # in seconds
    compute_feature_trackers=True
)

if __name__ == '__main__':
    n_strategies = len(storage_policies)
    for i in range(0, n_strategies):
        run_episode(simulation_parameters=params,
                    storage_strategy=storage_policies[i],
                    print_freq=1000, warm_start=True,
                    log_dir='./result_data_wepa/')

    # parallelize_heterogeneously(
    #     [run_episode] * n_strategies,
    #     list(zip(storage_policies,
    #              [j] * n_strategies)))
