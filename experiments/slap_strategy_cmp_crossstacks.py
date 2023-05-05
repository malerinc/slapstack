from experiments.experiment_commons import run_episode
# from slapstack.helpers import parallelize_heterogeneously
from slapstack.interface_templates import SimulationParameters
from slapstack_controls.storage_policies import ClosestOpenLocation
# from slapstack_controls.storage_policies import OriginalShortestLeg
from slapstack_controls.storage_policies import RandomOpenLocation
from slapstack_controls.storage_policies import ClosestToNextRetrieval
from slapstack_controls.storage_policies import ClosestToDestination
from slapstack_controls.storage_policies import ShortestLeg


def get_storage_strategies():
    storage_strategies = []
    storage_strategies += [
        ClosestOpenLocation(very_greedy=False),
        RandomOpenLocation(),
        ClosestToNextRetrieval(very_greedy=False),
        ClosestToDestination(very_greedy=False),
        ShortestLeg(very_greedy=False)
    ]
    return storage_strategies


params = SimulationParameters(
    use_case="crossstacks",
    use_case_n_partitions=1,
    use_case_partition_to_use=0,
    n_agvs=3,
    generate_orders=False,
    verbose=False,
    resetting=False,
    initial_pallets_storage_strategy=ClosestOpenLocation(),
    pure_lanes=False,
    # https://logisticsinside.eu/speed-of-warehouse-trucks/
    agv_speed=1.2,
    unit_distance=1.1,
    pallet_shift_penalty_factor=90,  # in seconds
    material_handling_time=45,
    compute_feature_trackers=True,
    n_levels=1,
    door_to_door=True,
    # update_partial_paths=False,
    agv_forks=1
)

if __name__ == '__main__':

    storage_policies = get_storage_strategies()
    n_strategies = len(storage_policies)
    constraints_breached = True
    n_agv = 7
    while constraints_breached:
        n_agv += 1
        params.n_agvs = n_agv
        constraints_breached = run_episode(
            simulation_parameters=params,
            storage_strategy=ClosestToDestination(),
            print_freq=1000,
            stop_condition=True,
            log_dir='./result_data_crosstacks/'
        )

    # params.n_agvs = n_agv
    # done_prematurely = parallelize_heterogeneously(
    #     [run_episode] * n_strategies,
    #     list(zip([params] * n_strategies,                    # params
    #              storage_policies,                           # policy
    #              [0] * n_strategies,                         # print_freq
    #              [False] * n_strategies,
    #              ['./result_data_dachser/'] * n_strategies,
    #              [True] * n_strategies,
    #              )))
