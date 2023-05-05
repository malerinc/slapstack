import logging
from copy import deepcopy
from enum import IntEnum
from os.path import join, exists
from os import makedirs
import marshal


# <editor-fold desc="Utility Functions">
from typing import List, Callable, Tuple, Dict, Union

import numpy as np
# noinspection PyPackageRequirements, PyProtectedMember
from joblib._multiprocessing_helpers import mp
from numpy import ndarray


# from slapstack.envs.interface_templates import SimulationParameters


def hole_found(storage_matrix, locations):
    empty_space = False
    elements = []
    for loc in locations:
        element = storage_matrix[loc]
        elements.append(element)
        if element == 0:
            empty_space = True
        if empty_space and element != 0:
            return True, elements
    return False, elements


def check_for_holes(storage_matrix, lane_clusters):
    """
    TODO: look into and document! Looks fishy ...
    :param storage_matrix:
    :param lane_clusters:
    :return:
    """
    for aisle, lane_dict in lane_clusters.items():
        fill_nfo = check_for_holes_by_direction(
            lane_dict, storage_matrix, AccessDirection.ABOVE)
        storage_locations_in_above_lane = fill_nfo[0]
        storage_locations_in_below_lane = fill_nfo[1]
        hole_found_bool = fill_nfo[2]
        elements = fill_nfo[3]

        if hole_found_bool:
            print("aisle: ", aisle)
            print(storage_locations_in_above_lane)
            print(elements)
            return True

        fill_nfo = check_for_holes_by_direction(
            lane_dict, storage_matrix, AccessDirection.BELOW)
        storage_locations_in_above_lane = fill_nfo[0]
        storage_locations_in_below_lane = fill_nfo[1]
        hole_found_bool = fill_nfo[2]
        elements = fill_nfo[3]

        if hole_found_bool:
            print("aisle: ", aisle)
            print(storage_locations_in_above_lane)
            print(elements)
            return True
    return False


def check_for_holes_by_direction(lane_dict, storage_matrix, direction):
    """
    TODO: Look into and document!
    :param lane_dict:
    :param storage_matrix:
    :param direction:
    :return:
    """
    storage_locations_in_above_lane = []
    storage_locations_in_below_lane = []
    if lane_dict[direction]:
        for loc in lane_dict[direction]:
            for z in range(0, 3):
                loc_with_level = loc + (z,)
                storage_locations_in_above_lane.append(loc_with_level)
    hole_found_bool, elements = hole_found(storage_matrix,
                                           storage_locations_in_above_lane)
    return (storage_locations_in_above_lane,
            storage_locations_in_below_lane, hole_found_bool, elements)


def create_folders(path):
    """
    Switches between '/' (POSIX) and '\'(windows) separated paths, depending on
    the current platform creating all non existing folders.

    :param path: A '/' separated *relative* path; the last entry is considered
        to be the file name and won't get created.
    :return: The platform specific file path.
    """
    segments = path.split('/')
    if not bool(segments):
        return path
    path_dir = join(*segments[:-1])
    file = segments[-1]
    if not exists(path_dir):
        makedirs(path_dir)
    return join(path_dir, file)
# </editor-fold>


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """


class UndefinedLegalActionCall(Error):
    def __init__(self, strategy_configuration, current_order):
        self.strategy_configuration = current_order
        self.current_order = current_order

    def __str__(self):
        return "A decision could not be made for a {0} order " \
               "with the representation {1}".\
            format(self.current_order, repr(self.strategy_configuration))


def print_3d_np(np_array, time=False):
    n_levels = np_array.shape[2]
    output = ""
    for i in range(0, n_levels):
        output += "level {}\n".format(i + 1)
        output += "   "
        for col in range(0, np_array.shape[1]):
            output += "{}".format(col).rjust(3)
        output += "\n"
        row = 0
        for s in np_array[:, :, i]:
            output += "{}".format(row).rjust(3)
            for elem in s:
                if elem == -1:
                    elem = "w"
                    if time:
                        elem = '-'
                if elem == -2:
                    elem = "a"
                if elem == 0:
                    elem = "-"
                    if time:
                        elem = 0
                if elem == -3:
                    elem = "so"
                if elem == -4:
                    elem = "si"
                if elem == -5:
                    elem = "m"
                output += "{}".format(elem).rjust(3)
            row += 1
            # print("{}".format(elem).rjust(3), end="")
            output += "\n"
    print(output)


# def get_use_case_parameters():
#     environment_parameters = SimulationParameters(
#         n_rows=13,
#         n_columns=13,
#         n_levels=2,
#         n_agvs=2,
#         n_skus=2,
#         generate_orders=False,
#         n_orders=None,
#         desired_fill_level=0.2,
#         verbose=False,
#         state_stack_size=1,
#         resetting=False,
#         order_list=None,
#         initial_pallets_sku_counts=None,
#         initial_pallets_storage_strategy=None,
#         n_sources=1,
#         n_sinks=1,
#         use_cases=None,
#         pure_lanes=False
#     )
#     seed = 123456
#     # log_path = '../3_visualization_d3js/sim_data/'
#     log_path = ''
#     return environment_parameters, seed, log_path


def create_storage_matrix(n_rows, n_columns, n_levels,
                          initial_indices, initial_SKUs, pallets_only):
    s = np.zeros((n_rows, n_columns, n_levels))
    source_sink_row = int(n_rows / 2)
    for i, SKU in zip(initial_indices, initial_SKUs):
        s[i[0]][i[1]][0] = SKU
    s = s.astype(int)
    if pallets_only:
        return s
    s[1, :, 0] = StorageKeys.AISLE.value
    s[:, 1, 0] = StorageKeys.AISLE.value
    s[n_rows - 2, :, 0] = StorageKeys.AISLE.value
    s[:, n_columns - 2, 0] = StorageKeys.AISLE.value
    s[:, 0, 0] = StorageKeys.WALL.value
    s[:, n_columns - 1, 0] = StorageKeys.WALL.value
    s[0, :, 0] = StorageKeys.WALL.value
    s[n_rows - 1, :, 0] = StorageKeys.WALL.value
    s[source_sink_row, 0:n_columns - 1, 0] = StorageKeys.AISLE.value
    s[source_sink_row, 0, 0] = StorageKeys.SOURCE.value
    s[source_sink_row, n_columns - 1, 0] = StorageKeys.SINK.value
    return s


def faster_deepcopy(obj, memo):
    cls = obj.__class__
    result = cls.__new__(cls)
    marshal_set = {list, tuple, set, str}
    basic_types = {int, bool, float, str}
    memo[id(obj)] = result
    for k, v in obj.__dict__.items():
        if id(v) in memo:
            setattr(result, k, memo[id(v)])
        elif type(v) == np.ndarray:
            arr_cp = v.copy()
            memo[id(v)] = arr_cp
            setattr(result, k, arr_cp)
        elif type(v) in marshal_set:
            if len(v) == 0 or next(iter(v)).__class__ in basic_types:
                collection_cp = marshal.loads(marshal.dumps(v))
                memo[id(v)] = collection_cp
                setattr(result, k, collection_cp)
            else:  # len(v) != 0 and next(iter(v)).__class__ not in basic_types
                # noinspection PyArgumentList
                collection_cp = deepcopy(v, memo)
                memo[id(v)] = collection_cp
                setattr(result, k, collection_cp)
        else:
            # noinspection PyArgumentList
            setattr(result, k, deepcopy(v, memo))
    return result


def parallelize_heterogeneously(fns: List[Callable],
                                args: List[Tuple]):
    logging.debug("parallelizing")
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        print("RunTimeError")
        pass
    n_threads = len(fns)
    assert n_threads == len(args)
    # n_threads = 1
    # n_threads = 24 # limit of 24
    returns = []
    # pool = NoDaemonPool(n_threads)  # we need pthos mp for this
    pool = mp.Pool(n_threads)
    workers = [
        pool.apply_async(fns[i], args=args[i])
        for i in range(n_threads)]
    for w in workers:
        fn_return = w.get()
        returns.append(fn_return)
    pool.close()
    pool.join()
    return returns


class AccessDirection(IntEnum):
    ABOVE = -1
    BELOW = 1


class StorageKeys(IntEnum):
    EMPTY = 0
    WALL = -1
    AISLE = -2
    SOURCE = -3
    SINK = -4
    MID_AISLE = -5


class VehicleKeys(IntEnum):
    N_AGV = -1
    FREE = 0
    BUSY = 1


class BatchKeys(IntEnum):
    NAN = -1


class TimeKeys(IntEnum):
    NAN = -1


class TravelEventKeys(IntEnum):
    RETRIEVAL_1STLEG = 0
    RETRIEVAL_2ND_LEG = 1
    DELIVERY_1ST_LEG = 2
    DELIVERY_2ND_LEG = 3


class PathKeys:
    ACCESS_TO_ACCESS = 1
    ACCESS_TO_AISLE = 2
    ACCESS_TO_STORAGE = 3
    AISLE_TO_ACCESS = 4
    AISLE_TO_AISLE = 5
    AISLE_TO_STORAGE = 6
    STORAGE_TO_AISLE = 7
    STORAGE_TO_STORAGE = 8
    STORAGE_TO_ACCESS = 9


def ravel(position: Tuple[int, int, int], dims: Tuple[int, int, int]) -> int:
    int_position = (position[0] * dims[1] * dims[2]
                    + position[1] * dims[2] + position[2])
    return int_position


def ravel2(position: Tuple[int, int], shape: Tuple[int, int]) -> int:
    return position[0] * shape[1] + position[1]


def unravel(position_enc: int, dims) -> Tuple[int, int, int]:
    z: int = position_enc % dims[2]
    y: int = (position_enc // dims[2]) % dims[1]
    x: int = (position_enc // dims[2]) // dims[1]
    return x, y, z


def unravel2(int_encoding: int, shape: Tuple[int, int]) -> Tuple[int, int]:
    y: int = int_encoding % shape[1]
    x: int = int_encoding // shape[1]
    return x, y
