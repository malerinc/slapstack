import json

from os.path import sep
from os.path import join, abspath
from collections import defaultdict
from typing import Tuple, Dict, List, Set, Mapping

import numpy as np
from numpy import genfromtxt


class UseCasePartition:
    """order = (order_type, sku, time, entrance/exit id, production
    batch/shipping id)"""
    layout = None
    ROOT_DIR = sep.join([sep.join(
        abspath(__file__).split(sep)[:-1]), "WEPAStacks"])
    INITIAL_PALLETS_PATH = 'Initial_fill_lvl.json'
    ORDERS_PATH = 'Orders_v5.json'
    LAYOUT_PATH = sep.join(['layouts', 'layout1_middle_aisles.csv'])
    MINI_LAYOUT_PATH = sep.join(['layouts', 'layout1_mini_middle_aisles.csv'])

    def __init__(self, initial_skus: Dict[int, int], all_skus: Set[int],
                 current_week: int = 0, first_order: int = 0):
        self.distinct_skus = all_skus
        self.order_list: List[Tuple[str, int, int, int, int]] = []
        self.initial_skus = {current_week: initial_skus,
                             current_week + 1: initial_skus.copy()}
        self.sku_in_counts = {current_week: defaultdict(int)}
        self.sku_out_counts = {current_week: defaultdict(int)}
        if UseCasePartition.layout is None:
            UseCasePartition.layout = UseCasePartition.get_layout(
                join(UseCasePartition.ROOT_DIR, UseCasePartition.LAYOUT_PATH)
            )
        self.n_levels = 3
        self.first_order = first_order
        self.last_order = first_order
        self.current_week = current_week

    @staticmethod
    def get_layout(path: str):
        data = genfromtxt(path, delimiter=',')
        data = data.astype(int)
        shape = data.shape
        data = np.delete(data, shape[1] - 1, axis=1)
        return data

    def add_order(self, order: Tuple[str, int, int, int, int]):
        """
        The order entries from left to right correspond to order_type, SKU,
        arrival time, source/sink (depending on the order type), batch and
        period.

        :param order: The iterable containing order information.
        :return: None.
        """
        if order[-1] != self.current_week:
            # start new future counts
            self.current_week = order[-1]
            self.initial_skus[self.current_week + 1] = (
                self.initial_skus[self.current_week].copy())
            self.sku_in_counts[self.current_week] = defaultdict(int)
            self.sku_out_counts[self.current_week] = defaultdict(int)
        self.order_list.append(tuple(order))
        key = order[1]
        if key not in self.distinct_skus:
            self.distinct_skus.add(key)
        if order[0] == 'delivery':
            self.initial_skus[self.current_week + 1][key] += 1
            self.sku_in_counts[self.current_week][key] += 1
        else:
            assert order[0] == 'retrieval'
            self.initial_skus[self.current_week + 1][key] -= 1
            self.sku_out_counts[self.current_week][key] += 1
        self.last_order += 1

    @staticmethod
    def load_initial_skus():
        with open(join(UseCasePartition.ROOT_DIR,
                       UseCasePartition.INITIAL_PALLETS_PATH)) as json_file:
            initial_fill_json = json.load(json_file)
        skus_ini = defaultdict(int)
        all_skus = set(skus_ini.keys())
        for sku, amount in initial_fill_json.items():
            skus_ini[int(sku)] = amount
            all_skus.add(int(sku))
        return skus_ini, all_skus

    @staticmethod
    def load_orders():
        with open(join(UseCasePartition.ROOT_DIR,
                       UseCasePartition.ORDERS_PATH)) as json_file:
            order_data = json.load(json_file)
        return order_data


def get_minimum_coverage_initial_skus(
        orders: List[Tuple[str, int, int, int, int]]) -> Dict[int, int]:
    """
    Calculates the minimum number of initial skus such that all retrieval orders
    can be serviced by the end of the simulation. This is done by subtracting
    the number of delivery skus from the number of retrieval skus.

    Note that some this could lead to long service times for some delivery
    orders.

    :param orders: The orders that will be fed into the simulation.
    :return: The dictionary mapping SKU to the amounts present at the beginning
        of the simulation.
    """
    sku_counts = defaultdict(int)
    for order in orders:
        order_type, sku = order[0], order[1]
        sku_counts[sku] += 1 if order_type == 'retrieval' else -1
    for sku, count in sku_counts.items():
        if count < 0:
            sku_counts[sku] = 0
    return sku_counts


def get_nowait_initial_skus(
        orders: List[Tuple[str, int, int, int, int]]) -> Dict[int, int]:
    """
    Calculates the minimum number of initial skus in the warehouse such that all
    arriving orders can be directly serviced, provided an agv is free. This
    means that retrieval orders will never have to wait for future deliveries.

    Note that this could result in more skus than available storage spaces.

    :param orders: The orders that will be fed into the simulation.
    :return: The dictionary mapping SKU to the amounts present at the beginning
        of the simulation.
    """
    min_skus = defaultdict(int)
    sku_counts = defaultdict(int)
    for order in orders:
        order_type, sku = order[0], order[1]
        sku_counts[sku] += -1 if order_type == 'retrieval' else 1
        if sku_counts[sku] < -min_skus[sku]:
            min_skus[sku] = -sku_counts[sku]
    return min_skus


def get_unique_skus(orders: List[Tuple[str, int, int, int, int]]) -> Set[int]:
    """
    Creates a set of the unique SKUs present over all the orders.

    :param orders: The orders that will be fed into the simulation.
    :return: The set of SKUs over all orders.
    """
    skus = set()
    for order in orders:
        skus.add(order[1])
    return skus


def get_initial_skus(orders: List[Tuple[str, int, int, int, int]],
                     warehouse_capacity: int = 19000,
                     initial_fill_level: float = 0.7,
                     percent_nowait: float = 0.8):
    """
    Combines get_nowait_initial_skus and get_minimum_coverage_initial_skus to
    create initial SKU numbers in such a way that the percent of SKUs that can
    be serviced instantaneously can be parameterized while ensuring that all
    orders can be serviced by the end of the simulation, assuming that no
    deadlock ensues.

    :param orders: The orders that will be passed to the simulation.
    :param warehouse_capacity: The maximum number of pallets in the warehouse.
    :param initial_fill_level: The desired initial fill level.
    :param percent_nowait: The relative amount of retrieval orders that should
        be serviceable without waiting for future orders.
    :return:
    """
    ini_sku_nw: Mapping[int, int] = get_nowait_initial_skus(orders)
    ini_sku_min: Dict[int, int] = get_minimum_coverage_initial_skus(orders)

    sorted_x = sorted(
        ini_sku_nw.items(), key=lambda kv: kv[1], reverse=True)
    n_ini_sku, i = 0, 0
    while n_ini_sku / warehouse_capacity < initial_fill_level:
        n_sku = int(percent_nowait * sorted_x[i][1])
        sku = sorted_x[i][0]
        # print(sku)
        ini_sku_min[sku] = max(n_sku, ini_sku_min[sku])
        n_ini_sku += ini_sku_min[sku]
        i += 1
        if i >= len(ini_sku_min):
            break
    return ini_sku_min


def partition_use_case(root_dir, n_partitions=40):
    """
    Splits the use case orders into n_partitions equal sections.

    :return: The use case partitions.
    """
    order_data = UseCasePartition.load_orders()
    all_skus = get_unique_skus(order_data)
    # skus_ini = get_initial_skus(order_data,
    #                             warehouse_capacity=19000,
    #                             initial_fill_level=0.6,
    #                             percent_nowait=0.8)
    skus_ini, _ = UseCasePartition.load_initial_skus()
    part_size = int(len(order_data) / n_partitions)
    week = order_data[0][-1]
    uc = UseCasePartition(skus_ini, all_skus)
    use_case_partitions = [uc]
    for order in order_data:
        if uc.last_order - uc.first_order >= part_size:
            # start a new partition
            uc = UseCasePartition(uc.initial_skus[uc.current_week].copy(),
                                  uc.distinct_skus, uc.current_week,
                                  uc.last_order)
            use_case_partitions.append(uc)
        uc.add_order(order)
    return use_case_partitions
