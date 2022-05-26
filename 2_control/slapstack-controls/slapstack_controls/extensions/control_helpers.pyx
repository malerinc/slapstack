# distutils: language=c++
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp cimport bool
cimport numpy as np
from libcpp.limits cimport numeric_limits


cpdef map[int, int] count_sku(list order_list,
                              bool retrieval_only):
    cdef map[int, int] order_counts
    for order in order_list:
        if retrieval_only and order.type != 'retrieval':
            continue
        sku = order.SKU
        if order_counts.find(sku) != order_counts.end():
            order_counts[sku] += 1
        else:
            order_counts[sku] = 1
    return order_counts


cpdef get_oldest_batch(
        vector[(int, int, int)] sku_pos,
        np.ndarray[np.float32_t, ndim=3] batch_matrix,
        np.ndarray[np.float32_t, ndim=3] arrival_time_matrix,
        # vector[vector[vector[int]]] batch_matrix,
        # vector[vector[vector[int]]] arrival_time_matrix,
        map[np.float32_t, np.float32_t] arrival_time_map):
    cdef numeric_limits[double] lm
    cdef float max_batch_arrival = -lm.infinity()
    cdef (int, int, int) choice_loc
    cdef float choice_arrival  = -lm.infinity()
    cdef:
        (int, int, int) loc
        float batch_id
        float batch_arrival_time
        float arrival_time
    for sku_location in sku_pos:
        # loc info
        loc = sku_location
        batch_id = batch_matrix[loc[0]][loc[1]][loc[2]]
        batch_arrival_time = arrival_time_map[batch_id]
        arrival_time = arrival_time_matrix[loc[0]][loc[1]][loc[2]]
        # decide if location is a candidate
        if batch_arrival_time > max_batch_arrival:
            max_batch_arrival = batch_arrival_time
            choice_loc = loc
            choice_arrival = arrival_time
        elif batch_arrival_time == max_batch_arrival:
            if arrival_time > choice_arrival:
                choice_loc = loc
                choice_arrival = arrival_time
        else:
            continue
    return choice_loc
