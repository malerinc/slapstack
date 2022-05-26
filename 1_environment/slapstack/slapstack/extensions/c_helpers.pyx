# distutils: language=c++

cpdef get_first_tile(
    set occupied_locations, set locked_locations, list lane,
    int lane_length, int n_levels, (int, int, int) s_shape):
    cdef int i = lane_length - 1
    cdef:
        (int, int, int) loc
        (int, int) storage_tile
        int j
        int int_loc
    while i > 0:
        j = n_levels - 1
        storage_tile = lane[i]
        # print(storage_tile)
        # print(storage_tile)
        while j > 0:
            loc = (storage_tile[0], storage_tile[1], j)
            int_loc = c_ravel(loc, s_shape)
            j -= 1
            # print(occupied_locations)
            if int_loc in occupied_locations and int_loc not in locked_locations:
                return int_loc
        i -= 1

cpdef int c_ravel((int, int, int) position, (int, int, int) shape):
    return (position[0] * shape[1] * shape[2]
            + position[1] * shape[2] + position[2])

cpdef c_unravel(int int_encoding, (int, int, int) shape):
    cdef int z = int_encoding % shape[2]
    cdef int y = (int_encoding // shape[2]) % shape[1]
    cdef int x = (int_encoding // shape[2]) // shape[1]
    return x, y, z
