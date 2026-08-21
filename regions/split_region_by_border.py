import numpy as np
from util.find_closest_points import find_closest_points

def get_unfiltered_distances_and_connections(loops):
    candidate_connections = np.zeros([len(loops), len(loops) + 1, 2], dtype=np.int32)
    candidate_connection_distances = np.zeros([len(loops), len(loops) + 1])
    for loop_index_1, loop_1 in enumerate(loops[:-1]):
        for loop_index_2, loop_2 in enumerate(loops[loop_index_1 + 1:]):
            loop_2_points = loop_2.points
            connection, distance = find_closest_points(loop_1.points, loop_2_points)
            candidate_connections[loop_index_1, loop_index_2] = connection
            candidate_connections[loop_index_2, loop_index_1] = connection
            candidate_connection_distances[loop_index_1, loop_index_2] = distance
            candidate_connection_distances[loop_index_2, loop_index_1] = distance
            if loop_index_1 == 0:
                loop_2_size = loop_2_points.shape[1]
                alternate_loop_2_connection_point = loop_2_points[[(connection + loop_2_size // 2) % loop_2_size]]
                connection, distance = find_closest_points(loop_1.points, alternate_loop_2_connection_point)
                candidate_connections[loop_index_2, len(loops)] = [connection[0], alternate_loop_2_connection_point]
                candidate_connection_distances[loop_index_2, len(loops)] = distance
    return candidate_connections, candidate_connection_distances


def filter_distances(unfiltered_distances):
    take_indexes = np.array([
        [take_index for take_index in range(unfiltered_distances.shape[1]) if take_index != row_index]
        for row_index in range(unfiltered_distances.shape[0])
    ])
    filtered_distances = np.take_along_axis(unfiltered_distances, take_indexes, axis=1)
    take_indexes[:, -1] = -1
    return filtered_distances, take_indexes


def make_available_distances_and_take_indexes(distances, take_indexes, taken_list):
    nonzero_taken = np.array([taken_index for taken_index in taken_list if taken_index != 0])
    indices_to_keep = np.nonzero(
        np.logical_not(
            np.isin(nonzero_taken, nonzero_taken)
        )
    )
    last_taken = taken_list[-1]
    if last_taken == -1:
        last_taken = 0
    distances_to_keep = distances[last_taken, *indices_to_keep]
    take_indexes_to_keep = take_indexes[last_taken, *indices_to_keep]
    return distances_to_keep, take_indexes_to_keep, last_taken


def take_next(distances, take_indexes, taken_list):
    available_distances, available_take_indexes, last_taken = (
        make_available_distances_and_take_indexes(distances, take_indexes, taken_list))

    if last_taken == 0:
        if len(available_distances) == 1:
            return None
        else:
            return take_indexes[np.argmin(available_distances[:-1])]
    else:
        return take_indexes[np.argmin(available_distances)]


def find_new_borders(distances, take_indexes, taken_list):
    next_to_take = take_next(distances, take_indexes, taken_list)
    if next_to_take is None:
        return []
    else:
        return [next_to_take, find_new_borders(distances, take_indexes, [*taken_list, next_to_take])]


def split_region(loops):
    candidate_connections, candidate_connection_distances_unfiltered = get_unfiltered_distances_and_connections(loops)
    distances, take_indexes = filter_distances(candidate_connection_distances_unfiltered)



