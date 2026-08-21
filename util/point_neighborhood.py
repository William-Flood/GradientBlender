import numpy as np
from scipy.sparse import coo_array

offset_matrix = np.array([
    [[1, 1, 0, -1, -1, -1, 0, 1]],
    [[0, 1, 1, 1, 0, -1, -1, -1]]
])


def get_neighbors(initial_set):
    return initial_set[:, :, np.newaxis] + offset_matrix


def get_neighbor_values(point_list, point_values, initial_set):
    offset_index_map = coo_array(
        (point_values + 1, point_list + 1),
        shape=(np.max(point_list, axis=1) + 3)
    )
    set_neighbors = initial_set[:, :, np.newaxis] + offset_matrix
    offset_set_neighbors = set_neighbors + 1
    return offset_index_map[*offset_set_neighbors].toarray() - 1


def get_neighbor_values_over_array(values_array, initial_set):
    padded_array_values = np.pad(values_array, ((1, 1), (1, 1)), mode="constant", constant_values=-1)
    set_neighbors = initial_set[:, :, np.newaxis] + offset_matrix
    offset_set_neighbors = set_neighbors + 1
    return padded_array_values[*offset_set_neighbors]



def get_neighbor_boolean_mask(point_list, initial_set):
    offset_index_map = coo_array(
        ([True] * point_list.shape[1], point_list + 1),
        shape=(np.max(point_list, axis=1) + 3)
    )
    set_neighbors = initial_set[:, :, np.newaxis] + offset_matrix
    offset_set_neighbors = set_neighbors + 1
    return offset_index_map[*offset_set_neighbors].toarray()
