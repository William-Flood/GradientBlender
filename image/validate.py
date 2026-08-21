import numpy as np
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components
from util.inspection_utils import *
from PIL import Image

offset_matrix = np.array([
    [[1, 1, 0, -1, -1, -1, 0, 1]],
    [[0, 1, 1, 1, 0, -1, -1, -1]]
])
neighbor_adjacent_matrix = np.array([[
       [False,  True,  True, False, False, False,  True,  True],
       [True,  False,  True, False, False, False, False, False],
       [True,  True,  False,  True,  True, False, False, False],
       [False, False,  True,  False,  True, False, False, False],
       [False, False,  True,  True,  False,  True,  True, False],
       [False, False, False, False,  True,  False,  True, False],
       [True, False, False, False,  True,  True,  False,  True],
       [True, False, False, False, False, False,  True,  False]]
])
def get_points_around_set(initial_set):
    return initial_set[:, :, np.newaxis] + offset_matrix


def validate(
        values_array,
        validation_highlight_file_name="validation_failures.png",
        validation_point_list_file_name="validation_failure_list.txt"
):
    """
    Used to ensure that no single-width lines exist within the region - the fill algorithm depends on identifying
    region borders as loops of points and backtracking over a border point isn't supported.  A ValueError
    will be raised if invalid geometry is detected
    Args:
        values_array:

    Returns:

    """
    padded_value_points = np.pad(values_array, ((1, 1), (1, 1)), mode="constant", constant_values=-1)
    value_points = np.reshape(np.indices(values_array.shape), [2, -1])[
        :, np.not_equal(values_array.flatten(), -1)
    ]
    neighbors_values = padded_value_points[
        *get_points_around_set(value_points + 1)
    ]
    neighbor_value_matches = np.equal(
            np.reshape(values_array[*value_points], [-1, 1]), neighbors_values
        )
    neighbor_total_matches = np.sum(neighbor_value_matches, axis=1)
    line_tips = value_points[:, np.equal(neighbor_total_matches.astype(np.int32),1)]
    potential_invalid = np.flatnonzero(np.equal(neighbor_total_matches, 2))
    if potential_invalid.shape[0] > 0:
        potential_invalid_neighbors = neighbor_value_matches[potential_invalid]
        potential_invalid_local_graphs = (
                (potential_invalid_neighbors[:, :, np.newaxis] & potential_invalid_neighbors[:, np.newaxis, :]) |
                (
                        np.logical_not(potential_invalid_neighbors[:, :, np.newaxis]) &
                        np.logical_not(potential_invalid_neighbors[:, np.newaxis, :])
                )
        ) & neighbor_adjacent_matrix
        connected_neighbors_indices = np.indices(potential_invalid_local_graphs.shape)
        combined_connected_neighbors_indices = np.reshape(
            connected_neighbors_indices[[1, 2]] + connected_neighbors_indices[[0]] * 8,
            [2, -1]
        )
        connected_neighbors_graph = coo_array(
            (potential_invalid_local_graphs.flatten(), combined_connected_neighbors_indices),
            shape=[potential_invalid.shape[0] * 8] * 2,
            dtype=np.bool
        ).toarray()
        _, neighbor_component_labels = connected_components(connected_neighbors_graph, directed=False)
        potential_invalid_index_and_neighbor_labels = np.stack(
            [
                np.concat([[point_index] * 8 for point_index in potential_invalid]),
                neighbor_component_labels
            ],
            axis=0
        )
        unique_labels_per_candidate = np.unique(potential_invalid_index_and_neighbor_labels, axis=1)
        candidate_index, candidate_components = np.unique(unique_labels_per_candidate[0], return_counts=True)
        single_width_line_point_indices = candidate_index[np.not_equal(candidate_components, 2)]
        invalid_points = np.concat([
            line_tips,
            value_points[:, single_width_line_point_indices]
        ], axis=1)
    else:
        invalid_points = line_tips
    if invalid_points.shape[1] > 0:
        validation_array = np.zeros([*values_array.shape, 4], dtype=np.uint8)
        validation_array[*value_points] = np.array([
            values_array[*value_points],
            values_array[*value_points],
            values_array[*value_points],
            [255] * value_points.shape[1]
        ]).T
        validation_array[*invalid_points] = [255, 0, 0, 255]
        Image.fromarray(validation_array).save(validation_highlight_file_name)
        with open(validation_point_list_file_name, "w") as list_file:
            list_file.writelines([f"{point[0]}, {point[1]}" for point in invalid_points.T])
        raise Exception(f"Invalid image geometry detected. Validation issues highlighted in {validation_highlight_file_name} and {validation_point_list_file_name}")
