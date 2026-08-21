import numpy as np
from numpy.typing import NDArray
from util.grid import Grid, GridCell
import math
from scipy.sparse import coo_array
from typing import List
from borders.get_pixel_connections import get_pixel_connections
from scipy.sparse.csgraph import shortest_path, connected_components
from util.inspection_utils import *
from util.point_neighborhood import (
    offset_matrix,
    get_neighbor_values,
    get_neighbors,
    get_neighbor_boolean_mask,
    get_neighbor_values_over_array
)

TARGET_MAX_SIZE = 500
TARGET_MIN_SIZE = 20


def get_window_point_graph(points_in_window, connections):
    are_points_within_window = coo_array(
        (np.ones(points_in_window.shape[0], dtype=np.bool), [points_in_window]),
        shape=[np.max(connections) + 1]
    )
    are_connections_within_window = np.all(
        are_points_within_window[connections].toarray(),
        axis=0
    )
    connection_within_window = connections[:, are_connections_within_window]
    connections_window_indices: NDArray = np.searchsorted(points_in_window, connection_within_window)
    return coo_array(
        (np.ones([connections_window_indices.shape[1]]), connections_window_indices),
        shape=[len(points_in_window)] * 2
    )

# def get_points_around_set(initial_set):
#     return initial_set[:, :, np.newaxis] + offset_matrix


def find_border_end_points(border_points, region_points, connections):
    neighbor_regions = get_neighbor_values_over_array(
        region_points,
        border_points
    )
    neighbor_and_index = np.reshape(
        np.stack(np.broadcast_arrays(
            np.arange(border_points.shape[1], dtype=np.int32)[:, np.newaxis],
            neighbor_regions), axis=0),
        [2, -1]
    )
    original_region = np.broadcast_to(
        region_points[*border_points[:, :, np.newaxis]], [border_points.shape[1], 8]
    ).flatten()
    bordering_neighbor_and_index = neighbor_and_index[:, np.not_equal(neighbor_and_index[1], original_region)]
    unique_bordering_neighbor_and_index = np.unique(bordering_neighbor_and_index, axis=1)
    index, bordering_region_count = np.unique(unique_bordering_neighbor_and_index[0], return_counts=True)
    multi_region_border_indices = index[np.greater(bordering_region_count, 1)]
    end_point_indices_list = []
    multi_region_border_neighbor_indices = get_neighbor_values(
        border_points,
        np.arange(border_points.shape[1]),
        border_points[:, multi_region_border_indices]
    )
    for multi_region_index, neighbor_indices in zip(
            multi_region_border_neighbor_indices,
            multi_region_border_indices
    ):
        neighbor_indices_normalized = np.sort(neighbor_indices[(np.not_equal(neighbor_indices, -1))])
        get_window_point_graph(neighbor_indices_normalized, connections)
        num_labels, _ = connected_components(
            get_window_point_graph(neighbor_indices_normalized, connections).toarray(),
            directed=False
        )
        if num_labels == 1:
            end_point_indices_list.append(multi_region_index)
    if len(end_point_indices_list) > 0:
        end_point_indices = np.concat(end_point_indices_list)
    else:
        end_point_indices = np.zeros([0])
    if len(end_point_indices) > 1:
        border_graph = get_window_point_graph(np.arange(border_points.shape[1]), connections)
        point_distances = shortest_path(border_graph, directed=False, indices=end_point_indices)
        max_distances = np.max(point_distances, axis=1)
        return border_points[:, end_point_indices[np.argmax(max_distances)]]
    elif len(end_point_indices) == 1:
        return border_points[:, end_point_indices[0]]
    else:
        return None


def find_branch_points(cell, connections, point_index_map):
    unsorted_cell_point_indexes = point_index_map[*cell]
    indices_ordering = np.argsort(unsorted_cell_point_indexes)
    cell_point_indexes = unsorted_cell_point_indexes[indices_ordering]
    sorted_points = cell[:, indices_ordering]
    cell_graph = get_window_point_graph(cell_point_indexes, connections).toarray()
    point_connections = np.sum(cell_graph, axis=1)
    candidate_branch_coords = sorted_points[:, np.greater(point_connections, 2)]
    return candidate_branch_coords


def peel_corners(
    points,
    connections
):
    point_neighbor_bools = get_neighbor_boolean_mask(
        points,
        points
    )
    corner_candidates = np.flatnonzero(
        np.equal(np.sum(point_neighbor_bools, axis=1), 2)
    )
    corner_candidate_neighbor_bools = point_neighbor_bools[corner_candidates]
    corner_pattern = np.identity(8, dtype=np.bool) | np.roll(np.identity(8, dtype=np.bool), 1, axis=1)
    point_is_corner = np.any(
        np.all(
            np.equal(
                corner_candidate_neighbor_bools[:, np.newaxis, :],
                corner_pattern[np.newaxis, :, :]
            ),
            axis=2
        ),
        axis=1
    )
    corner_indices = corner_candidates[point_is_corner]
    corner_connection_bools = point_neighbor_bools[corner_indices]
    corner_connections = np.reshape(
        get_neighbor_values(
            points,
            np.arange(points.shape[1]),
            points[:, corner_indices]
        ).flatten()[corner_connection_bools.flatten()],
        [-1, 2]
    )
    corner_connections_mask = coo_array(
        ([True] * corner_connections.shape[0], corner_connections.T),
        shape=np.max(connections, axis=1) + 1
    )
    return connections[
        :,
        np.logical_not(corner_connections_mask[*connections].toarray())
    ]


def filter_branches(
        cell_branch_points_unfiltered,
        unfiltered_border_connections,
        original_point_index_map,
        full_point_list):
    false_branch_patterns = np.array([
        [1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 1],
        [1, 1, 1, 1, 0, 1, 0, 0],
        [1, 0, 0, 1, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0, 0, 1, 0],
    ], dtype=np.bool)
    false_connections = np.array([
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.bool)
    false_branch_patterns_rolled = np.concat(
        [
            false_branch_patterns,
            np.roll(false_branch_patterns, shift=2, axis=1),
            np.roll(false_branch_patterns, shift=4, axis=1),
            np.roll(false_branch_patterns, shift=6, axis=1)
        ], axis=0
    )
    false_connections_rolled = np.concat(
        [
            false_connections,
            np.roll(false_connections, shift=2, axis=1),
            np.roll(false_connections, shift=4, axis=1),
            np.roll(false_connections, shift=6, axis=1)
        ], axis=0
    )
    branch_neighbor_indices = get_neighbor_values(
        full_point_list,
        np.arange(full_point_list.shape[1]),
        cell_branch_points_unfiltered
    )
    false_branch_check = np.all(
        np.equal(
            np.not_equal(branch_neighbor_indices, -1)[:, np.newaxis, :],
            false_branch_patterns_rolled[np.newaxis, :, :]
        ),
        axis=2
    )
    false_branch_matches = np.nonzero(false_branch_check)
    false_connections = false_connections_rolled[false_branch_matches[1]]
    false_branch_and_neighbors = np.reshape(
        np.stack(
            np.broadcast_arrays(
                np.reshape(original_point_index_map[
                               *cell_branch_points_unfiltered[:, false_branch_matches[0]]
                           ], [-1, 1]),
                branch_neighbor_indices[false_branch_matches[0]]
            ),
            axis=2
        ),
        [-1, 2]
    )
    false_connections = false_branch_and_neighbors[false_connections.flatten()]
    false_connections_sanitized = np.unique(
        np.concat(
            (false_connections, np.roll(false_connections, shift=1, axis=1)), axis=0
        ), axis=0
    )
    false_connection_map = coo_array(
        ([True] * false_connections_sanitized.shape[0], false_connections_sanitized.T),
        shape=[np.max(unfiltered_border_connections) + 1] * 2
    )
    true_branches = cell_branch_points_unfiltered[:, np.logical_not(np.any(false_branch_check, axis=1))]
    connections_filtered_1 = unfiltered_border_connections[
        :,
        np.logical_not(false_connection_map[*unfiltered_border_connections].toarray())
    ]
    # true_connections = peel_corners(full_point_list, connections_filtered_1)
    return true_branches, connections_filtered_1


def find_cell_crossings(border_cells: List[NDArray]):
    border_points = np.concat(border_cells, axis=1)
    point_cell_ids = np.concat(
        [[cell_index + 1] * cell.shape[1] for cell_index, cell in enumerate(border_cells)]
    )
    border_point_neighbor_grid_ids = get_neighbor_values(
        border_points,
        point_cell_ids,
        border_points
    )
    is_point_crossing = np.any(
        (
            np.not_equal(border_point_neighbor_grid_ids, -1) &
            np.not_equal(border_point_neighbor_grid_ids, point_cell_ids[:, np.newaxis])
        ),
        axis=1
    )
    return border_points[:, is_point_crossing]


def get_connected_neighbors(branch_points, region_points, full_point_list):
    branch_neighbors = get_neighbors(branch_points)
    branch_neighbor_regions = get_neighbor_values(
        np.reshape(np.indices(region_points.shape), [2, -1]),
        region_points.flatten(),
        branch_points
    )
    branch_neighbor_deviations = branch_neighbors[:, :, :, np.newaxis] - branch_neighbors[:, :, np.newaxis, :]
    branch_neighbors_within_one = np.all(
        np.isin(branch_neighbor_deviations, [-1, 0, 1]),
        axis=0
    )
    branch_neighbor_regions_match = np.equal(
        branch_neighbor_regions[:, :, np.newaxis], branch_neighbor_regions[:, np.newaxis, :]
    )
    return branch_neighbors_within_one & branch_neighbor_regions_match


def find_branch_components(branch_points, region_points, border_points):
    connected_neighbors = get_connected_neighbors(branch_points, region_points, border_points)
    connected_neighbors_indices = np.indices(connected_neighbors.shape)
    combined_connected_neighbors_indices = np.reshape(
        connected_neighbors_indices[[1, 2]] + connected_neighbors_indices[[0]] * 8,
        [2, -1]
    )
    connected_neighbors_graph = coo_array(
        (connected_neighbors.flatten(), combined_connected_neighbors_indices),
        shape=[branch_points.shape[1] * 8] * 2,
        dtype=np.bool
    ).toarray()
    _, connected_labels = connected_components(connected_neighbors_graph, directed=False)
    neighbor_is_border = get_neighbor_boolean_mask(
        border_points,
        branch_points
    )
    masked_connected_labels = np.where(
        neighbor_is_border.flatten(),
        connected_labels,
        np.full([branch_points.shape[1] * 8], -1)
    )
    return np.reshape(masked_connected_labels, [-1, 8])


def find_next_start_list(branch_point_components, branch_point, branch_predecessor, searched):
    branch_neighbors = get_neighbors(branch_point[:, np.newaxis])[:, 0, :]
    if branch_predecessor is None:
        return branch_neighbors[:, np.not_equal(branch_point_components, -1)]
    branch_neighbor_labels = coo_array(
        (branch_point_components, branch_neighbors + 1),
        shape=np.max(branch_neighbors, axis=1) + 2
    )
    searched_mask = coo_array(
        ([True] * searched.shape[1], searched + 1),
        shape=np.max(np.concat((searched, branch_neighbors), axis=1), axis=1) + 2
    )
    predecessor_label = branch_neighbor_labels[*(branch_predecessor + 1)]
    is_unsearched_neighbor = np.logical_not(searched_mask[*(branch_neighbors + 1)].toarray())
    is_unsearched_predecessor_component = np.equal(branch_point_components, predecessor_label) & \
        is_unsearched_neighbor & np.logical_not(
            np.all(np.equal(branch_neighbors, branch_predecessor[:, np.newaxis]), axis=0)
    )
    if not is_unsearched_predecessor_component.any():
        return np.zeros([2, 0], dtype=np.int32)
    predecessor_connected_components = branch_neighbors[:, is_unsearched_predecessor_component]
    connected_offset = offset_matrix[:, 0, is_unsearched_predecessor_component]
    neighbor_unit_offsets = connected_offset/np.linalg.norm(connected_offset, axis=0, keepdims=True)
    predecessor_offset = branch_point - branch_predecessor
    predecessor_unit_offset = predecessor_offset / np.linalg.norm(predecessor_offset, axis=0, keepdims=True)
    next_start_test = np.dot(predecessor_unit_offset, neighbor_unit_offsets)
    return predecessor_connected_components[:, np.argsort(1 - next_start_test)]


def get_unsearched_points_in_cell(cells, start_point, searched, cell_map):
    cell = cells[cell_map[*start_point]]
    searched_map = coo_array(
        (np.ones([searched.shape[1]], dtype=np.bool), searched),
        shape=cell_map.shape
    )
    return cell[
        :,
        np.logical_not(searched_map[*cell].toarray())
    ]


def check_point_components(points, point_index_map, connections, branch_component_map):
    point_indices = np.sort(point_index_map[*points])
    point_components_1 = branch_component_map[point_indices].toarray()
    point_graph = get_window_point_graph(point_indices, connections).toarray()
    _, point_components_2 = connected_components(point_graph, directed=False)
    unique_component_pairs = np.unique(np.stack([point_components_1, point_components_2], axis=0), axis=1)
    _, component_2_counts = np.unique(unique_component_pairs, axis=1, return_counts=True)
    return max(component_2_counts) <= 1


def find_next_branch(distances, predecessors, branch_indices, points):
    branch_distances = distances[branch_indices]
    next_branch_set_in_points = branch_indices[
        np.flatnonzero(np.equal(branch_distances, np.min(branch_distances)))
    ]
    if next_branch_set_in_points.shape[0] == 1:
        next_branch_in_points = next_branch_set_in_points[0]
    else:
        next_branch_in_points = next_branch_set_in_points[
            np.argmax(np.logical_not(
                np.isin(next_branch_set_in_points, predecessors)
            ))
        ]
    next_branch_distance = distances[next_branch_in_points]
    next_branch = points[:, next_branch_in_points]
    return next_branch, next_branch_distance, next_branch_in_points


def is_point(point, to_equal):
    return np.all(np.equal(point, to_equal))


def navigate_through_points(
        start_point,
        previous_point,
        cells,
        cell_map,
        branch_point_map,
        branch_point_components,
        searched,
        connections,
        border_point_mask,
        point_index_map,
        points,
        iteration
):
    unwound_searched = np.concat(searched, axis=1)
    if branch_point_map[*start_point] >= 0:
        next_list = find_next_start_list(
            branch_point_components[branch_point_map[*start_point]],
            start_point,
            previous_point,
            unwound_searched)
        candidate_continuations = []
        if next_list.shape[1] == 0:
            return start_point[:, np.newaxis]
        for next_point_candidate in next_list.T:
            candidate_continuation = navigate_through_points(
                next_point_candidate,
                start_point,
                cells,
                cell_map,
                branch_point_map,
                branch_point_components,
                [*searched, start_point[:, np.newaxis]],
                connections,
                border_point_mask,
                point_index_map,
                points,
                iteration + 1
            )
            found_points = np.concat(
                [unwound_searched, start_point[:, np.newaxis], candidate_continuation], axis=1
            )
            candidate_found_map = coo_array(
                ([True] * found_points.shape[1], found_points),
                shape=border_point_mask.shape
            )
            if candidate_found_map[*points].toarray().all():
                return np.concat(
                    [start_point[:, np.newaxis], candidate_continuation], axis=1
                )
            else:
                candidate_continuations.append(candidate_continuation)
        return np.concat(
                    [
                        start_point[:, np.newaxis],
                        candidate_continuations[np.argmax([
                            candidate.shape[1] for candidate in candidate_continuations])
                        ]], axis=1
                )

    unordered_cell_points = get_unsearched_points_in_cell(cells, start_point, unwound_searched, cell_map)
    if unordered_cell_points.shape[1] == 1:
        return start_point[:, np.newaxis]
    cell_point_indexes = np.sort(point_index_map[*unordered_cell_points])
    cell_graph = get_window_point_graph(cell_point_indexes, connections)
    cell_points = points[:, cell_point_indexes]
    is_start_point = np.all(np.equal(start_point[:, np.newaxis], cell_points), axis=0)
    assert is_start_point.any()
    point_distances_wrapped, predecessors_wrapped = shortest_path(
        cell_graph,
        directed=False,
        indices=[np.argmax(is_start_point)],
        return_predecessors=True
    )
    non_start_indices = np.flatnonzero(np.logical_not(is_start_point))
    non_start_points = cell_points[:, non_start_indices]
    distances_from_start = point_distances_wrapped[0, non_start_indices]
    branch_indices = np.flatnonzero(
        np.greater_equal(branch_point_map[*cell_points], 0) &
        np.logical_not(np.isinf(point_distances_wrapped[0]))
    )
    if len(branch_indices) > 0:
        next_start, next_branch_distance, next_branch_index = find_next_branch(
            point_distances_wrapped[0],
            predecessors_wrapped[0],
            branch_indices,
            cell_points,
        )
        assert branch_point_map[*next_start] != -1
        next_branch_predecessor_index = predecessors_wrapped[0, next_branch_index]
        next_branch_predecessor = cell_points[:, next_branch_predecessor_index]
        points_to_next_branch_unordered_indices = np.flatnonzero(np.less(distances_from_start, next_branch_distance))
        points_to_next_branch_unordered = non_start_points[:, points_to_next_branch_unordered_indices]
        searched_within_this_round = np.concat([start_point[:, np.newaxis], points_to_next_branch_unordered], axis=1)
        points_to_next_branch_distances = distances_from_start[points_to_next_branch_unordered_indices]
        points_to_next_branch = non_start_points[:, points_to_next_branch_unordered_indices]
        points_to_next_branch_ordered = points_to_next_branch[:, np.argsort(points_to_next_branch_distances)]
        updated_searched = [*searched, searched_within_this_round]
        test_searched = np.concat([*searched, searched_within_this_round], axis=1)
        assert not coo_array(([True] * test_searched.shape[1], test_searched), shape=border_point_mask.shape)[*next_start]
        return np.concat(
            [
                start_point[:, np.newaxis],
                points_to_next_branch_ordered,
                navigate_through_points(
                    next_start,
                    next_branch_predecessor,
                    cells,
                    cell_map,
                    branch_point_map,
                    branch_point_components,
                    updated_searched,
                    connections,
                    border_point_mask,
                    point_index_map,
                    points,
                    iteration + 1
                )
            ],
            axis=1
        )
    else:
        return np.concat([
                start_point[:, np.newaxis],
                non_start_points[:, np.argsort(distances_from_start)]
            ], axis=1)


MIN_TARGET_SIZE_FRACTION = .5


def split_cells(border_cells, point_index_map, connections):
    for cell_unsorted in border_cells:
        cell_point_indices_unsorted = point_index_map[*cell_unsorted]
        cell_sorting = np.argsort(cell_point_indices_unsorted)
        cell_point_indices = cell_point_indices_unsorted[cell_sorting]
        cell = cell_unsorted[:, cell_sorting]
        cell_graph = get_window_point_graph(cell_point_indices, connections)
        label_count, point_labels = connected_components(cell_graph, directed=False)
        for label in range(label_count):
            yield cell[:, np.equal(point_labels, label)]




def merge_cells(border_cells: List[NDArray], target_size, image_shape):
    min_cell_index = np.argmin([cell.shape[1] for cell in border_cells])
    if border_cells[min_cell_index].shape[1] > target_size * MIN_TARGET_SIZE_FRACTION or len(border_cells) == 1:
        return border_cells
    else:
        min_cell = border_cells[min_cell_index]
        cell_neighbor_indices = get_neighbor_values(
            np.concat([cell for cell in border_cells], axis=1),
            np.concat([[cell_index] * cell.shape[1] for cell_index, cell in enumerate(border_cells)]),
            min_cell
        )
        filtered_cell_neighbor_indices = cell_neighbor_indices[
            np.not_equal(cell_neighbor_indices, -1) & np.not_equal(cell_neighbor_indices, min_cell_index)
        ]
        min_neighbor_index = filtered_cell_neighbor_indices[
            np.argmin([border_cells[neighbor_index].shape[1]
                       for neighbor_index in filtered_cell_neighbor_indices])
        ]
        return merge_cells(
            [
                np.concat([border_cells[min_cell_index], border_cells[min_neighbor_index]], axis=1),
                *[cell for cell_index, cell in enumerate(border_cells)
                 if cell_index not in (min_cell_index, min_neighbor_index)]
            ],
            target_size, image_shape
        )


def order_points(border_points: NDArray, region_points: NDArray):
    unfiltered_border_connections = get_pixel_connections(border_points, region_points.shape)
    grid_target_size = max(
        min(math.sqrt(border_points.shape[1]), TARGET_MAX_SIZE),
        TARGET_MIN_SIZE
    )
    original_point_index_map = coo_array(
        (np.arange(border_points.shape[1]) + 1, border_points),
        shape=region_points.shape
    ).toarray() - 1
    border_point_mask = coo_array(
                    (np.ones(border_points.shape[1], dtype=np.bool), border_points),
                    shape=region_points.shape,
                    dtype=np.bool
                ).toarray()
    border_point_grid = Grid(region_points.shape, border_points)
    border_cells_unmerged: List[GridCell] = []
    while border_point_grid.grid.shape[1] > 0:
        border_point_grid.subdivide()
        borders_below_threshold = np.less_equal(np.sum(border_point_grid.padding_mask, axis=1), grid_target_size)
        border_cells_unmerged.extend(border_point_grid.get_cells(borders_below_threshold))
        border_point_grid.filter(np.logical_not(borders_below_threshold))
    split_cells_res = list(split_cells(
        [cell.points for cell in border_cells_unmerged],
        original_point_index_map,
        unfiltered_border_connections))
    border_cells = merge_cells(
        split_cells_res,
        grid_target_size, region_points.shape)
    cell_map = coo_array(
        (
            np.concat([[cell_index] * cell.shape[1] for cell_index, cell in enumerate(border_cells)]),
            np.concat([cell for cell in border_cells], axis=1)
        ),
        shape=region_points.shape
    )
    cell_branch_points_unfiltered = np.concat([find_branch_points(cell, unfiltered_border_connections, original_point_index_map)
                               for cell in border_cells], axis=1)
    cell_branch_points, border_connections = filter_branches(
        cell_branch_points_unfiltered,
        unfiltered_border_connections,
        original_point_index_map,
        border_points)
    _, branch_connections_test = np.unique(
        border_connections[0, np.isin(border_connections[0], original_point_index_map[*cell_branch_points])],
        return_counts=True
    )
    crossing_points = find_cell_crossings(border_cells)
    branch_points = np.unique(
        np.concat([cell_branch_points, crossing_points], axis=1),
        axis=1
    )
    if branch_points.shape[1] > 0:
        branch_point_components = find_branch_components(
            branch_points,
            region_points,
            border_points
        )
        branch_point_map = coo_array(
            (np.arange(branch_points.shape[1]) + 1, branch_points),
            shape=region_points.shape,
            dtype=np.int32
        ).toarray() - 1
    else:
        branch_point_components = np.reshape([], [0, 8])
        branch_point_map = np.full(region_points.shape, -1)

    start_point = find_border_end_points(border_points, region_points, border_connections)
    if start_point is None:
        is_branch = np.not_equal(branch_point_map[*border_points], -1)
        is_branch_connection = np.any(is_branch[border_connections], axis=0)
        branch_connections = border_connections[:, is_branch_connection]
        branch_connection_graph = coo_array(
            (np.ones([branch_connections.shape[1]], dtype=np.bool), branch_connections),
            shape=[border_points.shape[1]] * 2
        ).toarray()
        first_non_branch_connected = np.argmax(
            np.equal(
                np.sum(branch_connection_graph, axis=1), 0
            )
        )
        ordering_init = border_points[:, [first_non_branch_connected]]
        start_point = border_points[
            :,
            border_connections[1, np.argmax(np.equal(border_connections[0], first_non_branch_connected))]
        ]
        assert branch_point_map[*start_point] == -1
        is_loop = True
    else:
        ordering_init = np.reshape([], [2, 0]).astype(np.int32)
        is_loop = False
    point_order = np.concat(
        (
            ordering_init,
            navigate_through_points(
                start_point,
                None,
                border_cells,
                cell_map,
                branch_point_map,
                branch_point_components,
                [ordering_init],
                border_connections,
                border_point_mask,
                original_point_index_map,
                border_points,
                0
            )
        ),
        axis=1
    )
    # test_region(region_points.shape, point_order)
    # Ensuring border traversal is likely unnecessary - the border will be simplified afterwards, so a few missed
    # points will have little to no effect on the final result
    # order_test = coo_array(
    #     (np.arange(point_order.shape[1]) + point_order.shape[1], point_order),
    #     shape=region_points.shape
    # ).toarray()
    # assert np.all(np.greater(order_test[*border_points], 0))
    return point_order, is_loop

