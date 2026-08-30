import numpy as np
from numpy.typing import NDArray
from util.grid import Grid, GridCell
from scipy.sparse import coo_array
from typing import List, Generator
from scipy.sparse.csgraph import shortest_path, connected_components
from util.inspection_utils import *
from util.point_neighborhood import (
    offset_matrix,
    get_neighbor_values,
    get_neighbors,
    get_neighbor_boolean_mask,
    get_neighbor_values_over_array
)
from util.inspection_utils_gui import visualize_point_list

TARGET_MAX_SIZE = 500

def get_point_list_graph(points_in_list, connections):
    """
    Constructs a sparse array representing the graph connections of a list of points, generally selected out of some
    feature of an image for batch processing.
    Args:
        points_in_list: A sorted list of indices corresponding to points in a matrix.
        connections: The full list of connections between points in the matrix.

    Returns:

    """
    if points_in_list.shape[0] == 0:
        return coo_array(
            ([], np.zeros([2, 0])),
            shape=[0] * 2)
    elif points_in_list.shape[0] == 1:
        return coo_array(
            ([], np.zeros([2, 0])),
            shape=[1] * 2)
    assert np.all(np.greater(points_in_list[1:], points_in_list[:-1]))
    are_points_within_window = coo_array(
        (np.ones(points_in_list.shape[0], dtype=np.bool), [points_in_list]),
        shape=[np.max(connections) + 1]
    )
    are_connections_within_window = np.all(
        are_points_within_window[connections].toarray(),
        axis=0
    )
    connection_within_window = connections[:, are_connections_within_window]
    connections_window_indices: NDArray = np.searchsorted(points_in_list, connection_within_window)
    return coo_array(
        (np.ones([connections_window_indices.shape[1]]), connections_window_indices),
        shape=[len(points_in_list)] * 2
    )


def find_border_start_point(border_points, region_points, connections):
    """
    Returns the starting point for traversing a border
    Args:
        border_points: The list of points in the border
        region_points: A map of which points in the full image correspond to which region
        connections: The full list of connections between points in the matrix.

    Returns: If the border is a closed loop, returns None.  Otherwise, the function returns
    one of the border end points

    """
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
            multi_region_border_indices,
            multi_region_border_neighbor_indices
    ):
        neighbor_indices_normalized = np.sort(neighbor_indices[(np.not_equal(neighbor_indices, -1))])
        get_point_list_graph(neighbor_indices_normalized, connections)
        num_labels, _ = connected_components(
            get_point_list_graph(neighbor_indices_normalized, connections).toarray(),
            directed=False
        )
        if num_labels == 1:
            end_point_indices_list.append(multi_region_index)
    if len(end_point_indices_list) > 0:
        end_point_indices = np.array(end_point_indices_list)
    else:
        end_point_indices = np.zeros([0])
    if end_point_indices.shape[0] > 1:
        border_graph = get_point_list_graph(np.arange(border_points.shape[1]), connections)
        point_distances = shortest_path(border_graph, directed=False, indices=end_point_indices)
        max_distances = np.max(point_distances, axis=1)
        return border_points[:, end_point_indices[np.argmax(max_distances)]]
    elif end_point_indices.shape[0]:
        return border_points[:, end_point_indices[0]]
    else:
        return None


def find_cell_crossings(border_cells: List[NDArray]):
    """
    Given a list of points representing chunks of a region border, identifies which points transition between chunks
    Args:
        border_cells:

    Returns:

    """
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


def get_connected_neighbors(branch_points, region_points):
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
    connected_neighbors = get_connected_neighbors(branch_points, region_points)
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


def find_exterior_connecting_branch_and_previous(branch_predecessor, exterior_points):
    exterior_point_deviation_from_prececessor = exterior_points - branch_predecessor[:, np.newaxis]
    is_connecting_point = np.all(
        np.isin(exterior_point_deviation_from_prececessor, [-1, 0, 1]),
        axis=0
    )
    assert is_connecting_point.any()
    return exterior_points[:, np.argmax(is_connecting_point)]


def traverse_exterior_to_next(branch_point, branch_predecessor, region_points, predecessor_connected_component):
    neighboring_points = get_neighbors(branch_point[:, np.newaxis])
    neighboring_regions = get_neighbor_values_over_array(region_points, branch_point[:, np.newaxis])
    exterior_indices = np.flatnonzero(np.not_equal(neighboring_regions, region_points[*branch_point]))
    exterior_points = neighboring_points[:, 0, exterior_indices]
    exterior_self_connections = np.array(
        np.nonzero(
            np.all(
                np.isin(exterior_points[:, :, np.newaxis] - exterior_points[:, np.newaxis, :], [-1, 0, 1]),
                axis=0
            ) & np.any(
                np.isin(exterior_points[:, :, np.newaxis] - exterior_points[:, np.newaxis, :], [-1, 1]),
                axis=0
            )
        )
    )
    exterior_to_candidate_point_deviations = exterior_points[:, :, np.newaxis] - \
                                             predecessor_connected_component[:, np.newaxis, :]
    exterior_connections_to_predecessor_connected_components = np.array(
        np.nonzero(
            np.all(
                np.isin(exterior_to_candidate_point_deviations, [-1, 0, 1]), axis=0
            ) & np.any(
                np.equal(exterior_to_candidate_point_deviations, 0), axis=0
            )
        )
    )
    full_connections = np.concat([
        exterior_self_connections,
        exterior_connections_to_predecessor_connected_components + np.array([[0], [exterior_points.shape[1]]])
    ], axis=1)
    connections_graph = coo_array(
        ([True] * full_connections.shape[1], full_connections),
        shape=[exterior_points.shape[1] + predecessor_connected_component.shape[1]] * 2
    ).toarray()
    branch_prececessor_joining_exterior_point = find_exterior_connecting_branch_and_previous(
        branch_predecessor,
        exterior_points
    )
    branch_prececessor_joining_exterior_point_index = np.argmax(
        np.all(
            np.equal(exterior_points, branch_prececessor_joining_exterior_point[:, np.newaxis])
        )
    )
    shortest_path_from_connected_exterior = shortest_path(
        connections_graph,
        directed=False,
        indices=[branch_prececessor_joining_exterior_point_index]
    )[0]
    predecessor_connected_component_shortest_path = shortest_path_from_connected_exterior[exterior_points.shape[1]:]
    return predecessor_connected_component[:, np.argmin(predecessor_connected_component_shortest_path)]


def find_next_start_list(branch_point_components, branch_point, branch_predecessor, searched, region_points):
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
    predecessor_connected_component = branch_neighbors[:, is_unsearched_predecessor_component]
    connected_offset = offset_matrix[:, 0, is_unsearched_predecessor_component]
    neighbor_unit_offsets = connected_offset/np.linalg.norm(connected_offset, axis=0, keepdims=True)
    predecessor_offset = branch_point - branch_predecessor
    predecessor_unit_offset = predecessor_offset / np.linalg.norm(predecessor_offset, axis=0, keepdims=True)
    next_start_test = np.dot(predecessor_unit_offset, neighbor_unit_offsets)
    return predecessor_connected_component[:, np.argsort(1 - next_start_test)]


def find_next_start(branch_point_components, branch_point, branch_predecessor, searched, region_points):
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
        return None
    predecessor_connected_component = branch_neighbors[:, is_unsearched_predecessor_component]
    if predecessor_connected_component.shape[1] == 1:
        return predecessor_connected_component[:, 0]
    else:
        return traverse_exterior_to_next(branch_point, branch_predecessor, region_points, predecessor_connected_component)


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
    point_graph = get_point_list_graph(point_indices, connections).toarray()
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


def test_candidate_branches_with_full_exclusion(
        branch_point,
        next_list,
        cells,
        cell_map,
        branch_point_map,
        branch_point_components,
        searched,
        connections,
        border_point_mask,
        point_index_map,
        points,
        iteration,
        region_points) -> Generator[NDArray, None, None]:
    for candidate_index, next_point_candidate in enumerate(next_list.T):
        exclusion_list = np.stack(
            [
                branch_point,
                *[exclusion_point for exclusion_index, exclusion_point in enumerate(next_list.T) if exclusion_index != candidate_index]
            ],
            axis=1
        )
        candidate_branch = navigate_through_points(
            next_point_candidate,
            branch_point,
            cells,
            cell_map,
            branch_point_map,
            branch_point_components,
            [*searched, exclusion_list],
            connections,
            border_point_mask,
            point_index_map,
            points,
            iteration + 1,
            region_points
        )
        yield candidate_branch


def navigate_from_branch(
        branch_point,
        next_list,
        cells,
        cell_map,
        branch_point_map,
        branch_point_components,
        searched,
        connections,
        border_point_mask,
        point_index_map,
        points,
        iteration,
        region_points):
    candidate_branches = list(test_candidate_branches_with_full_exclusion(
        branch_point,
        next_list,
        cells,
        cell_map,
        branch_point_map,
        branch_point_components,
        searched,
        connections,
        border_point_mask,
        point_index_map,
        points,
        iteration,
        region_points
    ))
    longest_candidate = np.argmax([branch.shape[1] for branch in candidate_branches])
    path_from_longest_continuation = candidate_branches[longest_candidate][:, 1:]
    other_branches = np.stack([
        candidate_branch for candidate_index, candidate_branch in enumerate(next_list.T)
        if candidate_index != longest_candidate
    ], axis=0)
    other_branch_paths = [
        navigate_through_points(
            next_point_candidate,
            branch_point,
            cells,
            cell_map,
            branch_point_map,
            branch_point_components,
            [*searched, path_from_longest_continuation],
            connections,
            border_point_mask,
            point_index_map,
            points,
            iteration + 1,
            region_points
        )
        for next_point_candidate in other_branches
    ]
    other_branch_paths_continuation_occurrence = np.array([
        (np.argmax(
            np.all(np.equal(branch_path, next_list[:, [longest_candidate]]), axis=1)
        ), branch_index) for branch_index, branch_path in enumerate(other_branch_paths)
        if np.any(np.all(np.equal(branch_path, next_list[:, [longest_candidate]]), axis=1))
    ])
    if other_branch_paths_continuation_occurrence.shape[0] > 0:
        max_path_index_and_distance = other_branch_paths_continuation_occurrence[
            np.argmax(other_branch_paths_continuation_occurrence[:, 0])
        ]
        branch_to_continuation = other_branch_paths[max_path_index_and_distance[1]]
        return np.concat([
            branch_point[:, np.newaxis],
            branch_to_continuation[:, :max_path_index_and_distance[0] + 1],
            path_from_longest_continuation
        ], axis=1)
    else:
        return np.concat([
            branch_point[:, np.newaxis],
            candidate_branches[longest_candidate]
        ], axis=1)



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
        iteration,
        region_points
) -> NDArray:
    unwound_searched = np.concat(searched, axis=1)
    if branch_point_map[*start_point] >= 0:
        # next_list = find_next_start_list(
        #     branch_point_components[branch_point_map[*start_point]],
        #     start_point,
        #     previous_point,
        #     unwound_searched,
        #     region_points
        # )
        # if next_list.shape[1] == 0:
        #     return start_point[:, np.newaxis]
        # elif next_list.shape[1] == 1:
        #     return np.concat([
        #         start_point[:, np.newaxis],
        #         navigate_through_points(
        #             next_list[:, 0],
        #             start_point,
        #             cells,
        #             cell_map,
        #             branch_point_map,
        #             branch_point_components,
        #             [*searched, start_point[:, np.newaxis]],
        #             connections,
        #             border_point_mask,
        #             point_index_map,
        #             points,
        #             iteration + 1,
        #             region_points
        #         )
        #     ], axis=1)
        # else:
        #     return navigate_from_branch(
        #         start_point,
        #         next_list,
        #         cells,
        #         cell_map,
        #         branch_point_map,
        #         branch_point_components,
        #         searched,
        #         connections,
        #         border_point_mask,
        #         point_index_map,
        #         points,
        #         iteration,
        #         region_points
        #     )
        next_start = find_next_start(
            branch_point_components[branch_point_map[*start_point]],
            start_point,
            previous_point,
            unwound_searched,
            region_points
        )
        if next_start is None:
            return start_point[:, np.newaxis]
        else:
            return np.concat([
                start_point[:, np.newaxis],
                navigate_through_points(
                    next_start,
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
                    iteration + 1,
                    region_points
                )
            ], axis=1)

    unordered_cell_points = get_unsearched_points_in_cell(cells, start_point, unwound_searched, cell_map)
    if unordered_cell_points.shape[1] == 1:
        return start_point[:, np.newaxis]
    cell_point_indexes = np.sort(point_index_map[*unordered_cell_points])
    cell_graph = get_point_list_graph(cell_point_indexes, connections)
    cell_points = points[:, cell_point_indexes]
    is_start_point = np.all(np.equal(start_point[:, np.newaxis], cell_points), axis=0)
    assert is_start_point.any()
    point_distances_wrapped, predecessors_wrapped = shortest_path(
        cell_graph,
        directed=False,
        indices=[np.argmax(is_start_point)],
        return_predecessors=True
    )
    non_start_indices = np.flatnonzero(np.logical_not(is_start_point | np.isinf(point_distances_wrapped[0])))
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
                    iteration + 1,
                    region_points
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
        cell_graph = get_point_list_graph(cell_point_indices, connections)
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


def order_points(border_points: NDArray, region_points: NDArray, border_connections):
    unique_point_indices, index_counts = np.unique(border_connections[0], return_counts=True)
    branch_points = border_points[
        :,
        unique_point_indices[np.greater(index_counts, 2)]
    ]
    grid_target_size = TARGET_MAX_SIZE
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
        border_connections))
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
    _, branch_connections_test = np.unique(
        border_connections[0, np.isin(border_connections[0], original_point_index_map[*branch_points])],
        return_counts=True
    )
    crossing_points = find_cell_crossings(border_cells)
    navigation_point = np.unique(
        np.concat([branch_points, crossing_points], axis=1),
        axis=1
    )
    if navigation_point.shape[1] > 0:
        branch_point_components = find_branch_components(
            navigation_point,
            region_points,
            border_points
        )
        branch_point_map = coo_array(
            (np.arange(navigation_point.shape[1]) + 1, navigation_point),
            shape=region_points.shape,
            dtype=np.int32
        ).toarray() - 1
    else:
        branch_point_components = np.reshape([], [0, 8])
        branch_point_map = np.full(region_points.shape, -1)

    start_point = find_border_start_point(border_points, region_points, border_connections)
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
                0,
                region_points
            )
        ),
        axis=1
    )
    # test_region(region_points.shape, point_order)
    # Ensuring border traversal is likely unnecessary - the border will be simplified afterwards, so a few missed
    # points will have little to no effect on the final result
    order_test = coo_array(
        (np.arange(point_order.shape[1]) + point_order.shape[1], point_order),
        shape=region_points.shape
    ).toarray()
    # assert np.all(np.greater(order_test[*border_points], 0))
    return point_order, is_loop

