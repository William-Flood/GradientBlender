import numpy as np
import math
from numpy.typing import NDArray
from scipy.sparse import coo_array
from scipy.sparse.csgraph import shortest_path, connected_components

from borders.get_pixel_connections import get_pixel_connections
from borders.border import Border
from util.inspection_utils import *

INITIAL_WINDOW_SIZE = 200


def find_graph_distances_from_search_to_selected(selected_points, search_point, connections):
    selection_search_point_index = np.searchsorted(selected_points, search_point)
    selection_points_with_search = np.insert(selected_points, selection_search_point_index, search_point)
    point_connection_graph = get_window_point_graph(selection_points_with_search, connections)
    graph_distances = shortest_path(
        point_connection_graph, directed=False, indices=[selection_search_point_index])[0]
    assert not np.isinf(graph_distances).any()
    return np.delete(graph_distances.astype(np.int32), selection_search_point_index)


def find_offset_window(pixel_indices_map, search_point, search_vector, target_size, sample_bounds):
    base_area = np.abs(np.prod(search_vector))
    if base_area == 0:
        scale = target_size
    else:
        scale = math.sqrt(target_size / base_area)
    unbounded_window_vector = np.clip(
        search_vector * scale,
        [1, 1],
        [target_size, target_size]
    ).astype(np.int32)
    window_vector = np.clip(unbounded_window_vector + search_point,
                            [0, 0],
                            [
                                    pixel_indices_map.shape[0],
                                    pixel_indices_map.shape[1]
                                   ]) - search_point
    window_points = np.sort([
        [search_point[0], search_point[0] + window_vector[0]],
        [search_point[1], search_point[1] + window_vector[1]]
    ])
    window_and_sample = np.stack([window_points, sample_bounds], axis=0)
    sample_adjusted_window = np.stack(
        (
            np.min(
                window_and_sample[:, :, 0], axis=0
            ),
            np.max(
                window_and_sample[:, :, 1], axis=0
            )
        ),
        axis=1
    )
    return sample_adjusted_window


def get_window_point_graph(points_in_window, connections):
    are_connections_within_window = np.all(np.isin(connections, points_in_window), axis=0)
    connection_within_window = connections[:, are_connections_within_window]
    connections_window_indices: NDArray = np.searchsorted(points_in_window, connection_within_window)
    return coo_array(
        (np.ones([connections_window_indices.shape[1]]), connections_window_indices),
        shape=[len(points_in_window)] * 2
    )


def test_points(points_in_window_coords, search_point, angle_cutoff, test_vector):
    """
    Identifies points outside the region defined by the search point, angle cutoff, and test vector.  Points meeting
    this criteria can be navigated to, allowing the algorithm to skip over sections of low curvature.
    Args:
        points_in_window_coords:
        search_point:
        angle_cutoff:
        test_vector:

    Returns:

    """
    displacements_from_search = points_in_window_coords - search_point[:, np.newaxis]
    unit_vectors_from_search = (displacements_from_search
                                /
                                np.linalg.norm(displacements_from_search, axis=0, keepdims=True))
    dots_to_average = np.dot(test_vector, unit_vectors_from_search)
    return np.less(dots_to_average, math.cos(angle_cutoff))


def filter_points_by_direction(points_to_filter, direction_start, test_direction_vector, distances):
    """
    Returns a set of indices identifying which points in points_to_filter best follow a test vector from
    direction_start, excluding points of equal connection distance to direction_start that deviate from the test vector
    more than those that were kept.  When applied to the list of indices used to generate points_to_filter from the
    global points list, the result is the indices in the global list that best match the direction filter
    """
    vectors_from_sample = points_to_filter - direction_start[:, np.newaxis]
    unit_vectors_from_sample = vectors_from_sample / np.linalg.norm(vectors_from_sample, axis=0)
    direction_losses = 1 - np.dot(test_direction_vector, unit_vectors_from_sample)
    direction_loss_orderings = np.argsort(direction_losses)
    direction_loss_inverse_orderings = np.argsort(direction_loss_orderings)
    direction_loss_ordered_distances = distances[direction_loss_orderings]
    _, lowest_loss_distances = np.unique(direction_loss_ordered_distances, return_index=True)
    return np.sort(direction_loss_inverse_orderings[lowest_loss_distances])


def filter_component_points_by_direction(
        points_in_window_in_search_component,
        search_point_index,
        connections,
        points,
        test_direction_vector
):
    window_component_search_point = np.searchsorted(points_in_window_in_search_component, search_point_index)
    component_connection_graph = get_window_point_graph(points_in_window_in_search_component, connections)
    graph_distances = shortest_path(
        component_connection_graph,
        directed=False,
        indices=[window_component_search_point])[0].astype(np.int32)
    points_in_window_connected_to_search = np.delete(
        points_in_window_in_search_component,
        window_component_search_point
    )
    graph_distances_from_search = np.delete(
        graph_distances,
        window_component_search_point
    )
    direction_filter = filter_points_by_direction(
        points[:, points_in_window_connected_to_search],
        points[:, search_point_index],
        test_direction_vector,
        graph_distances_from_search)

    return points_in_window_connected_to_search[direction_filter]


def component_defuzz_filter(
        direction_filtered_points,
        search_point_index,
        points,
        test_direction_vector,
        connections
):
    """
    Handles the case of the test vector pointing away from the longer half of the boundary in the window - in which case,
    filter_component_points_by_direction will fail to filter out all of those points, producing a discontinuous border
    segment.  This function returns a boolean mask indicating which points lie in the component closest to the direction
    of the test vector
    Args:
        direction_filtered_points:
        search_point_index:
        points:
        test_direction_vector:
        connections:

    Returns:

    """
    search_insertion_index = np.searchsorted(direction_filtered_points, search_point_index)
    direction_filtered_points_and_search = np.insert(
        direction_filtered_points,
        search_insertion_index,
        search_point_index
    )
    window_connection_graph = get_window_point_graph(direction_filtered_points_and_search, connections)
    components_count, labels = connected_components(window_connection_graph)
    if components_count == 1:
        return [True] * direction_filtered_points.shape[0]
    else:
        search_component_label = labels[search_insertion_index]
        labels_sans_search = np.delete(labels, search_insertion_index)
        return np.equal(labels_sans_search, search_component_label)


def filter_points(points_in_window, connections, search_point_index, searched, points, test_direction_vector):
    assert not searched[search_point_index]
    fresh_points_in_window = np.sort(points_in_window[np.logical_not(searched[points_in_window])])
    window_connection_graph = get_window_point_graph(fresh_points_in_window, connections)
    window_search_point = np.searchsorted(fresh_points_in_window, search_point_index)
    _, labels = connected_components(window_connection_graph)
    search_point_label = labels[window_search_point]
    search_point_component = np.flatnonzero(np.equal(labels, search_point_label))
    points_in_window_in_search_component = fresh_points_in_window[search_point_component]
    direction_filtered_points = filter_component_points_by_direction(
        points_in_window_in_search_component,
        search_point_index,
        connections,
        points,
        test_direction_vector
    )
    is_in_valid_component = component_defuzz_filter(
        direction_filtered_points,
        search_point_index,
        points,
        test_direction_vector,
        connections)
    return direction_filtered_points[is_in_valid_component]



def get_valid_points_in_window(
        window_points,
        pixel_indices,
        search_point_index,
        connections,
        searched,
        points,
        test_direction_vector
):
    window = pixel_indices[
        window_points[0, 0]:window_points[0, 1],
        window_points[1, 0]:window_points[1, 1]
    ]
    points_in_window = window[*np.nonzero(
        np.not_equal(window, -1)
    )]
    return filter_points(points_in_window, connections, search_point_index, searched, points, test_direction_vector)


def find_points_around_target(
        search_point,
        search_point_index,
        take_size,
        pixel_indices,
        connections,
        searched,
        points,
        test_direction_vector):
    window_points = np.clip([
            [search_point[0] - take_size, search_point[0] + take_size],
            [search_point[1] - take_size, search_point[1] + take_size]
        ],
        [[0, 0], [0, 0]],
        [
           [pixel_indices.shape[0], pixel_indices.shape[0]],
           [pixel_indices.shape[1], pixel_indices.shape[1]]
        ]
    )
    points_in_window = get_valid_points_in_window(
        window_points,
        pixel_indices,
        search_point_index,
        connections,
        searched,
        points,
        test_direction_vector
    )
    if points_in_window.shape[0] == 0:
        return []
    graph_distances = find_graph_distances_from_search_to_selected(
        points_in_window, search_point_index, connections
    )
    node_is_in_distance = np.less_equal(graph_distances, take_size)
    point_indices = points_in_window[node_is_in_distance]
    return point_indices


def find_search_vector(
        points,
        connections,
        point_index_map,
        search_point_index,
        take_size,
        searched,
        momentum_vector
):
    search_point = points[:, search_point_index]
    points_in_sample = find_points_around_target(
        search_point,
        search_point_index,
        take_size,
        point_index_map,
        connections,
        searched,
        points,
        momentum_vector
    )
    if len(points_in_sample) == 0:
        return None, None, False
    sample_bounds = np.stack([
        np.min(points[:, points_in_sample], axis=1),
        np.max(points[:, points_in_sample], axis=1)
    ], axis=1)
    vectors_from_sample = points[:, points_in_sample] - search_point[:, np.newaxis]
    unit_vectors_from_sample = vectors_from_sample / np.linalg.norm(vectors_from_sample, axis=0)
    return np.average(unit_vectors_from_sample, axis=1), sample_bounds, True


def find_furthest(window_points, search_point_index, connections):
    window_search_location = np.searchsorted(window_points, search_point_index)
    point_coords_with_search = np.insert(window_points,
                                        window_search_location,
                                        search_point_index
                                        )
    points_graph = get_window_point_graph(point_coords_with_search, connections)
    distances = shortest_path(points_graph, indices=[window_search_location])[0]
    distances_sans_search = np.delete(distances, window_search_location)
    return window_points[np.argmax(distances_sans_search)]



def get_updated_sample_bounds(
        points,
        next_to_connect,
        take_size,
        pixel_indices_map,
        connections,
        searched,
        search_vector,
        sample_bounds,
        search_point_index
):
    """
    Provides an expanded search bounds if no return point could be found within a loop iteration of find_next_point,
    ensuring the window in the next iteration includes a new sample of points by handling the edge case of the border
    curving away from the window.  The function additionally returns a flag indicating whether new a new sample
    of points to check even exists, ensuring find_next_point halts once it exhausts the points in points reachable
    from the start point in navigate_from_point.
    Args:
        points:
        next_to_connect:
        take_size:
        pixel_indices_map:
        connections:
        searched:
        search_vector:
        sample_bounds:
        search_point_index:

    Returns:

    """
    points_around_next = find_points_around_target(
        points[:, next_to_connect],
        next_to_connect,
        take_size,
        pixel_indices_map,
        connections,
        searched,
        points,
        search_vector
    )
    if np.not_equal(points_around_next, search_point_index).any():
        points_around_next_sans_search = points_around_next[np.not_equal(points_around_next, search_point_index)]
        sample_bounds_around_next = np.stack([
            np.min(points[:, points_around_next_sans_search], axis=1),
            np.max(points[:, points_around_next_sans_search], axis=1) + 1
        ], axis=1)
        window_and_sample = np.stack([sample_bounds, sample_bounds_around_next], axis=0)
        new_search_bounds = np.stack(
            (
                np.min(
                    window_and_sample[:, :, 0], axis=0
                ),
                np.max(
                    window_and_sample[:, :, 1], axis=0
                )
            ),
            axis=1
        )
        return new_search_bounds, False
    else:
        return None, True


def find_next_point(
        points,
        connections,
        pixel_indices_map,
        searched,
        search_point_index,
        angle_cutoff,
        direction_sample,
        momentum_vector
    ):
    window_scale_factor = 1
    search_vector, sample_bounds, sample_exists = find_search_vector(
        points,
        connections,
        pixel_indices_map,
        search_point_index,
        direction_sample,
        searched,
        momentum_vector
    )
    if not sample_exists:
        searched[search_point_index] = True
        return None, False
    search_point = points[:, search_point_index]
    while True:
        window_points = find_offset_window(
            pixel_indices_map,
            search_point,
            search_vector,
            INITIAL_WINDOW_SIZE * window_scale_factor,
            sample_bounds
        )
        points_in_window = get_valid_points_in_window(
            window_points,
            pixel_indices_map,
            search_point_index,
            connections,
            searched,
            points,
            search_vector
        )
        # Should be guaranteed by the sample_exists check
        assert len(points_in_window) > 0
        test_results = test_points(
            points[:, points_in_window],
            search_point,
            angle_cutoff,
            search_vector
        )
        if test_results.any():
            points_outside_direction_threshold = points_in_window[test_results]
            distances_from_search = np.linalg.norm(
                points[:, points_outside_direction_threshold] - search_point[:, np.newaxis], axis=0
            )
            next_point = points_outside_direction_threshold[np.argmin(distances_from_search)]
            searched[points_in_window[np.logical_not(test_results)]] = True
            searched[search_point_index] = True
            return next_point, True
        else:
            next_to_connect = find_furthest(
                points_in_window,
                search_point_index,
                connections
            )
            searched[points_in_window[np.not_equal(points_in_window, next_to_connect)]] = True
            sample_bounds, should_end_search = get_updated_sample_bounds(
                points,
                next_to_connect,
                direction_sample,
                pixel_indices_map,
                connections,
                searched,
                search_vector,
                sample_bounds,
                search_point_index
            )
            if should_end_search:
                searched[search_point_index] = True
                return next_to_connect, False
            connections = np.concat(
                ([[search_point_index, next_to_connect], [next_to_connect, search_point_index]], connections),
                axis=1
            )
        window_scale_factor += 1


def navigate_from_point(
        points,
        connections,
        pixel_indices,
        searched,
        start_point,
        decimate_angle_cutoff,
        direction_sample,
        momentum_vector):
    still_navigating = True
    search_point = start_point
    point_list = []
    while still_navigating:
        next_point, still_navigating = find_next_point(
            points,
            connections,
            pixel_indices,
            searched,
            search_point,
            decimate_angle_cutoff,
            direction_sample,
            momentum_vector
        )
        if still_navigating:
            point_list.append(next_point)
            next_to_last_displacement = points[:, next_point] - points[:, search_point]
            momentum_vector = next_to_last_displacement / np.linalg.norm(next_to_last_displacement, axis=0)
            search_point = next_point
        elif not searched.all():
            point_list.append(next_point)
    return point_list



def decimate(border: Border, image_shape, decimate_angle_cutoff, direction_sample):
    """
    Identifies regions of low curvature within a border and replaces them with line segments.
    Requires the border to be continuous - assured by the divide function that originally creates the
    border objects
    Args:
        border:
        image_shape:
        decimate_angle_cutoff:
        direction_sample:

    Returns:

    """
    border_connections = get_pixel_connections(border.full_points, image_shape)
    pixel_indices = np.full(image_shape, -1, dtype=np.int32)
    pixel_indices[*border.full_points] = np.arange(border.full_points.shape[1])
    searched = np.zeros([border.full_points.shape[1]], dtype=np.bool)
    forward_list = navigate_from_point(
        border.full_points,
        border_connections,
        pixel_indices,
        searched,
        0,
        decimate_angle_cutoff,
        direction_sample,
        np.full([2], math.sqrt(2) / 2)
    )
    if not searched.all():
        searched[0] = False
        reverse_list = navigate_from_point(
            border.full_points,
            border_connections,
            pixel_indices,
            searched,
            0,
            decimate_angle_cutoff,
            direction_sample,
            np.full([2], -math.sqrt(2) / 2)
        )
        border.decomposed_points = border.full_points[
            :,
            [
                *reversed(reverse_list),
                0,
                *forward_list
            ]
        ]
        border.is_loop = False
    else:
        border.decomposed_points = border.full_points[:, [0, *forward_list]]
        border.is_loop = True
