import math
import numpy as np
from numpy.typing import NDArray
from borders.border import Border
from scipy.sparse.csgraph import breadth_first_order, shortest_path
from scipy.sparse import coo_array
from scipy.stats import linregress
from util.inspection_utils import test_region
from borders.get_pixel_connections import get_pixel_connections


def get_point_ordering(connection_graph, points):
    adjusted_points = points - np.min(points, axis=1, keepdims=True)
    adjusted_window = np.max(points, axis=1) - np.min(points, axis=1) + 1
    first_point_finder = breadth_first_order(connection_graph, 0, return_predecessors=False)
    first_endpoint_index = first_point_finder[-1]
    distances_from_first = shortest_path(connection_graph, indices=first_endpoint_index)
    # An assignment array which, when applied to points, produces the correct ordering of points in the border curve
    # (if the border does not contain any loops)
    point_sorting_over_first = np.argsort(distances_from_first)
    first_point_sorting_inspection = np.zeros(adjusted_window) - adjusted_points.shape[1]
    first_point_sorting_inspection[*adjusted_points] = point_sorting_over_first
    pivot_index = int(point_sorting_over_first[connection_graph.shape[0] // 2])
    pivot_distances = shortest_path(connection_graph, indices=pivot_index)
    pivot_distances_inspection = np.zeros(adjusted_window) - adjusted_points.shape[1]
    pivot_distances_inspection[*adjusted_points] = pivot_distances + adjusted_points.shape[1]
    first_endpoint_pivot_predecessors = pivot_distances[first_endpoint_index]
    pivot_equivalent_predecessors = pivot_distances[distances_from_first == distances_from_first[pivot_index]]
    loop_found = first_endpoint_pivot_predecessors < np.max(pivot_equivalent_predecessors)
    if loop_found:
        pivot_to_first = pivot_distances < pivot_distances[first_endpoint_index]
        antifirst_point = np.argmax(distances_from_first)
        pivot_to_antifirst = pivot_distances < pivot_distances[antifirst_point]
        # Indicates the location the point at each index will be at after point_sorting_over_first is applied
        point_order_over_first = np.argsort(point_sorting_over_first)
        first_to_pivot = point_order_over_first < point_order_over_first[pivot_index]
        # test_region(adjusted_window, adjusted_points[:, first_to_pivot])
        antifirst_to_pivot_arc = np.logical_and(pivot_to_antifirst, np.logical_not(first_to_pivot))
        pivot_to_first_arc = np.logical_and(pivot_to_first, first_to_pivot)
        pivot_side_assignment = np.logical_or(antifirst_to_pivot_arc, pivot_to_first_arc)
        adjusted_distances = np.copy(distances_from_first)
        pivot_side_distances = distances_from_first[pivot_side_assignment]
        pivot_side_inverse_sort = np.argsort(pivot_side_distances)
        pivot_side_inverse_ordering = np.argsort(pivot_side_inverse_sort)
        pivot_side_ordering = np.sum(pivot_side_assignment) - pivot_side_inverse_ordering - 1
        adjusted_distances[pivot_side_assignment] = pivot_side_ordering + np.sum(np.logical_not(pivot_side_assignment))
        return np.argsort(adjusted_distances), True
    else:
        return point_sorting_over_first, False


def find_longest_line_within_r_2_threshold(point_list, start, end, r2_threshold, length_threshold):
    if start == end:
        return start
    elif end <= length_threshold:
        return end
    else:
        check_point = (start + end) // 2
        if np.min(point_list[1, :check_point]) == np.max(point_list[1, :check_point]):
            check_value = 1
        else:
            check_value = linregress(point_list[1, :check_point], point_list[0, :check_point]).rvalue ** 2
        if check_value < r2_threshold:
            return find_longest_line_within_r_2_threshold(point_list, start, check_point, r2_threshold, length_threshold)
        elif check_value > r2_threshold:
            if check_point == start:
                return check_point
            else:
                return find_longest_line_within_r_2_threshold(point_list, check_point, end, r2_threshold, length_threshold)
        else:
            return check_point


def find_point_past_angle_cutoff(point_list, initial_direction_sample, angle_cutoff):
    zeroed_point_list = point_list - point_list[:, [0]]
    distances_from_start = np.linalg.norm(zeroed_point_list, axis=0, keepdims=True)
    MIN_DISTANCE = 3
    distance_is_valid = np.greater_equal(distances_from_start, MIN_DISTANCE)
    if distance_is_valid.any():
        valid_points = np.flatnonzero(distance_is_valid)
        unit_vectors_from_start = zeroed_point_list[:, valid_points] / distances_from_start[0, valid_points]
        sample_average_deviation = np.average(unit_vectors_from_start[:, :initial_direction_sample], axis=1)
        dots_to_average = np.dot(sample_average_deviation, unit_vectors_from_start)
        points_outside_cutoff = np.less(dots_to_average, math.cos(angle_cutoff))
        return valid_points[np.argmax(points_outside_cutoff)]
    else:
        return point_list.shape[1]


def decimate_list(
        point_list,
        decimate_angle_cutoff,
        decimate_pixel_count_threshold
):
    if point_list.shape[1] > decimate_pixel_count_threshold:
        next_point = find_point_past_angle_cutoff(point_list, decimate_pixel_count_threshold, decimate_angle_cutoff)
        if next_point >= point_list.shape[1] - 1:
            return [point_list[:, 0], point_list[:, -1]]
        else:
            return [point_list[:, 0], *decimate_list(
                point_list[:, next_point:],
                decimate_angle_cutoff,
                decimate_pixel_count_threshold
            )]
    else:
        return [point_list[:, 0], point_list[:, -1]]


def sanitize(ordered_points, ordering, connection_graph, is_loop):
    if is_loop:
        displacements = ordered_points - np.roll(ordered_points, -1, axis=1)
    else:
        displacements = ordered_points[:, :-1] - ordered_points[:, 1:]
    distances_sqr = np.sum(np.square(displacements), axis=1)
    degenerate = np.greater(distances_sqr, 2)
    degenerate_graph_ordering = np.zeros(ordering.shape)
    if is_loop:
        degenerate_graph_ordering[ordering] = degenerate
    else:
        degenerate_graph_ordering[ordering[:-1]] = degenerate
        degenerate_graph_ordering[ordering[-1]] = degenerate[-1]


def decimate(border: Border, image_shape, decimate_angle_cutoff, decimate_pixel_count_threshold):
    border_connections = get_pixel_connections(border, image_shape)
    full_border_graph = coo_array(
        (np.ones([border_connections.shape[1]], dtype=np.bool), border_connections),
        shape=[border.full_points.shape[1]] * 2)
    point_ordering, has_loop = get_point_ordering(full_border_graph, border.full_points)
    border.is_loop = has_loop
    full_points_ordered = border.full_points[:, point_ordering]
    order_examiner = np.zeros(np.max(border.full_points, axis=1) - np.min(border.full_points, axis=1) + 1)
    order_examiner[*(full_points_ordered - np.min(border.full_points, axis=1, keepdims=True))] = np.arange(border.full_points.shape[1])
    decomposed_point_array = np.array(decimate_list(
        full_points_ordered,
        decimate_angle_cutoff,
        decimate_pixel_count_threshold
    ))
    border.decomposed_points = np.array(decomposed_point_array).T
    if not has_loop:
        border.endpoints = border.decomposed_points[[[0, 1], [0, 1]], [[0, 0], [-1, -1]]]

