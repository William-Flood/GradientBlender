import numpy as np
from numpy.typing import NDArray
from borders.border import Border
from scipy.sparse.csgraph import breadth_first_order, shortest_path
from scipy.stats import linregress


# Unused; retained for future reference
def regress_graph(points_per_grid, padding_mask):
    average_points_in_grids = np.sum(points_per_grid * padding_mask[np.newaxis, :, :], axis=2) / \
                              np.sum(padding_mask, axis=1, keepdims=True).T
    displacements_from_average_in_grids = (points_per_grid - average_points_in_grids[:, :, np.newaxis]) * \
                                          padding_mask[np.newaxis, :, :]
    betas = np.sum(
        displacements_from_average_in_grids[0, :, :] * displacements_from_average_in_grids[1, :, :],
        axis=1
    ) / \
            (np.sum(np.square(displacements_from_average_in_grids[1, :, :]), axis=1) + 1e-8)
    alphas = average_points_in_grids[0, :] - betas * average_points_in_grids[1, :]
    predicted_ys = alphas[:, np.newaxis] + betas[:, np.newaxis] * points_per_grid[1, :, :]
    residual_sum_of_squares = np.sum(
        np.square(
            points_per_grid[0, :, :] - predicted_ys
        ) * padding_mask,
        axis=1
    ) / np.sum(padding_mask, axis=1)
    r_squareds = 1 - residual_sum_of_squares / (
            np.sum(np.square(displacements_from_average_in_grids[0, :, :]) * padding_mask, axis=1) / np.sum(
        padding_mask, axis=1)
            + 1e-8
    )
    return alphas, betas, r_squareds


def get_point_ordering(connection_graph, points):
    adjusted_points = points - np.min(points, axis=1, keepdims=True)
    adjusted_window = np.max(points, axis=1) - np.min(points, axis=1) + 1
    first_point_finder = breadth_first_order(connection_graph, 0, return_predecessors=False)
    first_endpoint_index = first_point_finder[-1]
    distances_from_first = shortest_path(connection_graph, indices=first_endpoint_index)
    # An assignment array which, when applied to points, produces the correct ordering of points in the border curve
    # (if the border does not contain any loops)
    point_sorting_over_first = np.argsort(distances_from_first)
    pivot_index = int(point_sorting_over_first[connection_graph.shape[0] // 2])
    pivot_distances = shortest_path(connection_graph, indices=pivot_index)
    first_endpoint_pivot_predecessors = pivot_distances[first_endpoint_index]
    pivot_equivalent_predecessors = pivot_distances[distances_from_first == distances_from_first[pivot_index]]
    loop_found = first_endpoint_pivot_predecessors < np.max(pivot_equivalent_predecessors)
    if loop_found:
        pivot_to_first = pivot_distances < pivot_distances[first_endpoint_index]
        antifirst_point = np.argmax(point_sorting_over_first)
        pivot_to_antifirst = pivot_distances < pivot_distances[antifirst_point]
        first_to_pivot = point_sorting_over_first < point_sorting_over_first[pivot_index]
        antifirst_to_pivot_arc = np.logical_and(pivot_to_antifirst, np.logical_not(first_to_pivot))
        pivot_to_first_arc = np.logical_and(pivot_to_first, first_to_pivot)
        # a binary assignment for every point in points to one of the two arcs in the loop
        pivot_side_assignment_unsorted = np.logical_or(antifirst_to_pivot_arc, pivot_to_first_arc)
        pivot_side_assignment = pivot_side_assignment_unsorted[point_sorting_over_first]
        pivot_side_loop = point_sorting_over_first[pivot_side_assignment]
        antipivot_side_loop = point_sorting_over_first[np.logical_not(pivot_side_assignment)]
        assignment_inspection = np.zeros(adjusted_window)
        assignment_inspection[*adjusted_points] = (pivot_side_assignment.astype(np.int8) + 1)
        pivot_side_loop_inspection = np.zeros(adjusted_window, dtype=np.int8)
        adjusted_pivot_side_points = adjusted_points[:, pivot_side_loop]
        pivot_side_loop_inspection[*adjusted_pivot_side_points] = np.arange(adjusted_pivot_side_points.shape[1])
        loop_point_ordering = np.concat(
            [antipivot_side_loop,
             np.flip(pivot_side_loop, axis=0)]
        )
        return loop_point_ordering, True
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


def decimate_list(
        point_list,
        decimate_r_2_threshold,
        decimate_pixel_count_threshold
):
    if point_list.shape[1] > decimate_pixel_count_threshold:
        if np.min(point_list[1, :]) == np.max(point_list[1, :]):
            return [point_list[:, 0], point_list[:, -1]]
        else:
            test_full_line = linregress(point_list[1, :], point_list[0, :]).rvalue ** 2
            if test_full_line > decimate_r_2_threshold:
                return [point_list[:, 0], point_list[:, -1]]
            else:
                next_point = find_longest_line_within_r_2_threshold(
                    point_list,
                    0,
                    point_list.shape[1],
                    decimate_r_2_threshold,
                    decimate_pixel_count_threshold
                ) + 1
                if next_point >= point_list.shape[1] - 1:
                    return [point_list[:, 0], point_list[:, -1]]
                else:
                    return [point_list[:, 0], *decimate_list(
                        point_list[:, next_point:],
                        decimate_r_2_threshold,
                        decimate_pixel_count_threshold
                    )]
    else:
        return [point_list[:, 0], point_list[:, -1]]


def decimate(border: Border, image_shape, decimate_r_2_threshold, decimate_pixel_count_threshold):
    point_index_map = np.zeros(image_shape[:2], dtype=np.int32)
    point_index_map[*border.full_points] = np.arange(border.full_points.shape[1])
    points_indexes = point_index_map[*border.full_points]
    index_test = np.zeros([border.full_points.shape[1]], dtype=np.bool)
    index_test[points_indexes] = 1
    assert index_test.all()
    connecting_offsets = np.array(
        [[row_offset, column_offset] for column_offset in [-1, 0, 1] for row_offset in [-1, 0, 1]]
    ).T
    border_points_and_connections = border.full_points[:, np.newaxis, :] + connecting_offsets[:, :, np.newaxis]
    compare_points = np.repeat(border.full_points[:, np.newaxis, :], 9, axis=1)
    point_comparison = np.equal(border_points_and_connections[:, :, :, np.newaxis], compare_points[:, :, np.newaxis, :])
    full_border_graph = np.logical_and(
        np.any(
            np.all(
                point_comparison, axis=0
            ),
            axis=0
        ),
        np.logical_not(
            np.identity(border.full_points.shape[1]), dtype=np.bool
        )
    )
    point_ordering, has_loop = get_point_ordering(full_border_graph, border.full_points)
    border.loop = has_loop
    full_points_ordered = border.full_points[:, point_ordering]
    order_examiner = np.zeros(np.max(border.full_points, axis=1) - np.min(border.full_points, axis=1) + 1)
    order_examiner[*(full_points_ordered - np.min(border.full_points, axis=1, keepdims=True))] = np.arange(border.full_points.shape[1])
    decomposed_point_array = np.array(decimate_list(
        full_points_ordered,
        decimate_r_2_threshold,
        decimate_pixel_count_threshold
    ))
    border.decomposed_points = np.array(decomposed_point_array).T

