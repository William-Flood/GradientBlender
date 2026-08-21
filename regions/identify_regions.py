import numpy as np
from util.grid import Grid, GridCell
from typing import List
from scipy.sparse.csgraph import connected_components

def check_edge_shadows_intersect(first_edges, second_edges):
    """
    Given parrel edges of rectangles, identify which rectangles are intersect when projected in the axis of the edges
    Args:
        first_border_set: The first set of edges of the rectangles as a flat array
        second_border_set: The second set of edges of the rectangles as a flat array - should be in the positive direction to
            first_border_set

    Returns: A square matrix representing rectangle adjacency

    """
    second_edges_past_first_edge = second_edges[:, np.newaxis] >= first_edges[np.newaxis, :]
    first_edge_not_past_second_edge = first_edges[:, np.newaxis] <= second_edges[np.newaxis, :]
    return np.logical_and(
        second_edges_past_first_edge,
        first_edge_not_past_second_edge
    )


def check_edge_shadows_align(first_edges, second_edges):
    """
    Given parrel edges of rectangles, identify which rectangles are directly adjacent when projected in the axis of the edges
    Args:
        first_border_set: The first set of edges of the rectangles as a flat array
        second_border_set: The second set of edges of the rectangles as a flat array - should be in the positive direction to
            first_border_set

    Returns: A square matrix representing rectangle adjacency

    """
    first_aligned_with_second = np.equal(
                             *np.broadcast_arrays(first_edges[:, np.newaxis], second_edges[np.newaxis, :])
    )
    return np.logical_or(first_aligned_with_second, first_aligned_with_second.T)


def check_rectangle_contact_in_window(rectangles: List[GridCell]):
    """
    Given a set of integer coordinate pairs defining rectangles, identify which rectangles are adjacent to each other
    Args:
        rectangles: A set of coordinate pairs defining rectangles

    Returns: A matrix of boolean values indicating rectangle adjacency

    """
    rect_tops = np.array([rectangle.top for rectangle in rectangles])
    rect_lefts = np.array([rectangle.left for rectangle in rectangles])
    rect_bottoms = np.array([rectangle.bottom for rectangle in rectangles])
    rect_rights = np.array([rectangle.right for rectangle in rectangles])
    horizontal_edges_align = check_edge_shadows_align(rect_tops, rect_bottoms)
    vertical_edges_align = check_edge_shadows_align(rect_lefts, rect_rights)
    horizontal_edges_intersect = check_edge_shadows_intersect(rect_tops, rect_bottoms)
    vertical_edges_intersect = check_edge_shadows_intersect(rect_lefts, rect_rights)
    rectangles_contact = np.logical_or(
        np.logical_and(
            horizontal_edges_align, vertical_edges_intersect
        ),
        np.logical_and(
            vertical_edges_align, horizontal_edges_intersect
        )
    )
    return rectangles_contact


def points_in_window(window_rect, points):
    start_coord_check = np.greater_equal(points, window_rect[:, 0, np.newaxis])
    start_point_check = start_coord_check.all(axis=0)
    end_coord_check = np.less_equal(points, window_rect[:, 1, np.newaxis])
    end_point_check = end_coord_check.all(axis=0)
    return start_point_check & end_point_check


def point_in_windows(window_rects, point):
    start_coord_check = np.greater_equal(point[:, np.newaxis], window_rects[:, 0, :])
    start_point_check = start_coord_check.all(axis=0)
    end_coord_check = np.less_equal(point[:, np.newaxis], window_rects[:, 1, :])
    end_point_check = end_coord_check.all(axis=0)
    return start_point_check & end_point_check


def find_grids_in_window(window_rect, grid_rects):
    grid_start_check = points_in_window(window_rect, grid_rects[:, 0, :])
    grid_end_check = points_in_window(window_rect, grid_rects[:, 1, :])
    window_start_check = point_in_windows(grid_rects, window_rect[:, 0])
    window_end_check = point_in_windows(grid_rects, window_rect[:, 1])
    return grid_start_check | grid_end_check | window_start_check | window_end_check


def deduplicate_window_full_label_pair_list(window_full_existing_pairs, window_unique_labels):
    matches_per_label = np.equal(window_full_existing_pairs[np.newaxis, :, 0], window_unique_labels[:, np.newaxis])
    full_pair_starts = np.argmax(matches_per_label, axis=1)
    return window_full_existing_pairs[full_pair_starts]


def make_reassignment_map(full_labels, window_labels, window_grid_indices):
    window_existing_labels = full_labels[window_grid_indices]
    window_grid_has_label = np.not_equal(window_existing_labels, -1)
    if window_grid_has_label.any():
        full_and_window = np.stack((window_labels, window_existing_labels), axis=1)
        window_full_existing_pairs = np.unique(full_and_window[window_grid_has_label], axis=0)
        current_labels_with_match, matching_counts = np.unique(window_full_existing_pairs[:, 0], return_counts=True)
        window_label_joins_full = np.greater(matching_counts, 1)
        window_labels_joining_full = current_labels_with_match[window_label_joins_full]
        joining_pairs = window_full_existing_pairs[np.isin(window_full_existing_pairs[:, 0], window_labels_joining_full), :]
        if window_label_joins_full.any():
            window_full_existing_pairs = deduplicate_window_full_label_pair_list(
                window_full_existing_pairs,
                current_labels_with_match
            )
        window_new_components = np.unique(
            window_labels[np.logical_not(np.isin(window_labels, window_full_existing_pairs[:, 0]))]
        )
        window_new_component_labels = np.argsort(window_new_components) + 1 + np.max(full_labels)
        window_labels_to_new_labels = np.stack((window_new_components, window_new_component_labels), axis=1)
        window_label_reassignments = np.concat((window_full_existing_pairs, window_labels_to_new_labels), axis=0)
        reassignment_order = np.argsort(window_label_reassignments[:, 0])

        return window_label_reassignments[reassignment_order[window_labels], 1], joining_pairs
    else:
        return window_labels, []


def make_substitution_map(labels_to_join):
    substitutions = []
    for group_id in np.unique(labels_to_join[:, 0]):
        group = labels_to_join[np.equal(labels_to_join[:, 0], group_id), 1]
        assignment_value = group[0]
        substitutions.extend([[assignment_value, assignment_target] for assignment_target in group[1:]])
    return substitutions


def normalize_labels(full_labels, labels_to_join):
    substitutions = make_substitution_map(labels_to_join)
    for substitution_pair in substitutions:
        full_labels[np.equal(full_labels, substitution_pair[0])] = substitution_pair[1]


def combine_labels(full_labels, window_labels, window_grid_indices):
    window_full_labels, duplicate_label_sets = make_reassignment_map(full_labels, window_labels, window_grid_indices)
    full_labels[window_grid_indices] = window_full_labels
    if len(duplicate_label_sets) > 0:
        normalize_labels(full_labels, duplicate_label_sets)


def find_connected_regions(region_grids: List[GridCell], values_array):
    MAX_WINDOW_SIDE_LENGTH = 1000
    full_image_labels = np.full([len(region_grids)], -1)
    grid_rects = np.stack([[[grid.top, grid.bottom], [grid.left, grid.right]] for grid in region_grids], axis=2)
    max_window_size = np.sqrt(values_array.shape).astype(np.int32)
    window_size = np.clip(max_window_size, 0, MAX_WINDOW_SIDE_LENGTH)
    row_starts = np.arange(0, values_array.shape[0], window_size[0])
    column_starts = range(0, values_array.shape[1], window_size[1])
    rows = np.stack((row_starts, row_starts + window_size[0]), axis=1)
    columns = np.stack((column_starts, column_starts + window_size[1]), axis=1)
    windows_over_grid = np.stack(np.broadcast_arrays(rows[:, np.newaxis, :], columns[np.newaxis, :, :]), axis=2)
    windows = np.reshape(windows_over_grid, [-1, 2, 2])
    for window_rect in windows:
        grid_inclusion = find_grids_in_window(window_rect, grid_rects)
        if grid_inclusion.any():
            region_grids_in_window = [
                grid for grid, grid_is_in_window in
                zip(region_grids, grid_inclusion)
                if grid_is_in_window
            ]
            adjacent_regions = check_rectangle_contact_in_window(region_grids_in_window)

            region_top_lefts = np.array([
                [rectangle.top, rectangle.left] for rectangle in region_grids_in_window
            ]).T
            region_values = values_array[*region_top_lefts]
            region_matches = np.equal(region_values[:, np.newaxis], region_values[np.newaxis, :])
            grid_graph = np.logical_and(region_matches, adjacent_regions)
            num_components, component_labels = connected_components(grid_graph)
            combine_labels(full_image_labels, component_labels, np.flatnonzero(grid_inclusion))
    return full_image_labels


def check_single_region(grid: Grid, values_array):
    grid_values_unmasked = values_array[*grid.points_per_grid]
    grid_values = np.where(
        grid.padding_mask,
        grid_values_unmasked,
        np.full(grid_values_unmasked.shape, np.nan)
    )
    grid_min_value = np.nanmin(grid_values, axis=1)
    grid_max_value = np.nanmax(grid_values, axis=1)
    return np.equal(grid_max_value, grid_min_value)


def identify_regions(values_array):
    image_points = np.reshape(np.indices(values_array.shape), [2, -1])

    values_grid = Grid(values_array.shape, image_points)
    region_grids: List[GridCell] = []
    while values_grid.grid.shape[1] > 0:
        values_grid.subdivide()
        mono_region_grids = check_single_region(values_grid, values_array)
        region_grids.extend(values_grid.get_cells(mono_region_grids))
        values_grid.filter(np.logical_not(mono_region_grids))
    region_labels = find_connected_regions(region_grids, values_array)
    unique_labels = np.unique(region_labels)
    assert -1 not in unique_labels
    combined_regions_ids = [
        np.argwhere(region_labels == region_id).flatten() for region_id in unique_labels
    ]
    regions_points = [
        np.concat([region_grids[region_id].points for region_id in region_ids], axis=1)
        for region_ids in combined_regions_ids
    ]
    return regions_points
