import numpy as np
from grid import Grid, GridCell
from typing import List
from PIL import Image
from scipy.sparse.csgraph import connected_components
import time


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


def check_rectangle_contact(rectangles: List[GridCell]):
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


def find_regions_to_combine(region_grids, values_array):
    adjacent_regions = check_rectangle_contact(region_grids)

    region_top_lefts = np.array([
        [rectangle.top, rectangle.left] for rectangle in region_grids
    ]).T
    region_values = values_array[*region_top_lefts]
    region_matches = np.equal(region_values[:, np.newaxis], region_values[np.newaxis, :])
    return np.logical_and(region_matches, adjacent_regions)


def split_truncated_ids(truncated_regions_to_combine, last_fetched):
    next_connected_rows = truncated_regions_to_combine[last_fetched, :]
    truncated_ids = np.arange(truncated_regions_to_combine.shape[0])
    next_connected_ids = truncated_ids[
        np.any(next_connected_rows, axis=0)
    ]
    return next_connected_ids


def get_ids_to_combine_into_region(truncated_regions_to_combine, original_grid_indices):
    last_fetched = np.array([0])
    region_ids = []
    truncate_to = np.ones(original_grid_indices.shape, dtype=bool)
    while len(last_fetched) > 0:
        region_ids.extend(original_grid_indices[last_fetched])
        truncate_to[last_fetched] = False
        last_fetched = split_truncated_ids(truncated_regions_to_combine, last_fetched)
        truncated_regions_to_combine[:, last_fetched] = False
    truncated_regions_to_combine = (
        truncated_regions_to_combine[truncate_to]
    )[:, truncate_to]
    original_grid_indices = original_grid_indices[truncate_to]
    return truncated_regions_to_combine, region_ids, original_grid_indices


def combine_ids_into_regions(regions_to_combine):
    original_grid_indices = np.arange(regions_to_combine.shape[0])
    truncated_regions_to_combine = regions_to_combine
    regions_ids = []
    while len(truncated_regions_to_combine) > 0:
        truncated_regions_to_combine, region_ids, original_grid_indices = get_ids_to_combine_into_region(
            truncated_regions_to_combine,
            original_grid_indices
        )
        regions_ids.append(region_ids)
    return regions_ids


def test_region(image_size, points, name=None):
    test_image_array = np.zeros([*image_size, 4], np.uint8)
    test_image_array[*points, :] = 255
    test_image = Image.fromarray(test_image_array)
    if name is None:
        test_image.show()
    else:
        test_image.save(name)



def grid_index(region_grids, top, left):
    return [grid_id for grid_id, grid in enumerate(region_grids) if grid.top == top and grid.left == left]


def check_grid_values(values_array, grid):
    return values_array[grid.top:grid.bottom, grid.left:grid.right]


def identify_regions(values_array, defuzz_threshold):
    image_points = np.reshape(np.indices(values_array.shape), [2, -1])

    values_grid = Grid(values_array.shape, image_points)
    region_grids: List[GridCell] = []
    while values_grid.grid.shape[1] > 0:
        values_grid.subdivide()
        mono_region_grids = check_single_region(values_grid, values_array)
        region_grids.extend(values_grid.get_cells(mono_region_grids))
        values_grid.filter(np.logical_not(mono_region_grids))
    grid_merge_start = time.time()
    regions_to_combine = find_regions_to_combine(region_grids, values_array)
    grid_merge_end = time.time() - grid_merge_start
    n_regions, region_labels = connected_components(regions_to_combine)
    combined_regions_ids = [
        np.argwhere(region_labels == region_id).flatten() for region_id in range(n_regions)
    ]
    regions_points = [
        np.concat([region_grids[region_id].points for region_id in region_ids], axis=1)
        for region_ids in combined_regions_ids
    ]
    defuzzed_region_points = []
    defuzzed_region_ids = []
    for points, ids in zip(regions_points, combined_regions_ids):
        if points.shape[1] > defuzz_threshold:
            defuzzed_region_points.append(points)
            defuzzed_region_ids.append(ids)
            # test_region(values_array.shape, points, f"./TestRegions/{ids[0]}.png")
    # test_region(values_array.shape, regions_points[1])
    region_values = [
        values_array[region_grids[region_ids[0]].top, region_grids[region_ids[0]].left]
        for region_ids in defuzzed_region_ids
    ]
    return defuzzed_region_points, region_values