import numpy as np
from numpy.typing import NDArray
from borders.border import Border
from grid import Grid, GridCell
from typing import List
from scipy.sparse.csgraph import connected_components
from regions.identify import test_region
from scipy.sparse import csr_array


def get_cell_border_pixels(cell: GridCell):
    left_pixels = cell.points[1] == cell.left
    top_pixels = cell.points[0] == cell.top
    right_pixels = cell.points[1] == cell.right - 1
    bottom_pixels = cell.points[0] == cell.bottom - 1
    cell_border_pixels = left_pixels | top_pixels | right_pixels | bottom_pixels
    return cell.points[:, cell_border_pixels]


def regress_pixels(points_per_grid, padding_mask):
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
        np.sum(np.square(displacements_from_average_in_grids[0, :, :]) * padding_mask, axis=1) / np.sum(padding_mask, axis=1)
        + 1e-8
    )
    return alphas, betas, r_squareds


def make_cell_chunks_labels_map(border, point_index_map, linearishized_grids, full_border_graph, image_shape):
    cell_point_indices = [point_index_map[* cell.points] for cell in linearishized_grids]
    cell_point_pairs = np.concat([np.reshape(np.stack(np.broadcast_arrays(
                                         point_indices[:, np.newaxis],
                                         point_indices[np.newaxis, :]), axis=2), [-1, 2])
                                 for point_indices in cell_point_indices], axis=0
    ).T
    cells_graph = np.zeros([border.full_points.shape[1], border.full_points.shape[1]])
    cells_graph[*cell_point_pairs] = True
    cell_chunks = np.logical_and(cells_graph, full_border_graph)
    cell_chunks_csr = csr_array(cell_chunks)
    cell_chunks_count, cell_chunks_labels = connected_components(cell_chunks_csr, return_labels=True)
    cell_chunks_labels_map = np.zeros(image_shape[:2], dtype=np.int32)
    cell_chunks_labels_map[*border.full_points] = cell_chunks_labels + 1
    return cell_chunks_labels_map


def make_joins_points_labels_map(linearishized_grids, point_index_map, full_border_graph, image_shape):
    cell_borders = np.concat([get_cell_border_pixels(cell) for cell in linearishized_grids], axis=1)
    cell_borders_indices = point_index_map[*cell_borders]
    cell_borders_graph_indices = np.stack(np.broadcast_arrays(
        cell_borders_indices[:, np.newaxis],
        cell_borders_indices[np.newaxis, :]), axis=0)
    cell_borders_graph = full_border_graph[*cell_borders_graph_indices]
    _, cell_borders_labels = connected_components(cell_borders_graph)
    border_points_labels_map = np.zeros(image_shape[:2], dtype=np.int32)
    border_points_labels_map[*cell_borders] = cell_borders_labels + 1
    return border_points_labels_map


def make_cell_join_chunk_pairs(cell_chunks_and_borders):
    cell_chunks_borders_list_unflitered = np.reshape(cell_chunks_and_borders, [-1, 2])
    cell_chunks_borders_list = cell_chunks_borders_list_unflitered[
        np.logical_not(cell_chunks_borders_list_unflitered[:, 1] == 0)
    ]
    cell_chunks_joins_pairs_unfiltered = np.unique(cell_chunks_borders_list, axis=0)
    unique_borders, border_join_counts = np.unique(cell_chunks_joins_pairs_unfiltered[:, 1], return_counts=True)
    mere_edges = unique_borders[border_join_counts == 1]
    cell_chunks_border_pairs_false_positives = np.isin(cell_chunks_joins_pairs_unfiltered[:, 1], mere_edges)
    return cell_chunks_joins_pairs_unfiltered[np.logical_not(
        cell_chunks_border_pairs_false_positives
    )]


def find_end_chunks(cell_chunks_border_pairs):
    chunk_counts = np.stack(
        np.unique(cell_chunks_border_pairs[:, 0], return_counts=True),
        axis=1
    )
    chunks_with_one = chunk_counts[chunk_counts[:, 1] == 1, 0]
    borders_count = np.stack(
        np.unique(cell_chunks_border_pairs[:, 1], return_counts=True),
        axis=1
    )
    borders_with_two = borders_count[borders_count[:, 1] == 2, 0]
    end_chunks = np.logical_and(
        np.isin(cell_chunks_border_pairs[:, 0], chunks_with_one),
        np.isin(cell_chunks_border_pairs[:, 1], borders_with_two)
    )
    return cell_chunks_border_pairs[end_chunks]


def find_chunk_joins_anchor_points(joins_points_labels_map, point_index_map, border_points):
    joins_labels_flat_with_zeros = joins_points_labels_map.flatten()
    joins_points_indices = point_index_map.flatten()[joins_labels_flat_with_zeros != 0]
    joins_labels_flat = joins_labels_flat_with_zeros[joins_labels_flat_with_zeros != 0]
    max_join_label = np.max(joins_labels_flat)
    joins_labels_and_indexes = np.stack([joins_labels_flat, joins_points_indices], axis=1)

    def get_join_center(join_label):
        indices_to_average = joins_labels_and_indexes[joins_labels_and_indexes[:, 0] == join_label, 1]
        points_to_average = border_points[:, indices_to_average]
        return np.average(points_to_average, axis=1)

    return np.array([get_join_center(join_label + 1) for join_label in range(max_join_label)])


def get_chunk_points(chunk_map, chunk_id):
    """
    Returns the indices of elements inside chunk_map that match chunk_id
    Args:
        chunk_map:
        chunk_id:

    Returns:

    """
    filtered_chunk_map = np.equal(chunk_map, chunk_id)
    chunk_indices = filtered_chunk_map.nonzero()
    return np.indices(chunk_map.shape)[:, *chunk_indices]


def get_opposing_from_join(chunk_map, chunk_id, join_point):
    """
    Finds the furthest point in a chunk from a specified point
    Args:
        chunk_map:
        chunk_id:
        join_point:

    Returns:

    """
    chunk_points = get_chunk_points(chunk_map, chunk_id)
    chunk_point_displacement_from_join = chunk_points - join_point[:, np.newaxis]
    chunk_point_distances_from_join = np.sum(
        np.square(chunk_point_displacement_from_join),
        axis=0
    )
    return chunk_points[:, np.argmax(chunk_point_distances_from_join)]


def find_next_chunk(chunk_and_join_grid, join_index):
    join_chunks_array = chunk_and_join_grid[:, join_index]
    if sum(join_chunks_array) == 1:
        return np.argmax(join_chunks_array)
    else:
        candidate_indexes = np.flatnonzero(join_chunks_array)
        candidate_join_sums = np.sum(chunk_and_join_grid[candidate_indexes], axis=1)
        return candidate_indexes[np.argmax(candidate_join_sums)]



def order_joins(cell_chunks_joins_pairs, start_chunk):
    chunk_and_join_grid = np.zeros(np.max(cell_chunks_joins_pairs, axis=0), dtype=np.bool)
    chunk_and_join_grid[*(cell_chunks_joins_pairs.T - 1)] = 1
    chunk_joins_array = chunk_and_join_grid[start_chunk - 1]
    last_chunk = start_chunk - 1
    while chunk_joins_array.any():
        next_join = np.argmax(chunk_joins_array)
        yield next_join
        chunk_and_join_grid[last_chunk, next_join] = False
        next_chunk = find_next_chunk(chunk_and_join_grid, next_join)
        chunk_joins_array = chunk_and_join_grid[next_chunk]
        last_chunk = next_chunk
        chunk_and_join_grid[:, next_join] = False




def path_through_chunks(
        chunk_joins_anchor_points,
        cell_chunks_joins_pairs,
        chunk_map,
        start_chunk
):
    if start_chunk is not None:
        start_join_label = cell_chunks_joins_pairs[np.argmax(cell_chunks_joins_pairs[:, 0] == start_chunk), 1]
        start_join_index = start_join_label - 1
        start_join_point = chunk_joins_anchor_points[start_join_index]
        decimated_points = [get_opposing_from_join(
            chunk_map,
            start_chunk,
            start_join_point
        )]
        decimated_points.extend(
            chunk_joins_anchor_points[join_index] for join_index in
            order_joins(cell_chunks_joins_pairs, start_chunk)
        )
        decimated_points.append(get_opposing_from_join(
            chunk_map,
            start_chunk,
            decimated_points[-1]
        ))
        return np.array(decimated_points).T
    else:
        return np.array([
            chunk_joins_anchor_points[join_index] for join_index in order_joins(cell_chunks_joins_pairs, cell_chunks_joins_pairs[0, 0])]
        ).T


def test_grid(point_index_map, grid: Grid, pulled_cells):
    grid_cells = grid.get_cells()
    total_points = np.concat([grid.points for grid_list in [grid_cells, pulled_cells] for grid in grid_list], axis=1)
    points_indexes = point_index_map[*total_points]
    index_test = np.zeros([total_points.shape[1]], dtype=np.bool)
    index_test[points_indexes] = 1
    assert index_test.all()


def decimate(border: Border, image_shape, decimate_r_2_threshold):
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
    border_grid = Grid(image_shape[:2], border.full_points)
    linearishized_grids: List[GridCell] = []
    test_grid(point_index_map, border_grid, linearishized_grids)
    cycle_count = 0
    while border_grid.grid.shape[1] > 0:
        border_grid.subdivide()
        _, _, r_squared = regress_pixels(border_grid.points_per_grid, border_grid.padding_mask)
        r_squared_over_threshold = r_squared > decimate_r_2_threshold
        linearishized_grids.extend(border_grid.get_cells(r_squared_over_threshold))
        border_grid.filter(np.logical_not(r_squared_over_threshold))
        test_grid(point_index_map, border_grid, linearishized_grids)
        cycle_count += 1
    cell_chunks_labels_map = make_cell_chunks_labels_map(
        border,
        point_index_map,
        linearishized_grids,
        full_border_graph,
        image_shape)
    joins_points_labels_map = make_joins_points_labels_map(
        linearishized_grids, point_index_map, full_border_graph, image_shape
    )
    cell_chunks_and_joins = np.stack([cell_chunks_labels_map, joins_points_labels_map], axis=2)
    cell_chunks_joins_pairs = make_cell_join_chunk_pairs(cell_chunks_and_joins)
    if cell_chunks_joins_pairs.shape[0] > 0:
        chunk_joins_anchor_points = find_chunk_joins_anchor_points(
            joins_points_labels_map,
            point_index_map,
            border.full_points
        )
        end_chunks = find_end_chunks(cell_chunks_joins_pairs)
        if end_chunks.shape[0] > 0:
            border.decomposed_points = path_through_chunks(
                chunk_joins_anchor_points,
                cell_chunks_joins_pairs,
                cell_chunks_labels_map,
                end_chunks[0, 0]
            )
        else:
            border.decomposed_points = path_through_chunks(
                chunk_joins_anchor_points,
                cell_chunks_joins_pairs,
                cell_chunks_labels_map,
                None
            )
