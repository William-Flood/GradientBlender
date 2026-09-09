import numpy as np


def transpose_and_shape_new_grid(transposed_expanded_new_grid):
    """
    Reshapes a set of grid coordinates
    :param transposed_expanded_new_grid: An nx2x8 array
    :return: A 2xnx4x2 array
    """
    splits_axis_pulled_out = np.reshape(transposed_expanded_new_grid, [-1, 2, 4, 2])
    axes_aligned = np.transpose(splits_axis_pulled_out, [1, 0, 2, 3])
    return axes_aligned


def subdivide_grid(grid_rects):
    """
    For a set of grid squares, return a new grid consisting of the existing grid split in half horizontally and vertically
    :param grid_rects: An nx2x2 array of ints, representing a set of
        [[grid_top_y, grid_left_x], [grid_bottom_y, grid_right_x]] coordinates; grid_bottom_y, grid_right_x are exclusive

    :return: An nx4x2x2 array in which each grid in grid_rects has been split (as close as possible to) evenly into four
    sections with a horizontal and vertical cut
    """
    new_grid_transposed = np.transpose(grid_rects, [1, 0, 2])
    grid_splitter_matrix = np.array(
        [
            [
                [1, 0.5, 1, 0.5, 0.5, 0, 0.5, 0],
                [0, 0.5, 0, 0.5, 0.5, 1, 0.5, 1]
            ],

            [
                [1, 0.5, 0.5, 0, 1, 0.5, 0.5, 0],
                [0, 0.5, 0.5, 1, 0, 0.5, 0.5, 1]
            ]
        ]
    )
    subdivided_grid_expanded_transposed = \
        np.einsum("gao,aon->gan", new_grid_transposed, grid_splitter_matrix) \
            .astype(np.int32)
    subdivided_grid = transpose_and_shape_new_grid(subdivided_grid_expanded_transposed)
    return subdivided_grid


def get_division_matches(grid_divisions, points_per_grid, padding_mask):
    points_below_and_right = np.logical_and(
        points_per_grid[0, :, np.newaxis, :] >= grid_divisions[0, :, :, 0, np.newaxis],
        points_per_grid[1, :, np.newaxis, :] >= grid_divisions[1, :, :, 0, np.newaxis]
    )
    points_top_and_above = np.logical_and(
        points_per_grid[0, :, np.newaxis, :] < grid_divisions[0, :, :, 1, np.newaxis],
        points_per_grid[1, :, np.newaxis, :] < grid_divisions[1, :, :, 1, np.newaxis]
    )
    points_in_grid_divisions = np.logical_and(
        np.logical_and(
            points_below_and_right,
            points_top_and_above
        ),
        padding_mask[:, np.newaxis, :]
    )
    return points_in_grid_divisions


def shape_to_next_round(grid_divisions, points_per_grid_division, grid_division_padding, max_points_in_grid):
    points_per_grid = np.reshape(points_per_grid_division, [2, -1, max_points_in_grid])
    padding_mask = np.reshape(grid_division_padding, [-1, max_points_in_grid])
    grid = np.reshape(grid_divisions, [2, -1, 2])
    return points_per_grid, padding_mask, grid


def shape_and_truncate_to_next_round(
        grid_divisions,
        points_per_grid_division,
        grid_division_padding,
        max_points_in_grid,
        points_in_grid_divisions_count
):
    points_per_grid, padding_mask, grid = shape_to_next_round(
        grid_divisions,
        points_per_grid_division,
        grid_division_padding,
        max_points_in_grid
    )
    points_in_grid_count = np.reshape(points_in_grid_divisions_count, [-1])
    grid_has_points = (points_in_grid_count > 0)
    points_per_grid_truncated = points_per_grid[:, grid_has_points, :]
    padding_mask_truncated = padding_mask[grid_has_points]
    grid_truncated = grid[:, grid_has_points, :]
    return points_per_grid_truncated, padding_mask_truncated, grid_truncated


def make_next_round_grid(points_in_grid_divisions, points_per_grid, grid_divisions):
    points_in_grid_divisions_count = np.sum(points_in_grid_divisions, axis=2)
    max_points_in_grid = np.max(points_in_grid_divisions_count)
    points_in_grid_divisions_truncated = np.argpartition(
        np.logical_not(points_in_grid_divisions),
        max_points_in_grid - 1,
        axis=2
    )[:, :, :max_points_in_grid]
    points_per_grid_division = np.take_along_axis(
        points_per_grid[:, :, np.newaxis, :],
        points_in_grid_divisions_truncated[np.newaxis, :, :, :],
        axis=3
    )
    grid_division_padding = np.take_along_axis(
        points_in_grid_divisions,
        points_in_grid_divisions_truncated,
        axis=2
    )
    points_per_grid, padding_mask, grid = shape_and_truncate_to_next_round(
        grid_divisions,
        points_per_grid_division,
        grid_division_padding,
        max_points_in_grid,
        points_in_grid_divisions_count
    )
    return grid, padding_mask, points_per_grid


class GridCell:
    def __init__(self, coords, points, cell_mask):
        self.points = points[:, cell_mask]
        self.top = coords[0, 0]
        self.left = coords[1, 0]
        self.bottom = coords[0, 1]
        self.right = coords[1, 1]

    @property
    def border_point_mask(self):
        return (np.equal(self.points[0], self.top) |
                np.equal(self.points[0], self.bottom - 1) |
                np.equal(self.points[1], self.left) |
                np.equal(self.points[1], self.right - 1)
        )

    @property
    def border_points(self):
        return self.points[:, self.border_point_mask]


class Grid:
    """
    Used to split a set of 2-dimensional coordinates into rectangular regions
    """

    def __init__(self, grid_shape, points):
        image_height, image_width = grid_shape
        self.grid = np.array([[[0, image_height]], [[0, image_width]]], dtype=np.int32)
        self.points_per_grid = np.reshape(points, [2, 1, -1])
        self.padding_mask = np.ones([1, points.shape[1]], dtype=np.bool)

    def subdivide(self):
        """Recomputes the grid by dividing each cell in half horizontally and vertically"""
        grid_divisions = subdivide_grid(self.grid)
        points_in_grid_divisions = get_division_matches(grid_divisions, self.points_per_grid, self.padding_mask)
        self.grid, self.padding_mask, self.points_per_grid = make_next_round_grid(
            points_in_grid_divisions,
            self.points_per_grid,
            grid_divisions
        )

    def get_cells(self, cell_filter=None):
        if cell_filter is None:
            cell_filter = np.ones([self.grid.shape[1]], dtype=np.bool)
        points_per_cell = np.transpose(self.points_per_grid, [1, 0, 2])[cell_filter]
        coords_per_cell = np.transpose(self.grid, [1, 0, 2])[cell_filter]
        fetch_mask = self.padding_mask[cell_filter]
        return [
            GridCell(cell_coords, cell_points, cell_mask)
            for cell_coords, cell_points, cell_mask in zip(coords_per_cell, points_per_cell, fetch_mask)
        ]

    def filter(self, cell_filter):
        """
        Removes grid cells according to a boolean mask
        Args:
            cell_filter: An array of boolean values corresponding to cells in this grid - cells corresponding to True values
            will be kept

        Returns:

        """
        self.grid = self.grid[:, cell_filter, :]
        self.points_per_grid = self.points_per_grid[:, cell_filter, :]
        self.padding_mask = self.padding_mask[cell_filter]
