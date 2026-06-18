import numpy as np
from collections import defaultdict
from PIL import Image
from grid import Grid


def add_border_pixel(values_array, coords, border_tree, max_height, max_width):
    """
    Updates a dictionary tree of region border points based on a selected position in the image array
    :param values_array: An hxw array of pixel brightness values.  Should be pre-processed to remove aliasing - otherwise
    the number of regions may be excessive
    :param coords: The [y, x] location to update the tree by
    :param border_tree: A dictionary of points per brightness - uses float32 brightness values from values_1 as keys, and a nested dictionary as
    values. The nested dictionary uses float32 brightness values from values_2 as keys, and point coordinates as values.
    :param max_height: The maximum y value of the image
    :param max_width: The maximum x value of the image
    :return:
    """
    value_1 = values_array[*coords]
    if (coords[0] == 0 or coords[1] == 0 or coords[0] == max_height or coords[1] == max_width):
        if value_1 != -1:
            border_tree[value_1][value_1].append(coords)
    else:
        value_2_coords = np.array([coords[0] + 1, coords[1]])
        value_2 = values_array[*value_2_coords]
        value_3_coords = np.array([coords[0], coords[1] + 1])
        value_3 = values_array[*value_3_coords]

        if value_1 != -1:
            if value_2 != value_1:
                if value_2 == -1:
                    border_tree[value_1][value_1].append(coords)
                else:
                    border_tree[value_1][value_2].append(coords)
            if value_3 != value_1 and value_3 != value_2:
                if value_3 == -1:
                    border_tree[value_1][value_1].append(coords)
                else:
                    border_tree[value_1][value_3].append(coords)

        if value_2 != -1:
            if value_2 != value_1:
                if value_1 == -1:
                    border_tree[value_2][value_2].append(value_2_coords)
                else:
                    border_tree[value_2][value_1].append(value_2_coords)

        if value_3 != -1:
            if value_3 != value_1:
                if value_1 == -1:
                    border_tree[value_3][value_3].append(value_3_coords)
                else:
                    border_tree[value_3][value_1].append(value_3_coords)
    return border_tree



def add_border_pixels(values_array, border_locations):
    """
    Updates a dictionary of points bordering a given region of an image with new data
    :param values_array: An array of pixel brightness values
    :param border_locations: A boolean array indicating the location of pixels to add to the dictionary.
    :return: A dictionary of points per brightness - uses float32 brightness values from values_1 as keys, and a nested dictionary as
    values. The nested dictionary uses float32 brightness values from values_2 as keys, and point coordinates as values.
    """
    values_height, values_width = values_array.shape
    values_indexes = np.transpose(
        np.indices((values_height, values_width)),
        [1, 2, 0]
    )
    values_indexes_flattened = np.reshape(values_indexes, [values_height * values_width, 2])
    mask_flattened = np.reshape(border_locations, [values_height * values_width])

    border_indices = values_indexes_flattened[mask_flattened]
    border_tree = defaultdict(lambda: defaultdict(list))
    max_height = values_array.shape[0] - 1
    max_width = values_array.shape[1] - 1

    for coords in border_indices:
        add_border_pixel(values_array, coords, border_tree, max_height, max_width)
    return border_tree




def find_borders(values_array):
    values_vertical_compare = (np.equal(values_array[:-1, :], values_array[1:,:]) == False)
    values_horizontal_compare = (np.equal(values_array[:, :-1], values_array[:, 1:]) == False)
    border_array = np.logical_or(values_vertical_compare[:, :-1], (values_horizontal_compare[:-1, :]))
    border_array[0, :] = True
    border_array[-1, :] = True
    border_array[:, 0] = True
    border_array[:, -1] = True
    return border_array


def detect_drawn_border(values_array):
    """
    Identifies the boundaries of shaded regions drawn in the image
    :param values_array: The brightness values of pixels from a drawn image, as an hxw numpy array
    :return: A dictionary using float32 brightness values as keys, and a nested dictionary as
    values. The nested dictionary uses brightness values of the bordering values as keys, and point coordinates as values
    """
    border_array = find_borders(values_array)
    border_tree = add_border_pixels(values_array[:-1, :-1], border_array)

    return border_tree


def gridify_border(border_points, image_shape, division_steps):
    """
    Maps a set of border points to a grid
    :param border_points:
    :param image_shape:
    :param division_steps:
    :return:
    """
    # image_height, image_width, _ = image_shape
    # grid = np.array([[[0, 0], [image_height, image_width]]], dtype=np.int32)
    # points_per_grid = np.array([border_points])
    # max_points_in_grid = border_points.shape[0]
    # padding_mask = np.ones([1, max_points_in_grid])
    border_grid = Grid(image_shape[:2], border_points)
    for step in range(division_steps):
        # grid_divisions = subdivide_grid(grid)
        # points_in_grid_divisions = get_division_matches(grid_divisions, points_per_grid, padding_mask)
        # grid, padding_mask, points_per_grid = make_next_round_grid(
        #     points_in_grid_divisions,
        #     points_per_grid,
        #     grid_divisions
        # )
        border_grid.subdivide()
    return border_grid.grid, border_grid.points_per_grid, border_grid.padding_mask


def get_adjacent_to_border(border_points, filled_points, image_shape):
    filled_array = np.zeros(image_shape[:2], dtype=np.bool)
    filled_array[filled_points[..., 0], filled_points[..., 1]] = True
    filled_array[border_points[..., 0], border_points[..., 1]] = False
    adjacent = np.array(
        [[
            [1, 1], [0, 1], [-1, 1],
            [1, 0], [0, 0], [-1, 0],
            [1, -1], [0, -1], [-1, -1]
        ]],
        dtype=np.int32
    )
    adjacent_to_border_packed = border_points[:, np.newaxis, :] + adjacent
    adjacent_to_border = np.reshape(adjacent_to_border_packed, [-1, 2])
    return adjacent_to_border[
        filled_array[
            adjacent_to_border[..., 0],
            adjacent_to_border[..., 1]
        ]
    ]


def find_grid_slopes(points_per_grid, padding_mask):
    average_points_in_grids = np.sum(points_per_grid * padding_mask[:, :, np.newaxis], axis=1) / \
        np.sum(padding_mask, axis=1, keepdims=True)
    displacements_from_average_in_grids = (points_per_grid - average_points_in_grids[:, np.newaxis, :]) * \
                                          padding_mask[:, :, np.newaxis]
    betas = np.sum(
        displacements_from_average_in_grids[:, :, 0] * displacements_from_average_in_grids[:, :, 1],
        axis=1
    ) / \
           (np.sum(np.square(displacements_from_average_in_grids[:, :, 1]), axis=1) + 1e-8)
    alphas = average_points_in_grids[:, 0] - betas * average_points_in_grids[:, 1]
    predicted_ys = alphas[:, np.newaxis] + betas[:, np.newaxis] * points_per_grid[:, :, 1]
    residual_sum_of_squares = np.sum(
        np.square(
            points_per_grid[:, :, 0] - predicted_ys
        ) * padding_mask,
        axis=1
    ) / np.sum(padding_mask, axis=1)
    r_squareds = 1 - residual_sum_of_squares / (
        np.sum(np.square(displacements_from_average_in_grids[:, :, 0]) * padding_mask, axis=1)
        + 1e-8
    )
    return betas


def test_points_per_grid(points_per_grid, padding_mask, image_shape):
    test_image_array = np.zeros(image_shape, dtype=np.uint8)
    flattened_points_per_grid = np.reshape(points_per_grid, [2, -1])[
        :, padding_mask.flatten()
    ]
    test_image_array[*flattened_points_per_grid] = [255, 255, 255, 255]
    Image.fromarray(test_image_array).show()



def downsample_border(border_points, filled_points, image_shape, division_steps):
    """
    Creates a new border to simplify comparisons between interior and border points
    :param border_points:
    :param filled_points:
    :param image_shape:
    :param division_steps:
    :return:
    """
    # test_points_per_grid(np.array([border_points]), np.ones([1, border_points.shape[0]], dtype=np.bool), image_shape)
    grid, points_per_grid, padding_mask = gridify_border(border_points, image_shape, division_steps)
    test_points_per_grid(points_per_grid, padding_mask, image_shape)
    adjacent_to_border = get_adjacent_to_border(border_points, np.array(filled_points), image_shape)
    find_grid_slopes(points_per_grid, padding_mask)

