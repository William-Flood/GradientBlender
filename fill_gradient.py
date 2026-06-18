import time

from color_mapper import color_mapper
from PIL import Image
import numpy as np
from borders.detector import detect_drawn_border
from borders.divide import divide_borders
from borders.decimate import decimate
from regions.identify import identify_regions, test_region


def get_interpolations(
        interpolation_points_and_values,
        points_to_interpolate
):
    """
    Performs interpolation for an array of points based on the positions and values of provided interpolation anchors
    :param interpolation_points_and_values: The positions and values of the interpolation anchors
    :param points_to_interpolate: The positions of the points to perform interpolation on
    :return: The interpolated values calculated for the targets in points_to_interpolate
    """
    values, interpolation_points = np.split(interpolation_points_and_values, [1], axis=2)
    displacements = points_to_interpolate[:, np.newaxis, :] - interpolation_points
    distances = np.sqrt(
        np.sum(
            np.square(
                displacements
            )
            , axis=2
        )
    ) + 1e-8
    proportional_distances = distances / (np.sum(distances, axis=1, keepdims=True))
    value_portions = values[...,0] * (1 - proportional_distances)
    return np.sum(value_portions, axis=1)


def compare_raycasts(
        interpolation_targets_count,
        primary_anchor_indices,
        interpolation_targets_to_borders_displacements,
        interpolation_targets_to_borders_distances):
    """
    Finds the dot product between the unit rays between the primary anchor indices and each border point for each interpolation target
    :param interpolation_targets_count: The number of points to interpolate between
    :param primary_anchor_indices: The selections made for the first interpolation anchor to use for each interpolation target
    :param interpolation_targets_to_borders_displacements: The displacement vectors between the interpolation targets and all border points
    :param interpolation_targets_to_borders_distances: The distances between the interpolation targets and all border points
    :return: The dot product between the unit rays between the primary anchor indices and each border point for each interpolation target
    """
    sample_and_closest_border = np.stack((np.arange(interpolation_targets_count), primary_anchor_indices), axis=1)
    sample_points_to_borders_unit_displacements = ((
                                                          interpolation_targets_to_borders_displacements + 1e-8) /
                                                   np.sqrt(interpolation_targets_to_borders_distances[:,:,np.newaxis] + 1e-8))
    sample_and_closest_border_unit_displacements = sample_points_to_borders_unit_displacements[
        sample_and_closest_border[..., 0],
        sample_and_closest_border[..., 1]
    ]
    interpolation_targets_to_borders_displacements_transpose = np.transpose(sample_points_to_borders_unit_displacements, [0,2,1])
    return np.einsum(
        "pd,pdb->pb",
        -1 * sample_and_closest_border_unit_displacements,
        interpolation_targets_to_borders_displacements_transpose)


def raycast_to_border(
        interpolation_targets_count,
        interpolation_targets_to_borders_displacements,
        interpolation_targets_to_borders_distances,
        closest_border_points_to_interpolation_target_indices
):
    """
    Given a selection of points to perform pixel interpolations on and an initial selection of anchors for one side of
     the interpolation, perform a raycast to select the second anchor point out of the set of borders between regions
     within an image.
    :param interpolation_targets_count: The number of points to interpolate between.
    :param interpolation_targets_to_borders_displacements: The displacement vectors between the interpolation targets
    and all candidate anchors.
    :param interpolation_targets_to_borders_distances: The distances between the interpolation targets and all border points
    :param closest_border_points_to_interpolation_target_indices: The location within the set of border points of the
    closest border point to each target
    :return: The indices of the border point most closely opposite of the initial anchor point from the perspective of the interpolation targets
    """
    border_to_closest_dot = compare_raycasts(
        interpolation_targets_count,
        closest_border_points_to_interpolation_target_indices,
        interpolation_targets_to_borders_displacements,
        interpolation_targets_to_borders_distances)
    # Actually squared distance
    DISTANCE_FROM_RAY = 4
    threshold_dot = ((interpolation_targets_to_borders_distances - DISTANCE_FROM_RAY) / interpolation_targets_to_borders_distances)
    points_outside_threshold = np.logical_or(threshold_dot > np.square(border_to_closest_dot), border_to_closest_dot < 0)
    points_outside_threshold_adjustment = points_outside_threshold * np.max(interpolation_targets_to_borders_distances)
    adjusted_distances = interpolation_targets_to_borders_distances + points_outside_threshold_adjustment
    opposing_points_to_sample_indices = np.argmin(adjusted_distances, axis=1)
    return opposing_points_to_sample_indices


def get_interpolation_points_values(
        value,
        closest_border_points_to_samples_indices,
        opposing_border_point_indexes,
        values_indexes,
        all_borders_array
    ):
    """
    Determines the second interpolation anchor point (opposite the closest border point) to either a selected border point
    or the midpoint between the two border points, and computes the brightness values to interpolate between
    :param value: The brightness value of the region to produce a gradient across
    :param closest_border_points_to_samples_indices: The selected first anchor point
    :param opposing_border_point_indexes: The border points found by raycasting from the closest border points to the target point
    :param values_indexes: The value of each border point in the full list of border points
    :param all_borders_array: The full list of points on the outside border of the region to produce a gradient across
    :return: The values and positions of the interpolation anchors as an nx2x3 array, with each 2x3 matrix along the
    0-axis representing the value, the y-coordinate, and the x-coordinate of the two anchor points for each
    interpolation target
    """
    closest_border_values = values_indexes[closest_border_points_to_samples_indices]
    opposite_border_values = values_indexes[opposing_border_point_indexes]
    closest_and_opposing_values = np.stack((closest_border_values, opposite_border_values),axis=1)
    border_is_lighter_than_value = closest_and_opposing_values > value
    both_borders_are_lighter_than_value = np.all(border_is_lighter_than_value, axis=1, keepdims=True)
    border_is_darker_than_value = closest_and_opposing_values < value
    both_borders_are_darker_than_value = np.all(border_is_darker_than_value, axis=1, keepdims=True)
    use_midpoint = np.logical_or(both_borders_are_lighter_than_value, both_borders_are_darker_than_value)
    closest_borders_points = all_borders_array[closest_border_points_to_samples_indices]
    closest_point_and_value = np.concat(
        (
            values_indexes[closest_border_points_to_samples_indices, np.newaxis],
            closest_borders_points
        ),
        axis=1
    )
    opposing_border_points = all_borders_array[opposing_border_point_indexes]
    halfway_point = (closest_borders_points + opposing_border_points) / 2
    opposing_point_and_value = np.concat(
        (
            values_indexes[opposing_border_point_indexes, np.newaxis],
            opposing_border_points
        ),
        axis=1
    )
    halfway_point_and_value = np.concat(
        (
            np.full([len(closest_border_points_to_samples_indices), 1], value),
            halfway_point
        ),
        axis=1
    )

    opposing_or_midway_points_values = np.where(use_midpoint, halfway_point_and_value, opposing_point_and_value)
    closest_opposing_points_values = np.stack(
        [closest_point_and_value, opposing_or_midway_points_values], axis=1
    )
    closest_opposing_points_values[:, :, 0] = (closest_opposing_points_values[:, :, 0] + value) / 2
    return closest_opposing_points_values

def get_pixel_values_from_brightness_values(values_array):
    """
    Converts a flat array of brightness values into an nx4 array of pixel values
    :param values_array: An array of n brightness values
    :return: An nx4 array of pixel values, all with full opacity
    """
    interpolations_expanded = np.repeat(values_array[:, np.newaxis],3,axis=1)
    return np.concat(
        (interpolations_expanded, np.full([values_array.shape[0], 1], 255)), axis=1
    )


def get_interpolation_points_and_values(sample_points, value, borders):
    all_borders_array = np.concat(list(borders.values()), axis=0)
    sample_points_to_borders_displacements = sample_points[:, np.newaxis, :] - all_borders_array[np.newaxis, :, :]
    # Actually squared distance
    sample_points_to_borders_distances = np.sum(
        np.square(
            sample_points_to_borders_displacements
        ),
        axis=2
    )
    closest_border_points_to_samples_indices = np.argmin(sample_points_to_borders_distances, axis=1)
    opposing_border_point_indexes = raycast_to_border(
        sample_points.shape[0],
        sample_points_to_borders_displacements,
        sample_points_to_borders_distances,
        closest_border_points_to_samples_indices)
    values_indexes = np.concat([np.full([len(border_points)],value) for value, border_points in borders.items()])
    return get_interpolation_points_values(
        value,
        closest_border_points_to_samples_indices,
        opposing_border_point_indexes,
        values_indexes,
        all_borders_array
    )


def fill_sample_points(
        value,
        points,
        borders,
        batch_stripe_distance,
        row_offset,
        column_offset,
        gradient_array
):
    """
    Updates the values in gradient_array by interpolating a selection of points between two points on the border between
    shaded regions on a guide image
    :param value: The brightness value of pixels on the guide image of a region to perform interpolation within
    :param points: The points in the region to perform interpolation on
    :param borders: The points on the edge of the region to perform interpolation on
    :param batch_stripe_distance: The distance between pixels to select for interpolation
    :param row_offset: The vertical offset from the top of the stripes to draw interpolation targets from
    :param column_offset: The horizontal offset from the top of the stripes to draw interpolation targets from
    :param gradient_array: The array used to store the pixel values of the image with the gradient applied
    :return: None
    """
    points_offset_match = np.equal(points % batch_stripe_distance, np.array([[row_offset, column_offset]]))
    sample_points = points[np.all(points_offset_match, axis=1)]
    interpolation_values = get_interpolation_points_and_values(
        sample_points,
        value,
        borders)
    interpolations = get_interpolations(interpolation_values, sample_points)
    interpolations_with_alpha = get_pixel_values_from_brightness_values(interpolations)
    gradient_array[sample_points[..., 0], sample_points[..., 1]] = interpolations_with_alpha


def get_region_and_borders(guide_image_array, region_proportion_threshold):
    color_map = color_mapper(guide_image_array, region_proportion_threshold)
    values_array = np.full(guide_image_array.shape[:2], -1.0)
    for value, points in color_map.items():
        points_array = np.array(points)
        values_array[points_array[...,0], points_array[...,1]] = value
    borders = detect_drawn_border(values_array)
    return color_map, borders



def closest_and_opposite_interpolation_fill(guide_image_file, result_image_file, batch_stripe_distance=5, region_proportion_threshold=0.01):
    """
    Uses a guide image to blend regions of varying shades of grey together into smooth gradient
    :param guide_image_file: The file name of the guide image
    :param result_image_file: The name of the file to save the image with the applied gradients to
    :param batch_stripe_distance: Used to divide processing into batches
    :param region_proportion_threshold: Used to recognize aliasing between region boundaries: pixels with brightness
    values representing less than the indicated proportion will be moved into the region of brightness values closest
    to those pixels instead
    :return: None
    """
    guide_image = Image.open(guide_image_file)
    guide_image_array = np.array(guide_image)
    color_map, borders = get_region_and_borders(guide_image_array, region_proportion_threshold)
    gradient_array = np.zeros(guide_image_array.shape)
    for value, points in color_map.items():
        for row_offset in range(batch_stripe_distance):
            for column_offset in range(batch_stripe_distance):
                fill_sample_points(
                    value,
                    np.array(points),
                    borders[value],
                    batch_stripe_distance,
                    row_offset,
                    column_offset,
                    gradient_array
                )
    Image.fromarray(gradient_array.astype(np.uint8)).save(result_image_file)


def subdivided_border_tangent_interpolation_fill(
        guide_image_file,
        result_image_file,
        region_defuzz_threshold=20,
        border_defuzz_threshold=5,
        region_proportion_threshold=0.01,
        border_decimate_r_2_threshold=.9):
    guide_image = Image.open(guide_image_file)
    guide_image_array = np.array(guide_image)
    guide_image_shape = guide_image_array.shape[:2]
    color_map = color_mapper(guide_image_array, region_proportion_threshold)
    values_array = np.full(guide_image_shape, -1.0)
    for value, points in color_map.items():
        points_array = np.array(points)
        values_array[points_array[...,0], points_array[...,1]] = value
    regions_points, region_values = identify_regions(values_array, region_defuzz_threshold)
    regions_array = np.full(guide_image_shape, -1.0)
    for region_id, points in enumerate(regions_points):
        regions_array[*points] = region_id - 1
    regions_borders_raw = detect_drawn_border(regions_array)
    regions_borders = dict()
    total_borders = []
    for region_id, border_map in regions_borders_raw.items():
        region_border = dict()
        regions_borders[region_id] = region_border
        for bordering_region_id, border_points in border_map.items():
            split_borders = divide_borders(
                np.stack(border_points, axis=1)
            )
            total_borders.extend(split_borders)
            region_border[bordering_region_id] = split_borders
    start_time = time.time()
    for border in total_borders:
        decimate(border, guide_image_shape, border_decimate_r_2_threshold)
    elapsed = time.time() - start_time
    print(elapsed)
    # color_map, borders = get_region_and_borders(guide_image_array, region_proportion_threshold)
    # for value, points in color_map.items():
    #     border_points = np.concat([value_border for value_border in borders[value].values()]).T
    #     downsample_border(border_points, points, guide_image_array.shape, 5)
