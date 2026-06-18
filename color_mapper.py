import numpy as np
from collections import defaultdict


def color_mapper(image_array, region_proportion_threshold):
    """
    Divides the image array into regions based on brightness values
    :param image_array: The guide image to split into regions
    :param region_proportion_threshold: Used to recognize aliasing between region boundaries: pixels with brightness
    values representing less than the indicated proportion will be moved into the region of brightness values closest
    to those pixels instead
    :return:
    """
    points_by_value_with_default = defaultdict(list)
    values_and_alpha_array = np.stack(
        [
            np.average(image_array[...,:3], axis=2),
            image_array[...,3]
         ], axis=2
    )
    image_height, image_width, _ = image_array.shape
    image_index_rect = np.transpose(
        np.indices((image_height, image_width)),
        [1, 2, 0]
    )
    values_alphas_and_indices = np.concat((values_and_alpha_array, image_index_rect), axis=2)
    values_alphas_and_indices_flat = np.reshape(values_alphas_and_indices, [image_height * image_width, 4])
    values_alphas_and_indices_flat_filtered = values_alphas_and_indices_flat[values_alphas_and_indices_flat[...,1] > 0]
    values_and_indices_flat = values_alphas_and_indices_flat_filtered[...,[0,2,3]]
    for pixel in values_and_indices_flat:
        if pixel[1] == 0:
            continue
        value = pixel[0]
        points_by_value_with_default[value].append(pixel[1:].astype(np.int32))
    total_filled_pixels = np.sum(image_array[...,3] > 0)
    count_threshhold = total_filled_pixels * region_proportion_threshold
    points_by_value = dict(points_by_value_with_default)
    values = list(points_by_value.keys())
    for value in values:
        points = points_by_value[value]
        if len(points) < count_threshhold:
            current_other_values = np.array([other_value for other_value in points_by_value.keys() if other_value != value])
            distances_squared = np.square(current_other_values - value)
            closest_other_value = current_other_values[np.argmin(distances_squared)]
            points_by_value[closest_other_value].extend(points)
            del points_by_value[value]

    return points_by_value