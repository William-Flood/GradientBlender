import numpy as np


offset_matrix = np.array([
    [[1,  0, -1, 0]],
    [[0,  1, 0, -1]]
])


def get_neighbor_regions(
        region_points,
        region_array
    ):
    padded_region_array = np.pad(region_array, [[1, 1], [1, 1]], mode="constant", constant_values=-1)
    adjusted_region_point_neighbors = region_points[:, :, np.newaxis] + offset_matrix[:, :] + 1
    return padded_region_array[*adjusted_region_point_neighbors]


def find_borders(region_array):
    region_points = np.reshape(np.indices(region_array.shape), [2, -1])
    region_points_values = region_array.flatten()
    neighbor_regions = get_neighbor_regions(
        region_points,
        region_array
    )
    is_border_with = np.not_equal(region_points_values[:, np.newaxis], neighbor_regions)
    is_border = np.any(is_border_with, axis=1)
    return (
        region_points[:, is_border],
        region_points_values[is_border],
        neighbor_regions[is_border]
    )


def is_border_for(
        point_regions,
        region_values,
        border_neighbor_regions,
        region_one_id,
        region_two_id):
    if -1 in region_values[[region_one_id, region_two_id]]:
        if region_values[region_one_id] == -1:
            min_region_id = region_one_id
            max_region_id = region_two_id
        else:
            min_region_id = region_two_id
            max_region_id = region_one_id
    else:
        min_region_id = min(region_one_id, region_two_id)
        max_region_id = max(region_one_id, region_two_id)
    return np.equal(point_regions, max_region_id) & np.any(
        np.equal(border_neighbor_regions, min_region_id),
        axis=1
    )


def detect_drawn_border(regions_array, values_array):
    """
    Identifies the boundaries of shaded regions drawn in the image
    :param regions_array: Indicates contiguous regions of uniform pixel brightness values, as an hxw numpy array
    :param values_array: Indicates the brightness value of each pixel
    :return: A dictionary using integer region ids as keys, and a nested dictionary as
    values. The nested dictionary uses region ids of the bordering values as keys, and point coordinates as values
    """
    region_values = values_array.flatten()[
        np.unique(regions_array.flatten(), return_index=True)[1]
    ]
    border_points, point_regions, border_neighbor_regions = find_borders(regions_array)
    region_ids = np.unique(regions_array)
    border_tree = dict(
        (
            region_id, dict(
                (
                    other_id,
                    np.unique(border_points[
                        :,
                        is_border_for(point_regions, region_values, border_neighbor_regions, region_id, other_id)
                    ], axis=1)
                ) for other_id in [*region_ids, -1] if
                other_id != region_id and
                np.any(is_border_for(point_regions, region_values, border_neighbor_regions, region_id, other_id))
            )
        ) for region_id in region_ids
    )
    return border_tree
