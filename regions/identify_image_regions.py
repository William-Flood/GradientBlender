from util.point_neighborhood import get_neighbor_values_over_array
import numpy as np
from regions.split_region_by_points import split_region
from regions.identify_regions_floodfill import identify_regions
from scipy.sparse import coo_array
from numpy.typing import NDArray
from regions.region import Region


def check_inner_loops(region_points, image_shape):
    region_map = coo_array(([True] * region_points.shape[1], region_points), shape=image_shape).toarray()
    outside_points = np.array(np.nonzero(np.logical_not(region_map)))
    filtered_outside_points = outside_points[
        :,
        np.all(
            np.greater_equal(outside_points, np.min(region_points, axis=1, keepdims=True) - 1) &
            np.less_equal(outside_points, np.max(region_points, axis=1, keepdims=True) + 1),
            axis=0
        )
    ]
    border_matrix = coo_array(
        ([True] * filtered_outside_points.shape[1], filtered_outside_points),
        shape=image_shape).toarray()
    border_regions = identify_regions(border_matrix, border_matrix)
    return len(border_regions) > 1


def remove_holes(region_points, values_array):
    test_i = 0
    for points in region_points:
        if values_array[*points[:, 0]] == -1:
            yield points
        else:
            print(f"Checking for holes in {test_i}")
            if check_inner_loops(points, values_array.shape):
                print(f"Splitting {test_i}")
                split_point_set = split_region(points, values_array.shape)
                if len(split_point_set) > 1:
                    for point_set in split_point_set:
                        yield point_set
                else:
                    yield split_point_set[0]
            else:
                yield points
        test_i += 1

def find_region_to_merge(region_points, min_region, region_select_index):
    region_map = coo_array(
        (
            np.concat([[region_id] * region.shape[1] for region_id, region in enumerate(region_points)]),
            np.concat(region_points, axis=1)
        )
    ).toarray()
    fuzz_region_neighbors = get_neighbor_values_over_array(region_map, min_region)
    return fuzz_region_neighbors.flatten()[region_select_index]


def defuzz_regions(region_points: list[NDArray], values_array, defuzz_threshold):
    region_sizes = [region.shape[1] for region in region_points]
    if min(region_sizes) <= defuzz_threshold:
        min_index = np.argmin(region_sizes)
        min_region = region_points[min_index]
        region_value = values_array[*min_region[:, 0]]
        region_values_search = np.copy(get_neighbor_values_over_array(values_array, min_region)).flatten()
        neighbor_to_merge_into_candidates = np.equal(region_values_search, region_value) | \
            np.equal(region_values_search, -1)
        if not neighbor_to_merge_into_candidates.any():
            neighbor_to_merge_into_candidates = np.equal(region_values_search, region_value)
        region_values_search[neighbor_to_merge_into_candidates] = None
        region_select_index = np.nanargmin(np.absolute(region_values_search - region_value))
        region_to_merge_into = find_region_to_merge(region_points, min_region, region_select_index)
        new_list = region_points.copy()
        new_list[region_to_merge_into] = np.concat([region_points[region_to_merge_into], min_region], axis=1)
        del new_list[min_index]
        return defuzz_regions(new_list, values_array, defuzz_threshold)
    else:
        return region_points


def get_value_mode(values_array, points):
    unique_values, counts = np.unique(values_array[*points], return_counts=True)
    return unique_values[np.argmax(counts)]


def identify_image_regions(values_array, defuzz_threshold):
    print("Creating initial region map")
    regions_points = identify_regions(values_array)
    defuzzed_regions_points = defuzz_regions(regions_points, values_array, defuzz_threshold)
    assert np.all(coo_array(
        (
            [True] * sum([points.shape[1] for points in defuzzed_regions_points]),
            np.concat(defuzzed_regions_points, axis=1)
         ), shape=values_array.shape
    ).toarray())
    split_region_points = []
    for points in remove_holes(defuzzed_regions_points, values_array):
        split_region_points.append(points)
    return [
                Region(get_value_mode(values_array, points), region_index, points)
                for region_index, points in enumerate(split_region_points)
           ]
