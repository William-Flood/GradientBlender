import numpy as np
from util.grid import Grid, GridCell
from typing import List
from PIL import Image
from scipy.sparse.csgraph import connected_components
from regions.split_region_by_points import split_region
from regions.identify_regions_floodfill import identify_regions


def remove_holes(region_points, values_array):
    test_i = 0
    for points in region_points:
        if values_array[*points[:, 0]] == -1:
            yield points
        else:
            print(f"Checking for holes in {test_i}")
            split_point_set = split_region(points, values_array.shape)
            if len(split_point_set) > 1:
                print(f"Split {test_i}")
                for point_set in split_point_set:
                    yield point_set
            else:
                yield split_point_set[0]
        test_i += 1


def identify_image_regions(values_array, defuzz_threshold):
    print("Creating initial region map")
    regions_points = identify_regions(values_array)
    defuzzed_region_points = []
    for points in remove_holes(regions_points, values_array):
        defuzzed_region_points.append(points)
        # if points.shape[1] > defuzz_threshold:
    # fuzzed_region_points = regions_points
    region_values = [
        values_array[*points[:, 0]]
        for points in defuzzed_region_points
    ]
    return defuzzed_region_points, region_values
