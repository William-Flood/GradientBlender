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
        if values_array[*points[:, 0]] == -1 or True:
        #if values_array[*points[:, 0]] == -1:
            yield points
        else:
            print(f"Checking for holes in {test_i}")
            split_point_set = split_region(points, values_array.shape)
            if len(split_point_set) > 1:
                print(f"Splitting {test_i}")
                for point_set in split_point_set:
                    split_values_array = np.zeros(values_array.shape, dtype=np.bool)
                    split_values_array[*point_set] = True
                    connected_components_in_set = identify_regions(split_values_array)
                    for sub_split in connected_components_in_set:
                        if split_values_array[*sub_split[:, 0]]:
                            yield sub_split
            else:
                yield split_point_set[0]
        test_i += 1


def identify_image_regions(values_array, defuzz_threshold):
    print("Creating initial region map")
    regions_points = identify_regions(values_array)
    # defuzzed_region_points = []
    # for points in remove_holes(regions_points, values_array):
    #     if points.shape[1] > defuzz_threshold:
    #         defuzzed_region_points.append(points)
    defuzzed_region_points = regions_points
    region_values = [
        values_array[*points[:, 0]]
        for points in defuzzed_region_points
    ]
    return defuzzed_region_points, region_values
