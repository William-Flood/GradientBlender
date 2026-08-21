import numpy as np
from scipy.spatial.distance import cdist
from typing import List
import sys
if "borders.border" not in sys.modules:
    from borders.border import Border


endpoint_selection_matrix = np.array([[0, 0], [0, -1], [-1, 0], [-1, -1]])


def defuzz_loop(loop_borders: List[Border], defuzz_threshold) -> List[Border]:
    new_loop_indexes = [border_index
                        for border_index, border in enumerate(loop_borders)
                        if border.full_points.shape[1] > defuzz_threshold]
    last_index = -1
    for index in new_loop_indexes:
        if index != last_index + 1:
            pinch_index = index - 1
            endpoint_distances = cdist(
                loop_borders[last_index].endpoints.T,
                loop_borders[index].endpoints.T,
                "euclidean")
            min_distance = np.argmin(endpoint_distances)
            endpoint_indices = endpoint_selection_matrix[min_distance]
            pinch_midpoint = np.average(loop_borders[pinch_index].endpoints, axis=1)
            last_endpoints = loop_borders[last_index].endpoints
            last_endpoints[:, endpoint_indices[1]] = pinch_midpoint
            loop_borders[last_index].endpoints = last_endpoints
            current_endpoints = loop_borders[index].endpoints
            current_endpoints[:, endpoint_indices[0]] = pinch_midpoint
            loop_borders[index].endpoints = current_endpoints
            last_index = index
    new_loop = [border for border in loop_borders if border.full_points.shape[1] > defuzz_threshold]
    return new_loop
