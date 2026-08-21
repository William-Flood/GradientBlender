import numpy as np
from numpy.typing import NDArray
import sys
if "regions.region" not in sys.modules:
    from regions.region import Region


class Border:
    def __init__(self, border_points: NDArray[np.int32], region_one: Region, region_two: Region):
        self.full_points = border_points
        self.region_one = region_one
        self.region_two = region_two
    decomposed_points: NDArray[np.int32]
    is_loop: bool

    @property
    def endpoints(self):
        return self.decomposed_points[[[0, 1], [0, 1]], [[0, 0], [-1, -1]]]

    @endpoints.setter
    def endpoints(self, new_endpoints):
        self.decomposed_points[[[0, 1], [0, 1]], [[0, 0], [-1, -1]]] = new_endpoints

    def get_offset(self, offset):
        endpoint_vector = self.decomposed_points[:, 0] - self.decomposed_points[:, -1]
        if endpoint_vector[1] < 0:
            endpoint_vector = endpoint_vector * -1
        border_normal_unsized = np.array([endpoint_vector[1], endpoint_vector[0] * -1])
        border_normal = border_normal_unsized / np.linalg.norm(border_normal_unsized)
        return (self.decomposed_points + border_normal[:, np.newaxis] * offset).astype(np.int32)

    def collide_line_segments(self, segments):
        # Adapted from https://www.jeffreythompson.org/collision-detection/line-line.php
        x1 = self.decomposed_points[1, :-1]
        y1 = self.decomposed_points[0, :-1]
        y2 = self.decomposed_points[0, 1:]
        x2 = self.decomposed_points[1, 1:]
        x3 = segments[1, :-1]
        y3 = segments[0, :-1]
        y4 = segments[0, 1:]
        x4 = segments[1, 1:]

        uA = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
        uB = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))

        uA_in_range = np.logical_and(np.greater_equal(uA, 0), np.less_equal(uA, 1))
        uB_in_range = np.logical_and(np.greater_equal(uB, 0), np.less_equal(uB, 1))
        return np.logical_and(uA_in_range, uB_in_range)