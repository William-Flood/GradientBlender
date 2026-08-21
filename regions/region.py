from typing import List
from borders.loop import Loop
from borders.find_loops import find_loops
import numpy as np
import sys
from regions.split_region_by_border import split_region
from image.validate import validate
if "borders.border" not in sys.modules:
    from borders.border import Border


class Region:
    loops: List[Loop]

    def __init__(self, value, id, points):
        self.total_borders: List[Border] = []
        self.value = value
        self.id = id
        self.points = points

    def add_border(self, new_border: Border):
        self.total_borders.append(new_border)

    def create_loops(self, defuzz_threshold):
        borders_with_endpoints = [border for border in self.total_borders if not border.is_loop]
        if len(borders_with_endpoints) > 0:
            unordered_loops = [
                *find_loops(borders_with_endpoints),
                *[Loop([border]) for border in self.total_borders if border.is_loop]
            ]
        else:
            unordered_loops = [Loop([border]) for border in self.total_borders]
        if len(unordered_loops) == 1:
            self.loops = unordered_loops
        else:
            loop_areas = np.array(loop.approximate_area for loop in unordered_loops)
            outer_loop = np.argmax(loop_areas)
            self.loops = [unordered_loops[outer_loop], *unordered_loops[:outer_loop], *unordered_loops[outer_loop+1:]]

    def get_offset_points(self, offset_amount):
        candidate_points = self.loops[0].get_offset(offset_amount)
        point_compare = np.equal(
            candidate_points.T[:, np.newaxis, :],
            self.points.T[np.newaxis, :, :]
        )
        candidate_point_matches = np.all(point_compare, axis=2)
        candidate_in_region = np.any(candidate_point_matches, axis=1)
        return candidate_points[candidate_in_region]

    def create_splits(self):
        if len(self.loops) == 1:
            return None
        else:
            return split_region(self.loops)

    @property
    def is_void(self):
        return self.value == -1
