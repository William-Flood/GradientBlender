from typing import List
from borders.loop import Loop
from borders.find_connections import find_connections
import numpy as np
import sys

if "borders.border" not in sys.modules:
    from borders.border import Border
from util.inspection_utils import *


class Region:
    loop: List[Border]

    def __init__(self, value, region_id, points):
        self.total_borders: List[Border] = []
        self.value = value
        self.id = region_id
        self.points = points
        self.border_orientation = []
        self.polygons = []

    def add_border(self, new_border: Border):
        self.total_borders.append(new_border)

    def create_loops(self, defuzz_threshold):
        borders_with_endpoints = [border for border in self.total_borders if not border.is_loop]
        if len(borders_with_endpoints) > 0:
            # defuzzed_borders = borders_with_endpoints
            defuzzed_borders = [border for border in borders_with_endpoints if border.length > defuzz_threshold]
            if len(defuzzed_borders) == 0:
                self.total_borders = []
            elif len(defuzzed_borders) == 1:
                self.total_borders = defuzzed_borders
                defuzzed_borders[0].make_into_loop()
                self.border_orientation = [0]
            else:
                loop = find_connections(defuzzed_borders)
                new_border_list = []
                for ((border_index, next_border), (endpoint_index, border_connected_point)) in loop:
                    self.border_orientation.append(endpoint_index)
                    new_border_list.append(defuzzed_borders[border_index])
                    defuzzed_borders[border_index].connecting_borders[endpoint_index].append(
                        (defuzzed_borders[next_border], border_connected_point)
                    )
                    defuzzed_borders[next_border].connecting_borders[border_connected_point].append(
                        (defuzzed_borders[border_index], endpoint_index)
                    )
                self.total_borders = new_border_list
        else:
            self.border_orientation = [0]

    def form_initial_polygon(self):
        if len(self.total_borders) > 0:
            border_points = np.concat(
                [
                    np.flip(border.decomposed_points, axis=1) if border_orientation == 0 else border.decomposed_points
                    for border, border_orientation in zip(self.total_borders, self.border_orientation)
                ],
                axis=1
                )
            unique_points, point_first_indices = np.unique(border_points, return_index=True, axis=1)
            self.polygons = [unique_points[:, np.argsort(point_first_indices)]]
        else:
            self.polygons = []

    @property
    def is_void(self):
        return self.value == -1

    def rasterize_polygon(self):
        return np.unique(
            np.concat(
                [np.flip(border.rasterize(), axis=1) if border_orientation == 0 else border.rasterize()
                 for border, border_orientation in zip(self.total_borders, self.border_orientation)],
                axis=1
            ),
            axis=1)
