from typing import List
from borders.defuzz_loop import defuzz_loop
import sys
import numpy as np
from scipy.spatial.distance import cdist
if "borders.border" not in sys.modules:
    from borders.border import Border


class Loop:
    def __init__(self, borders: List[Border]):
        self.borders = borders
        self._approxiamte_area = None
        self.add_or_subtract_adjust = 1

    def defuzz(self, defuzz_threshold):
        self.borders = defuzz_loop(self.borders, defuzz_threshold)
        self._approxiamte_area = None

    @property
    def approximate_area(self):
        if self._approxiamte_area is None:
            if len(self.borders) == 0:
                self._approxiamte_area = 0
            else:
                border_endpoints = np.stack([border.endpoints for border in self.borders], axis=2)
                border_widths = np.abs(border_endpoints[1, 0, :] - border_endpoints[1, 1, :])
                border_min_heights = np.min(border_endpoints[0, :, :], axis=1)
                border_max_heights = np.max(border_endpoints[0, :, :], axis=1)
                areas = np.prod(border_widths, (border_min_heights + border_max_heights) / 2)
                area = np.sum(
                    np.prod(areas, self.add_or_subtract_array)
                )
                if area < 0:
                    self._approxiamte_area = area * -1
                    self.add_or_subtract_adjust = -1
                else:
                    self._approxiamte_area = area
        return self._approxiamte_area

    @property
    def add_or_subtract_array(self):
        border_endpoint_horizontals = np.stack([border.endpoints[1] for border in self.borders], axis=1)
        border_average_horizontals = np.average(border_endpoint_horizontals, axis=0)
        border_average_horizontal_compare = np.roll(border_average_horizontals, 1, axis=0)
        return np.less(border_average_horizontals, border_average_horizontal_compare) * self.add_or_subtract_adjust

    def get_offset(self, offset_amount):
        offset_with_sign = offset_amount * -1 * self.add_or_subtract_array
        return np.concat([border.get_offset(border_offset)
                          for border, border_offset in zip(self.borders, offset_with_sign)], axis=1)

    @property
    def rect_points(self):
        total_points = np.concat([border.decomposed_points for border in self.borders], axis=1)
        mins = np.min(total_points, axis=1)
        maxes = np.max(total_points, axis=1)
        bounding_rect = np.array([[mins[0], mins[1]], [maxes[0], maxes[1]], [mins[0], maxes[1]], [maxes[0], mins[1]]])
        distances_from_rect_verts = cdist(bounding_rect, total_points, metric='euclidean')
        point_selection = np.argmin(distances_from_rect_verts, axis=1)
        return total_points[:, point_selection]

    @property
    def points(self):
        return np.concat([border.decomposed_points for border in self.borders], axis=1)
