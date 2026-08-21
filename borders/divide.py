import numpy as np
from scipy.sparse.csgraph import connected_components
from borders.border import Border
from typing import List
from regions.region import Region
from borders.order_points import order_points
from util.inspection_utils import *


def make_border_graph(border_points):
    connecting_offsets = np.array(
        [[row_offset, column_offset] for column_offset in [-1, 0, 1] for row_offset in [-1, 0, 1]]
    ).T
    border_points_and_connections = border_points[:, np.newaxis, :] + connecting_offsets[:, :, np.newaxis]
    compare_points = np.repeat(border_points[:, np.newaxis, :], 9, axis=1)
    point_comparison = np.equal(border_points_and_connections[:, :, :, np.newaxis], compare_points[:, :, np.newaxis, :])
    border_graph = np.logical_and(
        np.any(
            np.all(
                point_comparison, axis=0
            ),
            axis=0
        ),
        np.logical_not(
            np.identity(border_points.shape[1]), dtype=np.bool
        )
    )
    return border_graph


def divide_borders(border_points, region_one: Region, region_two: Region, region_map) -> List[Border]:
    border_graph = make_border_graph(border_points)
    n_components, labels = connected_components(border_graph)
    split_borders = []
    for split in range(n_components):
        border_segment = border_points[:,
                             np.argwhere(labels == split)[:, 0]
                             ]
        border_segment_filtered = np.unique(border_segment, axis=1)
        if border_segment_filtered.shape[1] > 2:
            segment_ordered, is_loop = order_points(border_segment_filtered, region_map)
            border = Border(segment_ordered, region_one, region_two)
            border.is_loop = is_loop
        else:
            border = Border(border_segment_filtered, region_one, region_two)
            border.is_loop = False
        split_borders.append(border)
    return split_borders
