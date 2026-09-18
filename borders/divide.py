import numpy as np
from scipy.sparse.csgraph import connected_components
from borders.border import Border
from typing import List
from regions.region import Region
# from borders.order_points import order_points
from util.inspection_utils import *
from borders.connection_filter import ConnectionFilter
from time import time
from util.point_neighborhood import get_neighbor_values, get_neighbor_values_over_array, get_neighbors
from scipy.sparse import coo_array
from borders.get_pixel_connections import get_pixel_connections
from gradengc import order_border_points

def get_border_connections(
        border_points,
        region_map,
        other_region_id
):
    # border_point_adjacency, _ = connection_filter.get_filtered_connections(border_points, region_map.shape)
    border_point_adjacency = get_pixel_connections(border_points, region_map.shape)
    point_neighbors = get_neighbors(border_points)
    is_point_neighbor_bordering_region = get_neighbor_values_over_array(
        np.equal(region_map, other_region_id),
        border_points
    )
    adjacent_points_neighbors = point_neighbors[:, border_point_adjacency]
    does_adjacent_points_neighbors_match = np.all(
        np.equal(
            adjacent_points_neighbors[:, 0, :, :, np.newaxis],
            adjacent_points_neighbors[:, 1, :, np.newaxis, :]
        ),
        axis=0
    )
    is_adjacent_points_neighbor_bordering_region = is_point_neighbor_bordering_region[border_point_adjacency[0]]
    do_adjacent_points_share_border_point = np.any(
        np.reshape(
            does_adjacent_points_neighbors_match & is_adjacent_points_neighbor_bordering_region[:, :, np.newaxis],
            [-1, 64]
        ),
        axis=1
    )
    return border_point_adjacency[:, do_adjacent_points_share_border_point]



def divide_borders(
        border_points,
        region_one: Region,
        region_two: Region,
        region_map
) -> List[Border]:
    border_connections = get_border_connections(
            border_points,
            region_map,
            region_two.id
    )
    border_graph = coo_array(
        ([True] * border_connections.shape[1], border_connections),
        shape=[border_points.shape[1]] * 2
    ).toarray()
    n_components, labels = connected_components(border_graph, directed=False)
    split_borders = []
    times = []
    for split in range(n_components):
        loop_start = time()
        border_segment = border_points[:,
                             np.argwhere(labels == split)[:, 0]
                             ]
        segment_ordered = order_border_points(
            np.ascontiguousarray(border_segment).astype(np.int32),
            np.ascontiguousarray(region_map).astype(np.int32)
        ).T
        border = Border(segment_ordered, region_one, region_two)
        segment_end_diffs = segment_ordered[:, 0] - segment_ordered[:, -1]
        border.is_loop = np.isin(segment_end_diffs, [-1, 0, 1]).all()
        # border_segment_filtered = np.unique(border_segment, axis=1)
        # if border_segment_filtered.shape[1] > 2:
        #     point_is_segment = np.equal(labels, split)
        #     full_to_segment_map = coo_array(
        #         (np.arange(np.sum(point_is_segment)), [np.flatnonzero(point_is_segment)]),
        #         shape=[border_points.shape[1]]
        #     ).toarray()
        #     segment_connections = full_to_segment_map[
        #             border_connections[:, point_is_segment[border_connections[0]]]
        #     ]
        #     # segment_ordered, is_loop = order_points(border_segment_filtered, region_map, segment_connections)
        # else:
        #     border = Border(border_segment_filtered, region_one, region_two)
        #     border.is_loop = False
        split_borders.append(border)
        times.append(time() - loop_start)
    return split_borders
