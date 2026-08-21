import numpy as np
from scipy.sparse.csgraph import connected_components, depth_first_order
from scipy.spatial.distance import cdist
from typing import List
import sys
from borders.loop import Loop
if "borders.border" not in sys.modules:
    from borders.border import Border


def find_loops(region_borders: List[Border]):
    def get_border_endpoints(border: Border):
        return border.endpoints
    border_endpoints = np.concat(list(map(get_border_endpoints, region_borders)), axis=1)
    endpoint_distances_unfiltered = cdist(border_endpoints.T, border_endpoints.T, metric='euclidean')
    border_ids_along_graph_side = np.concat([[border_id, border_id] for border_id in range(len(region_borders))])
    border_ids_per_point_unfiltered = np.stack(
        np.broadcast_arrays(
            border_ids_along_graph_side[:, np.newaxis],
            border_ids_along_graph_side[np.newaxis, :]
        ), axis=2
    )
    keep_cell = np.logical_not(
        np.equal(
            border_ids_per_point_unfiltered[..., 0],
            border_ids_per_point_unfiltered[..., 1]
        )
    )
    keep_indices_flat = np.nonzero(keep_cell)[1]
    keep_indices = np.reshape(keep_indices_flat, [len(region_borders) * 2, len(region_borders) * 2 - 2])
    border_column_ids_per_point = np.take_along_axis(
        border_ids_per_point_unfiltered,
        keep_indices[:, :, np.newaxis],
        axis=1
    )
    endpoint_distances = np.take_along_axis(
        endpoint_distances_unfiltered,
        keep_indices,
        axis=1
    )
    connections = np.argmin(endpoint_distances, axis=1)
    border_connections = border_column_ids_per_point[np.arange(connections.shape[0]), connections, :]
    border_connection_graph = np.zeros([len(region_borders), len(region_borders)], dtype=np.bool)
    border_connection_graph[border_connections] = True
    num_loops, loop_assignments = connected_components(border_connection_graph)
    loop_start_ids = map(lambda label_id: np.argmax(np.equal(loop_assignments, label_id)), range(num_loops))
    loops_ids = [
        depth_first_order(border_connection_graph, int(start_id), return_predecessors=False)
        for start_id in loop_start_ids
    ]
    return [Loop([region_borders[loop_id] for loop_id in loop_ids]) for loop_ids in loops_ids]
