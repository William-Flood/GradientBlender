import numpy as np
from numpy.typing import NDArray
import sys
if "borders.border" not in sys.modules:
    from borders.border import Border
from util.inspection_utils import *
import networkx as nx
from networkx.algorithms.approximation import traveling_salesman


def flat_to_node(flat_indices, border_count) -> NDArray:
    return np.array(np.unravel_index(flat_indices, [border_count * 2] * 2))


def node_to_border(node_indices, border_count):
    return np.unravel_index(node_indices, [border_count, 2])


def create_graph(endpoint_distances):
    pixel_graph = nx.Graph()
    border_count = endpoint_distances.shape[2]
    pixel_graph.add_nodes_from(range(border_count * 2))
    flat_distances_full = np.transpose(endpoint_distances, [2, 0, 3, 1]).flatten()
    node_ids_full = flat_to_node(np.arange((border_count ** 2) * 4), border_count)
    is_edge_node_self_connection = np.equal(node_ids_full[0], node_ids_full[1])
    # This removes node self-connections
    flat_distances = flat_distances_full[np.logical_not(is_edge_node_self_connection)]
    node_ids = node_ids_full[:, np.logical_not(is_edge_node_self_connection)]
    node_border_ids = np.reshape(
        node_to_border(node_ids.flatten(), border_count),
        [2, 2, flat_distances.shape[0]] #border/edge, from/to, edge
    )
    nodes_of_same_border = np.nonzero(np.equal(node_border_ids[0, 0], node_border_ids[0, 1]))
    flat_distances[nodes_of_same_border] = 0
    # edges = [(node_1, node_2, {'weight': weight}) for (node_1, node_2), weight in zip(node_ids.T, flat_distances)]
    # pixel_graph.add_edges_from(edges)
    for (node_1, node_2), weight in zip(node_ids.T, flat_distances):
        pixel_graph.add_edge(node_1, node_2, weight=weight)
    return pixel_graph


def traverse(endpoint_distances):
    border_count = endpoint_distances.shape[2]
    endpoint_graph = create_graph(endpoint_distances)
    endpoint_walk = traveling_salesman.greedy_tsp(endpoint_graph, weight='weight')
    endpoint_connections = np.stack(
        [
            endpoint_walk[1:-1],
            endpoint_walk[2:]
        ],
        axis=0
    )
    flattened_border_endpoint_walk = node_to_border(np.array(endpoint_connections).flatten(), border_count)
    border_endpoint_walk_with_self_crossing = np.transpose(
        np.reshape(
            flattened_border_endpoint_walk,
            [2, 2, (border_count * 2 - 1)] #border/endpoint,from/to, edge
        ),
        [2, 0, 1] #edge, border/endpoint, from/to
    )
    border_endpoint_walk = border_endpoint_walk_with_self_crossing[
        np.not_equal(
            border_endpoint_walk_with_self_crossing[:, 0, 0],
            border_endpoint_walk_with_self_crossing[:, 0, 1])
    ]
    check_indices = np.reshape(border_endpoint_walk, [-1, 4]).T[[2,3,0,1]]
    walk_distances = endpoint_distances[*check_indices]
    return border_endpoint_walk

def find_connections(region_borders: list[Border]):
    """
    Given a list of borders in a contiguous graphical, 2-dimensional region with no holes, determine the connectivity
    of the borders
    Args:
        region_borders:

    Returns:

    """
    def get_border_endpoints(border: Border):
        return border.endpoints
    # test_region([549, 1131], np.concat([border.rasterize() for border in region_borders], axis=1))
    border_endpoints = np.stack(list(map(get_border_endpoints, region_borders)), axis=2)
    comparison_order = [[0, 0], [0, 1], [1, 0], [1, 1]]
    endpoint_displacement = np.reshape(
        np.stack(
            [
                border_endpoints[:, id_1, :, np.newaxis] - border_endpoints[:, id_2, np.newaxis, :]
                for id_1, id_2 in comparison_order
            ],
            axis=1
        ),
        [2, 2, 2, len(region_borders), len(region_borders)]
    )
    endpoint_distance = np.linalg.norm(endpoint_displacement, axis=0)
    return traverse(endpoint_distance)


def naive_traversal(region_borders, endpoint_distance):
    """
    Navigates the endpoint graph by selecting the minimum edge distance each time.  Very thin regions produced edge
    cases that this approach was unable to handle
    Args:
        region_borders:
        endpoint_distance:

    Returns:

    """
    endpoint_distance[..., *np.nonzero(np.identity(len(region_borders), dtype=bool))] = None
    min_distances_pair = np.nanmin(endpoint_distance, axis=3, keepdims=True)
    min_distance_endpoint_squeezed = np.min(min_distances_pair, axis=1, keepdims=True)
    min_distance_endpoint = np.broadcast_to(
        np.broadcast_to(
            min_distance_endpoint_squeezed, [2, 2, len(region_borders), 1]
        ),
        [2, 2, len(region_borders), len(region_borders)]
    )
    is_connection = np.equal(endpoint_distance, min_distance_endpoint)
    border_index = 0
    endpoint_index = 0
    border_searched = np.zeros([len(region_borders)], dtype=bool)
    border_searched[border_index] = True
    loop = []
    loop_index = 0
    while not border_searched.all():
        endpoint_connection = np.array(np.nonzero(is_connection[endpoint_index, :, border_index]))
        assert endpoint_connection.shape[1] == 1
        next_border = endpoint_connection[1, 0]
        border_connected_point = endpoint_connection[0, 0]
        loop.append((border_index, endpoint_index, next_border, border_connected_point))
        border_index = int(next_border)
        endpoint_index = (border_connected_point + 1) % 2
        border_searched[border_index] = True
        assert np.sum(border_searched) == loop_index + 2
        loop_index += 1
    return loop
