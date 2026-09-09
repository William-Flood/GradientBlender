import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_array


def get_polygon_edges(polygon, vertex_index_lookup):
    polygon_edge_vertices_unsorted = np.stack(
        [
            vertex_index_lookup[*polygon].toarray() - 1,
            np.roll(vertex_index_lookup[*polygon].toarray() - 1, -1, axis=0)
        ], axis=0
    )
    return np.sort(polygon_edge_vertices_unsorted, axis=0)


def map_polygon_list(polygons: list[NDArray]):
    polygon_vertex_coordinates = np.concat(polygons, axis=1)
    polygon_per_polygon_vertex = np.concat([[polygon_id] * polygon.shape[1] for polygon_id, polygon in enumerate(polygons)])
    vertex_list = np.unique(polygon_vertex_coordinates, axis=1)
    vertex_index_lookup = coo_array(
        (np.arange(vertex_list.shape[1]) + 1, vertex_list),
        shape=np.max(vertex_list, axis=1) + 1
    )
    polygon_vertex_indices = vertex_index_lookup[*polygon_vertex_coordinates].toarray() - 1
    vertex_index_and_polygon = np.stack([polygon_vertex_indices, polygon_per_polygon_vertex], axis=0)
    vertex_polygon_lookup = coo_array(
        ([True] * vertex_index_and_polygon.shape[1], vertex_index_and_polygon),
        shape=[vertex_list.shape[1], len(polygons)]
    )
    polygon_edges = np.concat(
        list(map(lambda polygon: get_polygon_edges(polygon, vertex_index_lookup), polygons)),
        axis=1
    )
    polygon_per_polygon_edge = polygon_per_polygon_vertex
    edge_per_polygon_edge_unfiltered = np.concat([
        np.arange(polygon.shape[1]) + 1 for polygon in polygons
    ])
    edge_list = np.unique(polygon_edges, axis=1)
    edge_index_lookup = coo_array(
        (np.arange(edge_list.shape[1]) + 1, edge_list),
        shape=np.max(edge_list, axis=1) + 1
    )
    polygon_edge_indices = edge_index_lookup[*polygon_edges].toarray() - 1
    edge_index_and_polygon_unfiltered = np.stack([polygon_edge_indices, polygon_per_polygon_edge], axis=0)
    # Filtering step needed for the edge case of a degenerate 2-vertex polygon
    edge_index_and_polygon, pair_indices = np.unique(edge_index_and_polygon_unfiltered, return_index=True, axis=1)
    edge_per_polygon_edge = edge_per_polygon_edge_unfiltered[pair_indices]
    edge_polygon_lookup = coo_array(
        (edge_per_polygon_edge, edge_index_and_polygon),
        shape=[edge_list.shape[1], len(polygons)]
    )
    return vertex_index_lookup, vertex_polygon_lookup, edge_index_lookup, edge_polygon_lookup


def find_polygons_on_edge(polygons: list[NDArray], edge):
    (
        vertex_index_lookup, vertex_polygon_lookup, edge_index_lookup, edge_polygon_lookup
    ) = map_polygon_list(polygons)
    if np.greater_equal(np.max(edge, axis=1), vertex_index_lookup.shape).any():
        return [], []
    vertex_indices = vertex_index_lookup[*edge].toarray() - 1
    edge_index = edge_index_lookup[*np.sort(vertex_indices)] - 1
    if edge_index < 0:
        return [], []
    else:
        edge_polygons = edge_polygon_lookup[edge_index].toarray()
        polygon_indices = np.flatnonzero(edge_polygons)
        local_edge_indices = edge_polygons[polygon_indices] - 1
        return polygon_indices, local_edge_indices


def find_neighboring_polygon(polygons: list[NDArray], neighbor_edge_coords, next_edge_coords):
    (
        vertex_index_lookup, vertex_polygon_lookup, edge_index_lookup, edge_polygon_lookup
    ) = map_polygon_list(polygons)
    stacked_coords = np.stack([neighbor_edge_coords, next_edge_coords], axis=2)
    vertex_indices = vertex_index_lookup[*stacked_coords].toarray()
    edge_indices = edge_index_lookup[*np.sort(
        vertex_indices, axis=0
    )].toarray()
    edge_polygons = edge_polygon_lookup[edge_indices].toarray()

    neighbor_edge_polygons = edge_polygons[0] * np.equal(edge_polygons[1], 0)
    polygon_indices = np.flatnonzero(neighbor_edge_polygons)
    local_edge_indices = neighbor_edge_polygons[polygon_indices] - 1
    return polygon_indices, local_edge_indices
