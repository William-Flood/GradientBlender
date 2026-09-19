import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_array


def get_polygon_edge_vertex_indices(polygon, vertex_index_lookup):
    polygon_edge_vertices_unsorted = np.stack(
        [
            vertex_index_lookup[*polygon].toarray() - 1,
            np.roll(vertex_index_lookup[*polygon].toarray() - 1, -1, axis=0)
        ], axis=0
    )
    return np.sort(polygon_edge_vertices_unsorted, axis=0)


class Mesh:
    def __init__(self, polygons):
        polygon_vertex_coordinates = np.concat(polygons, axis=1)
        self.polygon_per_polygon_vertex = np.concat(
            [[polygon_id] * polygon.shape[1] for polygon_id, polygon in enumerate(polygons)])
        self.vertex_list = np.unique(polygon_vertex_coordinates, axis=1)
        self.polygon_vertex_indices = self.get_indices(polygon_vertex_coordinates)
        polygon_edge_vertex_indices = np.concat(
            list(map(lambda polygon: get_polygon_edge_vertex_indices(polygon, self.vertex_index_lookup), polygons)),
            axis=1
        )
        polygon_per_polygon_edge = self.polygon_per_polygon_vertex
        edge_per_polygon_edge_unfiltered = np.concat([
            np.arange(polygon.shape[1]) + 1 for polygon in polygons
        ])
        self.edge_list = np.unique(polygon_edge_vertex_indices, axis=1)
        polygon_edge_indices = self.edge_index_lookup[*polygon_edge_vertex_indices].toarray() - 1
        edge_index_and_polygon_unfiltered = np.stack([polygon_edge_indices, polygon_per_polygon_edge], axis=0)
        # Filtering step needed for the edge case of a degenerate 2-vertex polygon
        edge_index_and_polygon_unsorted, pair_indices_unsorted = np.unique(
            edge_index_and_polygon_unfiltered, return_index=True, axis=1
        )
        polygon_ordering = np.argsort(edge_index_and_polygon_unsorted[1], axis=0)
        self.edge_index_and_polygon = edge_index_and_polygon_unsorted[:, polygon_ordering]
        pair_indices = pair_indices_unsorted[polygon_ordering]
        self.edge_per_polygon_edge = edge_per_polygon_edge_unfiltered[pair_indices]
        self.last_id = len(polygons) - 1
        self.polygon_ids = np.arange(len(polygons))
        # edge_polygon_lookup = coo_array(
        #     (self.edge_per_polygon_edge, self.edge_index_and_polygon),
        #     shape=[self.edge_list.shape[1], len(polygons)]
        # )


    def get_indices(self, points):
        vertex_index_lookup =  coo_array(
            (np.arange(self.vertex_list.shape[1]) + 1, self.vertex_list),
            shape=np.max(
                np.concat([self.vertex_list, points], axis=1),
                axis=1) + 1
        )
        return vertex_index_lookup[*points].toarray() - 1

    @property
    def vertex_index_lookup(self):
        return coo_array(
            (np.arange(self.vertex_list.shape[1]) + 1, self.vertex_list),
            shape=np.max(self.vertex_list, axis=1) + 1
        )

    @property
    def polygon_storage_indices(self):
        indices = np.arange(self.last_id + 1)
        indices[self.polygon_ids] = np.arange(self.polygon_ids.shape[0])
        return indices


    @property
    def vertex_polygon_lookup(self):
        vertex_index_and_polygon = np.stack([
            self.polygon_vertex_indices,
            self.polygon_storage_indices[self.polygon_per_polygon_vertex]
        ], axis=0)

        return coo_array(
            ([True] * vertex_index_and_polygon.shape[1], vertex_index_and_polygon),
            shape=[self.vertex_list.shape[1], self.polygon_ids.size]
        )

    @property
    def edge_polygon_lookup(self):
        edge_index_and_polygon_storage_id = np.stack([
            self.edge_index_and_polygon[0],
            self.polygon_storage_indices[self.edge_index_and_polygon[1]]
        ], axis=0)
        return coo_array(
            (self.edge_per_polygon_edge, edge_index_and_polygon_storage_id),
            shape=[self.edge_list.shape[1], self.polygon_ids.size]
        )

    @property
    def edge_index_lookup(self):
        return coo_array(
            (np.arange(self.edge_list.shape[1]) + 1, self.edge_list),
            shape=np.max(self.edge_list, axis=1) + 1
        )

    def get_edge_indices(self, edge_vertex_indices):
        edge_index_lookup = coo_array(
            (np.arange(self.edge_list.shape[1]) + 1, self.edge_list),
            shape=np.max(
                np.concat([self.edge_list, edge_vertex_indices],axis=1),
                axis=1) + 1
        )
        return edge_index_lookup[*edge_vertex_indices].toarray() - 1

    def add_polygon(self, polygon):
        polygon_vertex_indices = self.get_indices(polygon)
        new_point_polygon_indices = np.less(polygon_vertex_indices, 0)
        new_vertices, vertex_index = np.unique(polygon[:, new_point_polygon_indices], axis=1, return_inverse=True)
        next_vertex_index = self.vertex_list.shape[1]
        self.vertex_list = np.concat([self.vertex_list, new_vertices], axis=1)
        polygon_vertex_indices[new_point_polygon_indices] = vertex_index + next_vertex_index
        self.polygon_vertex_indices = np.concat([self.polygon_vertex_indices, polygon_vertex_indices])
        this_polygon_id = self.last_id + 1
        self.polygon_per_polygon_vertex = np.concat(
            [self.polygon_per_polygon_vertex, [this_polygon_id] * polygon.shape[1]]
        )
        polygon_edge_vertex_indices = get_polygon_edge_vertex_indices(polygon, self.vertex_index_lookup)
        polygon_edge_indices = self.get_edge_indices(polygon_edge_vertex_indices)
        new_edge_polygon_indices = np.less(polygon_edge_indices, 0)
        new_edges = polygon_edge_vertex_indices[:, new_edge_polygon_indices]
        next_edge_index = self.edge_list.shape[1]
        self.edge_list = np.concat([self.edge_list, new_edges], axis=1)
        polygon_edge_indices[new_edge_polygon_indices] = np.arange(np.sum(new_edge_polygon_indices)) + next_edge_index
        self.edge_index_and_polygon = np.concat([
            self.edge_index_and_polygon,
            np.stack([polygon_edge_indices, [this_polygon_id] * polygon.shape[1]], axis=0)
        ], axis=1)
        assert np.max(self.edge_index_and_polygon) <= self.edge_list.shape[1] - 1
        self.edge_per_polygon_edge = np.concat(
            [
                self.edge_per_polygon_edge, np.arange(polygon.shape[1]) + 1
            ]
        )
        self.last_id = this_polygon_id
        self.polygon_ids = np.concat([self.polygon_ids, [this_polygon_id]])
        return this_polygon_id


    def remove_polygon(self, polygon_id):
        polygon_polygon_vertices_start = np.searchsorted(
            self.polygon_per_polygon_vertex,
            polygon_id,
            side="left"
        )
        polygon_polygon_vertices_end = np.searchsorted(
            self.polygon_per_polygon_vertex,
            polygon_id,
            side="right"
        )
        self.polygon_vertex_indices = np.delete(
            self.polygon_vertex_indices,
            np.arange(polygon_polygon_vertices_start, polygon_polygon_vertices_end)
        )
        self.polygon_per_polygon_vertex = np.delete(
            self.polygon_per_polygon_vertex,
            np.arange(polygon_polygon_vertices_start, polygon_polygon_vertices_end)
        )
        polygon_edge_index_and_polygon_start = np.searchsorted(
            self.edge_index_and_polygon[1],
            polygon_id,
            side="left"
        )
        polygon_edge_index_and_polygon_end = np.searchsorted(
            self.edge_index_and_polygon[1],
            polygon_id,
            side="right"
        )
        self.edge_index_and_polygon = np.delete(
            self.edge_index_and_polygon,
            np.arange(polygon_edge_index_and_polygon_start, polygon_edge_index_and_polygon_end),
            axis=1
        )
        self.edge_per_polygon_edge = np.delete(
            self.edge_per_polygon_edge,
            np.arange(polygon_edge_index_and_polygon_start, polygon_edge_index_and_polygon_end)
        )
        self.polygon_ids = self.polygon_ids[np.not_equal(self.polygon_ids, polygon_id)]

    def find_polygon_by_points(self, points):
        if np.any(np.greater(
                points,
            np.max(self.vertex_list, axis=1, keepdims=True)
        )):
            return -1
        vertex_indices = self.get_indices(points)
        if np.equal(vertex_indices, -1).any():
            return -1
        vertex_polygon_mask = self.vertex_polygon_lookup[vertex_indices].toarray()
        polygon_mask = np.all(vertex_polygon_mask, axis=0)
        if polygon_mask.any():
            return self.polygon_ids[np.argmax(polygon_mask)]
        else:
            return -1

    def find_polygon_by_edge(self, edge_points):
        if np.any(np.greater(
                edge_points,
            np.max(self.vertex_list, axis=1, keepdims=True)
        )):
            return -1, None
        vertex_indices = self.get_indices(edge_points)
        if np.equal(vertex_indices, -1).any():
            return -1, None
        edge_index = self.get_edge_indices(np.sort(vertex_indices[:, np.newaxis]))[0]
        if np.equal(edge_index, -1).any():
            return -1, None
        edge_polygon_row = self.edge_polygon_lookup[edge_index].toarray()
        if not np.greater(edge_polygon_row, 0).any():
            return -1, None
        polygon_storage_index = np.argmax(edge_polygon_row)
        polygon_index = self.polygon_ids[polygon_storage_index]
        polygon_edge_index = edge_polygon_row[polygon_storage_index] - 1
        return polygon_index, polygon_edge_index
