import numpy as np


def get_point_sample(points, row_range, column_range):
    points_at_or_after_start = (points[0] >= row_range[0]) & (points[1] >= column_range[0])
    points_before_end = (points[0] < row_range[1]) & (points[1] < column_range[1])
    points_in_range_bool = points_at_or_after_start & points_before_end
    points_in_range = points[:, points_in_range_bool]
    return points_in_range, np.flatnonzero(points_in_range_bool)


def get_connection_indices(points, point_indices):
    point_deviances = points[:, :, np.newaxis] - points[:, np.newaxis, :]
    projected_adjacent = (point_deviances >= -1) & (point_deviances <= 1)
    adjacent = np.all(projected_adjacent, axis=0) & np.logical_not(np.eye(points.shape[1], dtype=np.bool))
    adjacent_indices_matrix = np.stack(
        np.nonzero(adjacent),
        axis=0
    )
    return point_indices[adjacent_indices_matrix]


def create_point_connection_graph(points, image_shape):
    connection_graph = np.zeros([points.shape[1], points.shape[1]], dtype=np.bool)
    shape_sqrts = np.sqrt(image_shape).astype(int)
    row_starts = np.arange(0, image_shape[0], shape_sqrts[0])
    column_starts = np.arange(0, image_shape[1], shape_sqrts[1])
    row_ends = [*row_starts[1:] + 1, image_shape[0]]
    column_ends = [*column_starts[1:] + 1, image_shape[1]]
    for row_range in zip(row_starts, row_ends):
        for column_range in zip(column_starts, column_ends):
            points_in_range, point_indices = get_point_sample(points, row_range, column_range)
            if points_in_range.shape[1] > 0:
                connection_graph[get_connection_indices(points_in_range, point_indices)] = True
    return connection_graph
