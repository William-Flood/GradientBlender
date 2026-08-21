import numpy as np
from numpy.typing import NDArray
from borders.border import Border


def get_pixel_connections(points: NDArray, image_shape):
    point_index_map = np.full(np.array(image_shape[:2]) + 2, -1, dtype=np.int32)
    point_index_map[*(points + 1)] = np.arange(points.shape[1])
    connecting_offsets = np.array(
        [[row_offset, column_offset] for column_offset in [-1, 0, 1] for row_offset in [-1, 0, 1]
         if row_offset != 0 or column_offset != 0]
    ).T
    border_points_and_connections = points[:, :, np.newaxis] + connecting_offsets[:, np.newaxis, :]
    point_connections = point_index_map[*(border_points_and_connections + 1)]
    point_and_connections_bundled = np.stack(
        (np.repeat(np.arange(points.shape[1])[:, np.newaxis], 8, axis=1),
         point_connections),
        axis=0
    )
    point_and_connections = np.reshape(point_and_connections_bundled, [2, -1])
    connection_not_on_point = np.not_equal(point_and_connections[0], point_and_connections[1])
    connection_to_valid = np.not_equal(point_and_connections[1], -1)
    connection_valid = connection_not_on_point & connection_to_valid
    return point_and_connections[:, connection_valid]
