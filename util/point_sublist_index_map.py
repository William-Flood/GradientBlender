import numpy as np
from scipy.sparse import csr_array


def point_sublist_index_map(point_list, image_shape, fill=-1):
    """
    Returns a tensor based on a nested list of point coordinates, which the value at each position equaling the
    location of the list inside point_list in which that coordinate appears
    Args:
        point_list: A nested list of point coordinates
        image_shape: The shape of the tensor to fill
        fill: The value to fill in coordinates that do not appear inside point_list

    Returns:

    """
    point_sublist_indices = np.concat([[sublist_index] * points.shape[1]
                                       for sublist_index, points in enumerate(point_list)]
                                      )
    full_point_list = np.concat(point_list, axis=1)
    results = np.full(image_shape, fill)
    results[*full_point_list] = point_sublist_indices
    return results
