import numpy as np
from PIL import Image

def matrix_to_csv(matrix):
    return "\n".join(",".join([str(cell) for cell in column]) for column in matrix)


def show_image_array(image_array, name=None):
    test_image = Image.fromarray(image_array)
    if name is None:
        test_image.show()
    else:
        test_image.save(name)


def test_region(image_size, points, name=None):
    test_image_array = np.zeros([*image_size, 4], np.uint8)
    test_image_array[*points, :] = 255
    show_image_array(test_image_array, name)


def test_boolean_matrix(bool_matrix):
    channels_added = np.repeat(bool_matrix[:, :, np.newaxis], 4, axis=2)
    test_image_array = channels_added.astype(np.uint8) * 255
    show_image_array(test_image_array)


def show_matrix_at_values(test_matrix, value):
    test_boolean_matrix(np.equal(test_matrix, value))


def show_matrix_levels(test_matrix):
    matrix_range = np.max(test_matrix) - np.min(test_matrix)
    zeroed_matrix = test_matrix - np.min(test_matrix)
    subpixel_values = (zeroed_matrix * 255 / matrix_range).astype(np.uint8)
    pixel_greyscale_values = np.repeat(subpixel_values[:, :, np.newaxis], 3, axis=2)
    show_image_array(pixel_greyscale_values)


def test_around(array, point, window_size):
    window = np.clip([
        [point[0] - window_size, point[0] + window_size], [point[1] - window_size, point[1] + window_size],
        [0,0],
        np.array(array.shape)
    ])
    return array[window[0, 0]:window[0, 1], window[1, 0]:window[1, 1]]


def test_point_set(points):
    adjusted_points = points - np.min(points, axis=1, keepdims=True)
    test_region(np.max(adjusted_points, axis=1) + 1, adjusted_points)
