import numpy as np
from PIL import Image
from PIL.ImageDraw import ImageDraw
from borders.border import Border
from typing import List


def draw_borders(borders: List[Border], image_shape):
    border_image = Image.new("1", [image_shape[1], image_shape[0]], 1)
    border_drawer = ImageDraw(border_image)
    # border_drawer.line([0, 0, image_shape[1] - 1, image_shape[0] - 1], fill=0, width=1)
    for border in borders:
        draw_border(border, border_drawer)
    border_image.show()



def draw_border(border: Border, border_drawer):
    border_points_point_index_first = border.decomposed_points.T
    border_points_xy = np.take(border_points_point_index_first, np.array([1, 0]), axis=1).astype(np.int32)
    lines = np.array(list(zip(border_points_xy[:-1], border_points_xy[1:])))
    for line in lines:
        line_flattened = list(line.flatten())
        border_drawer.line(line_flattened, fill=0, width=1)
    if border.loop:
        border_drawer.line(
            (border_points_xy[-1, 0], border_points_xy[-1, 1], border_points_xy[0, 0], border_points_xy[0, 1]),
            fill=0, width=1
        )

