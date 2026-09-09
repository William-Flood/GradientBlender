import numpy as np
from PIL import Image
from PIL.ImageDraw import ImageDraw
import sys
if "borders.border" not in sys.modules:
    from borders.border import Border
from typing import List
from regions.region import Region
from util.rasterize import rasterize_polygon


def draw_borders(borders: List[Border], image_shape):
    border_array = np.ones(image_shape, dtype=bool)
    for border in borders:
        # draw_border(border, border_drawer)
        border_array[*border.rasterize()] = False
    border_image = Image.fromarray(border_array.astype(np.uint8) * 255)
    border_image.show()


def draw_region_borders(regions: list[Region], image_shape):
    border_array = np.ones(image_shape, dtype=bool)
    # border_drawer.line([0, 0, image_shape[1] - 1, image_shape[0] - 1], fill=0, width=1)
    for region in regions:
        for border in region.total_borders:
            border_array[*border.rasterize()] = False
    border_image = Image.fromarray(border_array.astype(np.uint8) * 255)
    border_image.show()


