import numpy as np
from PIL import Image
from regions.region import Region
from util.rasterize import rasterize_polygon


def draw_polygons(regions: list[Region], image_shape):
    border_array = np.ones(image_shape, dtype=bool)
    for region in regions:
        for polygon in region.polygons:
            polygon_points = rasterize_polygon(polygon)
            border_array[*polygon_points] = False
    border_image = Image.fromarray(border_array.astype(np.uint8) * 255)
    border_image.show()