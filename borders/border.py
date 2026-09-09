import numpy as np
from numpy.typing import NDArray
import sys
from util.rasterize import rasterize_line, rasterize_polygon
if "regions.region" not in sys.modules:
    from regions.region import Region
from util.inspection_utils import *


class Border:
    def __init__(self, border_points: NDArray[np.int32], region_one: Region, region_two: Region):
        self.full_points = border_points
        self.region_one = region_one
        self.region_two = region_two
        self.connecting_borders = [[],[]]
    decomposed_points: NDArray[np.int32]
    is_loop: bool

    @property
    def endpoints(self):
        if self.is_loop:
            return self .decomposed_points[:, [0, 0]]
        else:
            return self.decomposed_points[
                [[0, 0], [1, 1]],
                [[0, -1], [0, -1]]
            ]

    @endpoints.setter
    def endpoints(self, new_endpoints):
        if self.is_loop:
            self.decomposed_points[:, 0] = new_endpoints[:, 0]
        else:
            self.decomposed_points[
                [[0, 0], [1, 1]],
                [[0, -1], [0, -1]]
            ] = new_endpoints

    def get_offset(self, offset):
        endpoint_vector = self.decomposed_points[:, 0] - self.decomposed_points[:, -1]
        if endpoint_vector[1] < 0:
            endpoint_vector = endpoint_vector * -1
        border_normal_unsized = np.array([endpoint_vector[1], endpoint_vector[0] * -1])
        border_normal = border_normal_unsized / np.linalg.norm(border_normal_unsized)
        return (self.decomposed_points + border_normal[:, np.newaxis] * offset).astype(np.int32)

    def collide_line_segments(self, segments):
        # Adapted from https://www.jeffreythompson.org/collision-detection/line-line.php
        x1 = self.decomposed_points[1, :-1]
        y1 = self.decomposed_points[0, :-1]
        y2 = self.decomposed_points[0, 1:]
        x2 = self.decomposed_points[1, 1:]
        x3 = segments[1, :-1]
        y3 = segments[0, :-1]
        y4 = segments[0, 1:]
        x4 = segments[1, 1:]

        uA = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
        uB = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))

        uA_in_range = np.logical_and(np.greater_equal(uA, 0), np.less_equal(uA, 1))
        uB_in_range = np.logical_and(np.greater_equal(uB, 0), np.less_equal(uB, 1))
        return np.logical_and(uA_in_range, uB_in_range)

    @property
    def length(self):
        return np.sum(
            np.linalg.norm(
                self.decomposed_points[:, :-1] - self.decomposed_points[:, 1:], axis=0
            )
        )

    def rasterize(self):
        if self.decomposed_points.shape[1] == 1:
            return self.decomposed_points[:, 0]
        if self.decomposed_points.shape[1] == 2:
            return rasterize_line(
                self.decomposed_points[:, 0],
                self.decomposed_points[:, 1]
            )
        elif self.is_loop:
            return rasterize_polygon(self.decomposed_points)
        else:
            return np.concat(
                [
                    rasterize_line(point_1, point_2) for point_1, point_2 in
                    zip(self.decomposed_points[:, :-1].T, self.decomposed_points[:, 1:].T)
                ], axis=1
            )

    def cinch_endpoints(self):
        summed_connected_endpoints = np.array([
            np.sum([
                border.endpoints[:, endpoint_index] for border, endpoint_index in endpoint_connection
            ], axis=0)
            if len(endpoint_connection) > 0 else np.zeros([2])
            for point_index, endpoint_connection in enumerate(self.connecting_borders)
        ]).T
        if self.is_loop:
            summed_connected_endpoint = np.sum(summed_connected_endpoints, axis=1)
            new_endpoint = ((summed_connected_endpoint + self.decomposed_points[:, 0]) / (np.sum(
                [[len(endpoint_connection) for endpoint_connection in self.connecting_borders]]
            ) + 1)).astype(np.int32)
            self.decomposed_points[:, 0] = new_endpoint
            for connection_list in self.connecting_borders:
                for connected_border, endpoint_index in connection_list:
                    connected_border.cinch_from(endpoint_index, new_endpoint)
        else:
            new_endpoints = ((summed_connected_endpoints + self.endpoints) / np.array(
                [[len(endpoint_connection) + 1 for endpoint_connection in self.connecting_borders]]
            )).astype(np.int32)
            self.endpoints = new_endpoints
            for connection_list, endpoint in zip(self.connecting_borders, new_endpoints.T):
                for connected_border, endpoint_index in connection_list:
                    connected_border.cinch_from(endpoint_index, endpoint)

    def cinch_from(self, endpoint_index, new_endpoint):
        if endpoint_index == 0 or self.is_loop:
            self.decomposed_points[:, 0] = new_endpoint
        else:
            self.decomposed_points[:, -1] = new_endpoint

    def make_into_loop(self):
        self.is_loop = True
        self.decomposed_points[:, 0] = (self.decomposed_points[:, 0] + self.decomposed_points[:, -1]) // 2
        if self.decomposed_points.shape[1] > 1:
            self.decomposed_points = self.decomposed_points[:, :-1]

