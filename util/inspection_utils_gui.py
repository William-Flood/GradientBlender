import tkinter
from tkinter import StringVar
from PIL import Image, ImageTk
from PIL.Image import Resampling
from scipy.sparse import coo_array
from util.config import config_obj
import numpy as np
from multiprocessing import Process


def visualize_point_list(points, point_data=None):
    """
    Provides a visualization aid for a list of point coordinates
    Args:
        points: A list of integer point coordinates
        point_data: An optional list of data corresponding to each point

    Returns:

    """
    if point_data is None:
        point_data = np.arange(points.shape[1])
    assert len(point_data) == points.shape[1], "Point and data lengths mismatch"
    assert np.all(np.equal(np.unique(points, return_counts=True, axis=1)[1], 1)), "Duplicate points"

    def launch_window():
        inspector = InspectionWindow()
        inspector.load_points(points, point_data)

    Process(target=launch_window()).start()


def launch_array_window(array_nested, array_data_nested):
    inspector = InspectionWindow()
    inspector.load_array(array_nested, array_data_nested)


def visualize_array(array, array_data=None):
    if array_data is None:
        array_data = array

    Process(target=launch_array_window, args=(array, array_data)).start()


class InspectionWindow:
    """
    Intended as a debugging tool to visualize spatial data
    """

    def __init__(self):
        self.root = tkinter.Tk()
        inspection_labels = tkinter.PanedWindow(self.root, orient="horizontal")
        self.y_label_var = StringVar(self.root, "y")
        self.x_label_var = StringVar(self.root, "x")
        self.data_label_var = StringVar(self.root, "data")
        inspection_labels.add(tkinter.Label(inspection_labels, textvariable=self.y_label_var, width=25))
        inspection_labels.add(tkinter.Label(inspection_labels, textvariable=self.x_label_var, width=25))
        inspection_labels.add(tkinter.Label(inspection_labels, textvariable=self.data_label_var))
        inspection_labels.pack(side="top", fill="x")
        self.width = config_obj.inspection_window_width
        self.height = config_obj.inspection_window_height
        self.canvas = tkinter.Canvas(self.root,
                                     width=self.width,
                                     height=self.height)
        self.canvas.bind("<Button-1>", self.start_zoom)
        self.canvas.bind("<ButtonRelease-1>", self.zoom_end)
        self.canvas.bind("<Button-3>", self.zoom_out)
        self.canvas.pack(side="top", fill="both", expand=True)
        self.image = None
        self.zoom_window = np.array([
            [0, config_obj.inspection_window_height],
            [0, config_obj.inspection_window_width]
        ])
        self.zoom_click = np.zeros([2], dtype=np.int32)
        self.pixel_value_array = np.zeros([0, 0, 3], dtype=np.int32)
        self.in_zoom = False

    def load_points(self, point_list, point_values):
        point_list_min = np.min(point_list, axis=1)
        point_list_max = np.max(point_list, axis=1)
        adjusted_window_start = point_list_min
        adjusted_points = point_list - point_list_min[:, np.newaxis]
        adjusted_array_window_size = point_list_max - point_list_min
        point_index_array = coo_array(
            (np.arange(point_list.shape[1]) + 1, adjusted_points),
            shape=adjusted_array_window_size + 1
        )
        point_bool_array = coo_array(
            ([True] * point_list.shape[1], adjusted_points),
            shape=adjusted_array_window_size + 1
        ).toarray()
        self.pixel_value_array = np.repeat(point_bool_array[:, :, np.newaxis], 3, axis=2).astype(np.uint8) * 255
        self.zoom_window = np.array([
            [0, adjusted_array_window_size[0]],
            [0, adjusted_array_window_size[1]]
        ])
        self.redraw()
        self.canvas.bind("<Motion>", lambda event: self.roll_over_point(
            event,
            point_index_array,
            point_values,
            adjusted_window_start))
        self.root.mainloop()

    def event_to_zoom_window(self, event):
        canvas_x = self.canvas.canvasx(event.x) / self.width
        canvas_y = self.canvas.canvasy(event.y) / self.height
        return np.multiply(
            np.array([canvas_y, canvas_x]),
            self.zoom_window[:, 1] - self.zoom_window[:, 0]
        ).astype(np.int32)

    def roll_over_point(self, event, point_index_array, point_values, adjusted_window_start):
        zoom_window_coords = self.event_to_zoom_window(event)
        array_x = int(self.zoom_window[1, 0] + zoom_window_coords[1])
        array_y = int(self.zoom_window[0, 0] + zoom_window_coords[0])
        array_x_guarded = min(array_x, point_index_array.shape[1] - 1)
        array_y_guarded = min(array_y, point_index_array.shape[0] - 1)
        point_index = point_index_array[array_y_guarded, array_x_guarded]
        if point_index > 0:
            y_label_value = str(array_y + adjusted_window_start[0])
            x_label_value = str(array_x + adjusted_window_start[1])
            self.y_label_var.set(y_label_value)
            self.x_label_var.set(x_label_value)
            self.data_label_var.set(str(point_values[point_index - 1]))

    def load_array(self, array, array_data):
        matrix_range = np.max(array) - np.min(array)
        zeroed_matrix = array - np.min(array)
        subpixel_values = (zeroed_matrix * 255 / matrix_range).astype(np.uint8)
        self.pixel_value_array = np.repeat(subpixel_values[:, :, np.newaxis], 3, axis=2)
        self.zoom_window = np.array([
            [0, array.shape[0]],
            [0, array.shape[1]]
        ])
        self.redraw()
        def check_at_coords(y, x):
            return array_data[y, x]
        self.canvas.bind("<Motion>", lambda event: self.roll_over_array(
            event,
            check_at_coords,
            array.shape))
        self.root.mainloop()

    def roll_over_array(self, event, point_value_fn, array_shape):
        zoom_window_coords = self.event_to_zoom_window(event)
        array_x = int(self.zoom_window[1, 0] + zoom_window_coords[1])
        array_y = int(self.zoom_window[0, 0] + zoom_window_coords[0])
        array_x_guarded = min(array_x, array_shape[1] - 1)
        array_y_guarded = min(array_y, array_shape[0] - 1)
        y_label_value = str(array_y)
        x_label_value = str(array_x)
        self.y_label_var.set(y_label_value)
        self.x_label_var.set(x_label_value)
        self.data_label_var.set(str(point_value_fn(array_y_guarded, array_x_guarded)))

    def start_zoom(self, event):
        if not self.in_zoom:
            self.zoom_click = self.event_to_zoom_window(event)

    def zoom_end(self, event):
        if not self.in_zoom:
            release_coords = self.event_to_zoom_window(event)
            release_y = float(release_coords[0])
            release_x = float(release_coords[1])
            zoom_click_y = float(self.zoom_click[0])
            zoom_click_x = float(self.zoom_click[1])
            new_zoom_window = np.array(
                [
                    [min(zoom_click_y, release_y), max(zoom_click_y, release_y)],
                    [min(zoom_click_x, release_x), max(zoom_click_x, release_x)]
                ]
            ).astype(np.int32)
            if np.all(np.greater(new_zoom_window[:, 1] - new_zoom_window[:, 0], 0)):
                self.zoom_window = new_zoom_window
                self.redraw()
                self.in_zoom = True

    def zoom_out(self, event):
        self.zoom_window = np.array([
            [0, self.pixel_value_array.shape[0]],
            [0, self.pixel_value_array.shape[1]]
        ])
        self.redraw()
        self.in_zoom = False

    def redraw(self):
        zoomed_array = self.pixel_value_array[
            self.zoom_window[0, 0]:self.zoom_window[0, 1],
            self.zoom_window[1, 0]:self.zoom_window[1, 1]
        ]
        base_image = Image.fromarray(zoomed_array)
        scaled_image = base_image.resize([self.width, self.height], Resampling.NEAREST)
        self.image = ImageTk.PhotoImage(scaled_image)
        self.canvas.create_image(0, 0, image=self.image, anchor=tkinter.NW)
