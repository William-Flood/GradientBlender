import json
import os
import sys


class ConfigurationOptions:
    # The canvas height for the visual inspection tool used for debugging
    inspection_window_height = 500
    # The canvas width for the visual inspection tool used for debugging
    inspection_window_width = 500
    # The target size for partitioning a border into chunks for batch processing
    border_chunk_target_size = 500
    cut_direction_change_dropoff_rage = 5

    def load(self, file_name):
        if len(file_name) > 0:
            if os.path.isfile(file_name):
                with open(file_name, "r") as configFile:
                    configDict = json.load(configFile)
                    if "inspection_window_height" in configDict:
                        self.inspection_window_height = configDict["inspection_window_height"]
                    if "inspection_window_width" in configDict:
                        self.inspection_window_width = configDict["inspection_window_width"]
                    if "border_chunk_target_size" in configDict:
                        self.border_chunk_target_size = configDict["border_chunk_target_size"]
                    if "cut_direction_change_dropoff_rage" in configDict:
                        self.cut_direction_change_dropoff_rage = configDict["cut_direction_change_dropoff_rage"]
            else:
                sys.stderr.write(f"Configuration file {file_name} not found")


config_obj = ConfigurationOptions()

