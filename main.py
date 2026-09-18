from fill_gradient import subdivided_border_tangent_interpolation_fill, closest_and_opposite_interpolation_fill
import argparse
import os
import time
import math
import cProfile
import pstats
import io
import gradengc
import numpy as np
from util.inspection_utils import *


def build_default_output_filename(input_filename):
    """
    Adds 'Gradient' before the file extension.
    Example:
        image.png -> imageGradient.png
    """
    base, ext = os.path.splitext(input_filename)
    return f"{base}Gradient{ext}"


def closest_and_opposite_main():
    parser = argparse.ArgumentParser(
        description="Apply gradients to an image using a guide image."
    )

    # Positional arguments
    parser.add_argument(
        "guide_image_file",
        help="Input guide image file"
    )

    parser.add_argument(
        "result_image_file",
        nargs="?",
        help=(
            "Output image file. "
            "If omitted, 'Gradient' is added to the input filename."
        )
    )

    # Optional named arguments
    parser.add_argument(
        "-batch_stripe_distance",
        type=int,
        default=5,
        help="Used to divide processing into batches"
    )

    parser.add_argument(
        "-region_proportion_threshold",
        type=float,
        default=0.01,
        help=(
            "Threshold used to recognize aliasing between region boundaries"
        )
    )

    args = parser.parse_args()

    # Generate default output filename if omitted
    result_image_file = (
        args.result_image_file
        if args.result_image_file
        else build_default_output_filename(args.guide_image_file)
    )

    start_time = time.time()
    closest_and_opposite_interpolation_fill(
        guide_image_file=args.guide_image_file,
        result_image_file=result_image_file,
        batch_stripe_distance=args.batch_stripe_distance,
        region_proportion_threshold=args.region_proportion_threshold
    )
    print(f"Elapsed: {time.time() - start_time}")


def subdivided_border_main():
    parser = argparse.ArgumentParser(
        description="Apply gradients to an image using a guide image."
    )

    # Positional arguments
    parser.add_argument(
        "guide_image_file",
        help="Input guide image file"
    )

    parser.add_argument(
        "result_image_file",
        nargs="?",
        help=(
            "Output image file. "
            "If omitted, 'Gradient' is added to the input filename."
        )
    )

    parser.add_argument(
        "-region_defuzz_threshold",
        type=float,
        default=10,
        help=(
            "Threshold used to filter out errant pixels"
        )
    )

    parser.add_argument(
        "-sideways_cut_penalty",
        type=float,
        default=100,
        help=(
            "Multiplier to cuts at a right angle to the last cut during hole removal"
        )
    )

    parser.add_argument(
        "-border_defuzz_threshold",
        type=float,
        default=5,
        help=(
            "Threshold used to filter out errant pixels"
        )
    )

    parser.add_argument(
        "-region_proportion_threshold",
        type=float,
        default=0.01,
        help=(
            "Threshold used to recognize aliasing between region boundaries"
        )
    )

    parser.add_argument(
        "-decimate_deviation_cutoff",
        type=float,
        default=3,
        help=(
            "Used to decide how much to segment the border between region boundaries"
        )
    )

    parser.add_argument(
        "-edge_subdivide_ratio",
        type=float,
        default=3,
        help=(
            "Used during concave polygon splitting to decide when to subdivide an existing edge"
        )
    )

    parser.add_argument(
        "-system_config_file",
        type=str,
        default="config.json",
        help="Provides constants used for performance tuning and expected to be system-specific, such as batch sizes.  "
             + "If the config file does not exist, defaults have been hard-coded inside util.config.py"
    )

    parser.add_argument(
        "-profile",
        action="store_true"
    )

    args = parser.parse_args()

    # Generate default output filename if omitted
    result_image_file = (
        args.result_image_file
        if args.result_image_file
        else build_default_output_filename(args.guide_image_file)
    )

    if args.profile:
        with cProfile.Profile() as pr:
            # pr.enable()
            subdivided_border_tangent_interpolation_fill(
                guide_image_file=args.guide_image_file,
                result_image_file=result_image_file,
                region_defuzz_threshold=args.region_defuzz_threshold,
                sideways_cut_penalty=args.sideways_cut_penalty,
                border_defuzz_threshold=args.border_defuzz_threshold,
                region_proportion_threshold=args.region_proportion_threshold,
                decimate_deviation_cutoff=args.decimate_deviation_cutoff,
                edge_subdivide_ratio=args.edge_subdivide_ratio,
                config_file_name=args.system_config_file
            )
            pr.disable()
            s = io.StringIO()
            sort_by = pstats.SortKey.CUMULATIVE
            ps = pstats.Stats(pr, stream=s).sort_stats(sort_by)
            ps.print_stats()
            print(s.getvalue())
    else:
        start_time = time.time()
        subdivided_border_tangent_interpolation_fill(
            guide_image_file=args.guide_image_file,
            result_image_file=result_image_file,
            region_defuzz_threshold=args.region_defuzz_threshold,
            border_defuzz_threshold=args.border_defuzz_threshold,
            region_proportion_threshold=args.region_proportion_threshold,
            decimate_deviation_cutoff=args.decimate_deviation_cutoff,
            edge_subdivide_ratio=args.edge_subdivide_ratio,
            config_file_name=args.system_config_file
        )
        print(f"Elapsed: {time.time() - start_time}")


def testfoo():
    rng = np.random.default_rng(4451)

    # a = np.array([[7, 2, 6, 6],[4, 2, 5, 4]], dtype=np.int32)
    a = rng.integers(5, 150, [2, 500]).astype(np.int32)
    start_time = time.time()
    res = gradengc.foo([a])
    print(f"Elapsed: {time.time() - start_time}")
    return res


def test_point_orderer():
    circle_indices = np.indices([210, 210])
    flat_indices = np.reshape(circle_indices, [2, -1])
    flattened_circle = np.less_equal(np.sum(np.square(flat_indices - [[105], [105]]), axis=0), 10000).astype(np.int32)
    region_array = np.reshape(flattened_circle, [210, 210]).astype(np.int32)
    left_index_locations = np.argmax(region_array[5:206], axis=1)
    border_indices_sides = np.concat([
        circle_indices[:, np.arange(201) + 5, left_index_locations],
        circle_indices[:, np.arange(201) + 5, 210 - left_index_locations]
    ], axis=1)
    border_indices = np.concat([
        border_indices_sides,
        np.roll(border_indices_sides, 1, axis=0)
    ], axis=1).astype(np.int32)
    border_check = np.zeros([210, 210])
    border_check[*border_indices] = 1
    walk_start = time.time()
    walk = gradengc.order_border_points(np.ascontiguousarray(border_indices), np.ascontiguousarray(region_array))
    print(f"Walk: {time.time() - walk_start}")
    walk_filtered = walk[np.greater_equal(walk[:, 0], 0)]
    walk_check = np.zeros([210, 210])
    walk_check[*walk_filtered.T] = np.arange(walk_filtered.shape[0]) + 100
    show_matrix_levels(walk_check)
    return walk_filtered



if __name__ == "__main__":
    # t = test_point_orderer()
    subdivided_border_main()