from fill_gradient import subdivided_border_tangent_interpolation_fill, closest_and_opposite_interpolation_fill
import argparse
import os
import time


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
        default=20,
        help=(
            "Threshold used to filter out errant pixels"
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
        "-border_decimate_r_2_threshold",
        type=float,
        default=0.99,
        help=(
            "Used to decide how much to segment the border between region boundaries"
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
    subdivided_border_tangent_interpolation_fill(
        guide_image_file=args.guide_image_file,
        result_image_file=result_image_file,
        region_defuzz_threshold=args.region_defuzz_threshold,
        border_defuzz_threshold=args.border_defuzz_threshold,
        region_proportion_threshold=args.region_proportion_threshold,
        border_decimate_r_2_threshold=args.border_decimate_r_2_threshold
    )
    print(f"Elapsed: {time.time() - start_time}")


if __name__ == "__main__":
    subdivided_border_main()