"""
Extract the wells from a single chip image.

This only processes the image data within the chip. The limits of the cropping can be 
set below in CROP_COORDS.

The output is a folder with the same name as the input chip image. The output folder
contains two dataframes saved as feather files. 
"""
from pathlib import Path

from pyarrow.feather import write_feather
from utils.well_extraction import (
    crop_image,
    extract_wells_whole,
    get_binary_grid,
    get_image_dict,
    get_well_locs,
    load_image,
)

# set path to input chip image
CHIP_PATH = Path(
    "/path/to/test_chip_1.tif"
)

# Set this to False to attempt to find the chip corners automatically, beware though,
# this heavily relies on image quality and consistency and may not always work.
# otherwise, ensure CROP_COORDS specifies the top-left and bottom-right corners of
# the chip.
FIND_CORNERS = False
CROP_COORDS = (
    (
        828,  # top-left X
        828,  # top-left Y
        16008,  # bottom-right X
        16008,  # bottom-right Y
    )
    if not FIND_CORNERS
    else None
)

# initialize sub-folders and other variables
outdir = CHIP_PATH.parent / CHIP_PATH.stem
wells_outdir = outdir / "well_images"
images_df_path = outdir / "all_wells_raw.feather"
object_df_path = outdir / "all_wells_metadata.feather"

if not outdir.exists():
    outdir.mkdir()

print("Loading image file...")
chip_raw = load_image(CHIP_PATH)

print("Prepping image...")
chip_prepped = get_image_dict(chip_raw)

print("Cropping image...")
chip_cropped = crop_image(chip_prepped, coords=CROP_COORDS)

print("Binarizing image...")
chip_cropped["grid_binary"] = get_binary_grid(chip_cropped["grid"])

print("Getting well locations...")
chip_cropped["peaks"] = get_well_locs(chip_cropped)

print("Extracting wells...")
images_df, objects_df = extract_wells_whole(chip_cropped, wells_outdir)

print("Writing data...")
write_feather(images_df, images_df_path)
write_feather(objects_df, object_df_path)

print("Done!")
