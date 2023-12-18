"""
Extract the wells from a multi-chip experiment.

The only parameter in this script is the path to the folder containing a single 
experiment's images. The other parameters for extraction are not contained in this 
script. They must be placed in a file called `experiment_config.yml` within the 
folder that contains the images to be processed. See example experiment folder and 
notes below.

The other requirement is that the chip image must contain a fiducial. In our test
images, the orientation of the fiducial is in the top right, but if your images 
are oriented differently, you MUST change the window values in the function 
`well_extraction.get_fiducial_center` to consider a window that contains your 
chip's fiducial mark.

NOTES
-----
A multi-chip experiment is assumed to mean an experiment where different conditions 
are tested on various microwell chips. There must be a negative control chip that 
can serve as the basis for the well numbering. In contrast to extracting the wells
from a single chip, extracting the wells from multiple chips requires that the 
wells can be cross-referenced between chips, which increases the complexity of
the chip processing. Given the possible image capture variations, it is nontrivial to
ensure that wells are oriented in the exact same way between images.

The negative control chip is used as a reference, so that the numbers assigned to 
each well in the reference are assumed to be the same in the experimental chips.
This allows the wells to be referenced by ID in any of the chips, so that, for 
instance, if the different experimental chips refer to different time points, each 
well can be tracked over those different time points.
"""

from pathlib import Path

from numpy import array
from pyarrow.feather import write_feather
from sklearn.metrics import pairwise_distances_argmin
from utils.configuration import load_experiment_config
from utils.well_extraction import (
    crop_image,
    extract_wells_whole,
    get_binary_grid,
    get_fiducial_center,
    get_fiducial_wells,
    get_image_dict,
    get_well_locs,
    load_image,
)

# set path to folder containing images from an experiment (including the control)
EXPERIMENT_PATH = Path(
    "/path/to/example_experiment_images"
)

config = load_experiment_config(EXPERIMENT_PATH)

ref_chip_path = EXPERIMENT_PATH / config["input_chip_files"]["negative"]
exp_chip_paths = [
    EXPERIMENT_PATH / p for p in config["input_chip_files"]["experimental"]
]

# initialize sub-folders and other variables
outdir = ref_chip_path.parent / ref_chip_path.stem
wells_outdir = outdir / "well_images"
images_df_path = outdir / "all_wells_raw.feather"
object_df_path = outdir / "all_wells_metadata.feather"

if not outdir.exists():
    outdir.mkdir()

if "chip_cropping" in config:
    crop_coords = (
        config["chip_cropping"]["top_left_x"],
        config["chip_cropping"]["top_left_y"],
        config["chip_cropping"]["bottom_right_x"],
        config["chip_cropping"]["bottom_right_y"],
    )
else:
    crop_coords = None

print("Loading reference image file...")
ref_chip_raw = load_image(ref_chip_path)

print("Prepping reference image...")
ref_chip_prepped = get_image_dict(ref_chip_raw)

print("Cropping reference image...")
ref_chip_cropped = crop_image(ref_chip_prepped, coords=crop_coords)

print("Binarizing reference image...")
ref_chip_cropped["grid_binary"] = get_binary_grid(ref_chip_cropped["grid"])

print("Getting reference well locations...")
ref_chip_cropped["peaks"] = get_well_locs(ref_chip_cropped)

print("Extracting reference wells...")
ref_images_df, ref_objects_df = extract_wells_whole(ref_chip_cropped, wells_outdir)

print("Getting location of reference chip fiducial...")
ref_fid_x, ref_fid_y = get_fiducial_center(ref_chip_cropped["grid_binary"])

(
    ref_well_1_x,
    ref_well_1_y,
    _,
    _,
) = get_fiducial_wells(ref_objects_df, ref_fid_x, ref_fid_y)
ref_point = array([ref_well_1_y, ref_well_1_x])

print("Writing reference chip data...")
write_feather(ref_images_df, images_df_path)
write_feather(ref_objects_df, object_df_path)

print("Extracting non-reference chip images...")
for infile in exp_chip_paths:
    # initialize sub-folders and other variables
    outdir = infile.parent / infile.stem
    wells_outdir = outdir / "well_images"
    images_df_path = outdir / "all_wells_raw.feather"
    object_df_path = outdir / "all_wells_metadata.feather"

    if not outdir.exists():
        outdir.mkdir()

    print("Loading image file...")
    print(infile)
    chip_raw = load_image(infile)

    print("Prepping image...")
    chip_prepped = get_image_dict(chip_raw)

    print("Cropping image...")
    chip_cropped = crop_image(chip_prepped, coords=crop_coords)

    print("Binarizing image...")
    chip_cropped["grid_binary"] = get_binary_grid(chip_cropped["grid"])

    print("Getting well locations...")
    chip_cropped["peaks"] = get_well_locs(chip_cropped)

    print("Extracting wells...")
    images_df, objects_df = extract_wells_whole(chip_cropped, wells_outdir)

    print("Getting location of chip fiducial...")
    fid_x, fid_y = get_fiducial_center(chip_cropped["grid_binary"])

    (
        well_1_x,
        well_1_y,
        _,
        _,
    ) = get_fiducial_wells(objects_df, fid_x, fid_y)

    print("Aligning wells to reference...")
    point = array([well_1_y, well_1_x])

    correction_vector = ref_point - point

    paired_idxs = pairwise_distances_argmin(
        ref_objects_df[["centroid-0", "centroid-1"]],
        objects_df[["centroid-0", "centroid-1"]] + correction_vector,
    )

    images_df_mapped = images_df.iloc[paired_idxs]
    objects_df_mapped = objects_df.iloc[paired_idxs].reset_index(drop=True)

    print("Writing data...")
    write_feather(images_df_mapped, images_df_path)
    write_feather(objects_df_mapped, object_df_path)
    print()

print("Done!")
