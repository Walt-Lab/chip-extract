"""
Code for extracting the individual aggregate images. This is the meaty stuff.
"""

from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
from scipy.ndimage import binary_closing, distance_transform_edt
from scipy.signal import find_peaks
from tqdm import tqdm


def load_image(inpath: Path) -> np.ndarray:
    """
    Load a .tif image as a numpy.ndarray

    Parameters
    ----------
    inpath : Path
        Path to an image in `.tif` format

    Returns
    -------
    img : numpy.ndarray
        Raw image data
    """
    img = skimage.io.imread(inpath)

    # check array shape
    if img.ndim != 3:
        raise ValueError(
            "Incorrect image structure. Check the number of channels of the tif file."
        )

    # if first axis of input image is not the channel, correct it
    min_shape_idx = np.argmin(img.shape)
    if min_shape_idx != 0:
        img = np.moveaxis(img, min_shape_idx, 0)

    # swap agg and grid channels if agg channel is not first
    # we identify the grid channel by using the largest mean value of the channel
    # in testing there was a clear difference with the means, the grid channel
    # typically had a mean ~13k, with the other channels much lower (<3k)
    channel_means = [img[i, ...].mean() for i in range(img.shape[0])]
    if np.argmax(channel_means) == 0:
        img[[0, 1], ...] = img[[1, 0], ...]

    # need to check resolution of image
    # TODO:
    # this is a nasty hack!! The code needs to be fixed to process whatever
    # resolution we give it
    if img.shape[1] < 16300:
        new_channels = [
            skimage.transform.resize(img[i, ...], (16382, 16382), preserve_range=True)
            for i in range(img.shape[0])
        ]
        img = np.array(new_channels, dtype="uint16")

    return img


def get_image_dict(in_img: np.ndarray) -> dict:
    """Take a raw input sub image and return a dict with the individual image channels

    Parameters
    ----------
    in_img : np.ndarray
        Raw input image loaded with `load_image`

    Returns
    -------
    out : dict
        Dictionary of individual channel images
    """

    imgs = {}

    # separate individual image channels
    imgs["aggs"] = in_img[0, ...]
    imgs["grid"] = skimage.exposure.rescale_intensity(
        in_img[1, ...],
        in_range=(
            np.percentile(in_img[1, ...], 0.01),
            np.percentile(in_img[1, ...], 99.99),
        ),
        out_range=(0, 255),
    ).astype(np.uint8)

    if in_img.shape[0] == 3:
        imgs["beads1"] = in_img[2, ...]
    elif in_img.shape[0] == 4:
        imgs["beads1"] = in_img[2, ...]
        imgs["beads2"] = in_img[3, ...]

    return imgs


def get_chip_corners(
    grid: np.ndarray, window_x: int = 1500, window_y: int = 1500
) -> tuple:
    """Find top-left and bottom-right corners of raw chip image

    Parameters
    ----------
    grid : np.ndarray
        Brightfield channel of chip grid
    window_x(y) : int
        Number of pixels to "zoom-in" on the corner

    Returns
    -------
    out : tuple
        top_left : tuple
            (x,y)-coordinates of top-left corner
        bot_right : tuple
            (x,y)-coordinates of bottom-right corner
    """
    top_left = get_binary_grid(grid[:window_x, :window_y])
    bot_right = get_binary_grid(grid[-window_x:, -window_y:])

    # these were found experimentally; THEY MIGHT NOT ALWAYS WORK
    CORNER_K = 0.15
    CORNER_MIN_DIST = 50
    CORNER_THRESH = 0.02

    # get top-left corner coordinates
    coords_tl = skimage.feature.corner_peaks(
        skimage.feature.corner_harris(top_left, k=CORNER_K),
        min_distance=CORNER_MIN_DIST,
        threshold_rel=CORNER_THRESH,
    )

    # get bottom-right corner coordinates, these are in relation
    # to a sub-window, so we have to translate them to the
    # coordinates of the entire chip image
    coords_br_sub = skimage.feature.corner_peaks(
        skimage.feature.corner_harris(bot_right, k=CORNER_K),
        min_distance=CORNER_MIN_DIST,
        threshold_rel=CORNER_THRESH,
    )

    # check if more than one corner point was found. we don't want that!
    if coords_tl.shape[0] != 1:
        raise ValueError("More than one corner point detected in top-left.")
    if coords_br_sub.shape[0] != 1:
        raise ValueError("More than one corner point detected in bottom-right.")

    # get bottom-right corner coordinates in terms of the entire chip image
    coords_br = np.array(
        [
            grid.shape[0] - window_x + coords_br_sub[0, 0],
            grid.shape[1] - window_x + coords_br_sub[0, 1],
        ]
    )

    return coords_tl.reshape(2), coords_br


def crop_image(
    in_img: dict,
    coords: Union[Tuple[int, int, int, int], None] = None,
    buffer: int = 30,
    window_x: int = 1000,
    window_y: int = 1000,
):
    """Remove outside edges of chip image

    Parameters
    ----------
    in_img : dict
        Dictionary of image channels (see `get_image_dict`)
    coords : Union[Tuple[int, int, int, int], None]
        Either a tuple of xy-coordinates where:
            tl_x, tl_y, br_x, br_y = coords
        Or None
    buffer : int
        Add this number of pixels to corner locations to move away from chip edges
    window_x : int
        Size of window for finding the top-left and bottom-right corners
    window_y : int
        See `window_x`

    Returns
    -------
    out : dict
        Cropped version of input image
    """
    if coords is None:
        coords_tl, coords_br = get_chip_corners(
            in_img["grid"], window_x=window_x, window_y=window_y
        )

        # add a buffer to remove small regions of edge
        coords_tl += buffer
        coords_br -= buffer

        # the crop function requires a specific format for crop_width, see:
        # https://scikit-image.org/docs/stable/api/skimage.util.html#crop
        crop_arr = np.array(
            [
                [coords_tl[0], in_img["grid"].shape[0] - coords_br[0]],
                [coords_tl[1], in_img["grid"].shape[1] - coords_br[1]],
            ]
        )
    else:
        tl_x, tl_y, br_x, br_y = coords
        crop_arr = np.array(
            [
                [tl_x, in_img["grid"].shape[0] - br_x],
                [tl_y, in_img["grid"].shape[1] - br_y],
            ]
        )

    return {chan: skimage.util.crop(in_img[chan], crop_arr) for chan in in_img}


def get_binary_grid(grid_img: np.ndarray) -> np.ndarray:
    """
    Take in the grid image and produce a binary mask of the wells.

    Parameters
    ----------
    grid_img : numpy.ndarray
        Brightfield channel of wells

    Returns
    -------
    out : numpy.ndarray
        Binary array with same shape as grid_img

    Notes
    -----
    if you look at the histogram of the input image, you'll notice there
    are two separated groups. So we find a point in between these two
    groups as a threshold to say: values above this point are wells, values
    below this point are background
    """
    grid_img = skimage.filters.rank.enhance_contrast(
        grid_img, skimage.morphology.disk(25)
    )

    hist, _ = skimage.exposure.histogram(grid_img)

    peaks, _ = find_peaks(hist, prominence=300000)

    if peaks.shape[0] < 2:
        cutoff = grid_img.max() / 2
    else:
        cutoff = int((peaks[1] + peaks[0]) * 0.55)

    # actually assign binary values to the well or non-well areas
    img_binary = np.ones_like(grid_img)
    img_binary[grid_img > cutoff] = 0

    # remove small white artifacts
    img_binary = skimage.morphology.binary_opening(img_binary, np.ones((4, 4)))

    return img_binary


def get_well_locs(img: dict):
    """Get well locations

    Parameters
    ----------
    img : dict
        Input image after being prepped

    Returns
    -------
    out : numpy.ndarray
        Well center locations as image with intensities
    """
    # distance transform the binary grid
    xform = distance_transform_edt(
        skimage.morphology.remove_small_objects(img["grid_binary"], 20)
    )
    # identify peaks in edt map
    return skimage.feature.peak_local_max(xform, min_distance=25, threshold_rel=0.4)


def extract_wells_whole(img: dict, well_outdir: Path, save_raw: bool = False):
    """Pull out all wells from a (cropped) chip image

    Parameters
    ----------
    img : dict
        Input image after being prepped
    well_outdir : pathlib.Path
        Path to save feather file of well images
    obj_plot_path : pathlib.Path
        Path to save plot of identified objects (very large image)
    save_raw : bool
        Whether or not to save a separate `.png` image per well

    Returns
    -------
    out: tuple
        out[0]: dataframe of reshaped well images
        out[1]: dataframe with object metadata
    """
    if save_raw:
        raw_outdir = well_outdir / "raw"
        raw_outdir.mkdir(parents=True, exist_ok=True)

    # sort the index based on X then Y coordinates
    object_df = (
        pd.DataFrame(img["peaks"], columns=["centroid-0", "centroid-1"])
        .sort_values(by=["centroid-1", "centroid-0"])
        .reset_index(drop=True)
    )

    # Binary closing if less than 20564 wells identified
    if object_df.shape[0] < 20654:
        print(f"{object_df.shape[0]} wells detected, try binary_closing to fix it.")
        closed_binary = binary_closing(
            img["grid_binary"], structure=np.ones((10, 10))
        ).astype(bool)
        xform = distance_transform_edt(
            skimage.morphology.remove_small_objects(closed_binary, 20)
        )
        object_df = skimage.feature.peak_local_max(
            xform, min_distance=25, threshold_rel=0.7
        )
        object_df = (
            pd.DataFrame(object_df, columns=["centroid-0", "centroid-1"])
            .sort_values(by=["centroid-1", "centroid-0"])
            .reset_index(drop=True)
        )
        print(f"Done. {object_df.shape[0]} wells detected")

    if not object_df.shape[0] == 20654:
        save_peaks(xform, object_df, well_outdir)
        save_peaks(
            img["grid_binary"],
            object_df,
            well_outdir / f"{well_outdir.parent.name}_binary.png",
        )
        print(
            f"Incorrect number of wells identified in file: "
            f"'{well_outdir.parent.name, object_df.shape[0]}'"
        )
    # set minimum of X-coordincates to 45 if found less than 45.
    object_df.loc[object_df[object_df["centroid-1"] < 46].index, "centroid-1"] = 45

    well_imgs_list = []

    for idx, row in tqdm(object_df.iterrows(), total=object_df.shape[0]):
        # process a single well
        process_check = extract_single_well(
            img, row, well_outdir, idx, save_raw=save_raw
        )

        # if there's an error with well processing, don't add it to dataset
        if type(process_check) != int:
            well_imgs_list.append(process_check)

    well_imgs_df = pd.DataFrame(data=np.concatenate(well_imgs_list, axis=0))

    return well_imgs_df, object_df


def extract_single_well(
    sub_img: dict,
    object_series: pd.Series,
    outdir: Path,
    image_label: str,
    img_size: int = 90,
    test: bool = False,
    mask_well: bool = False,
    save_raw: bool = False,
):
    """Extract individual well from flourescence channel and save it to disk

    Returns
    -------
    out : varies
        -1 if well image is too close to edge (error case)
        Raw well image (reshaped) if no error
    """
    min_row = object_series["centroid-0"] - (img_size // 2)
    min_col = object_series["centroid-1"] - (img_size // 2)
    max_row = min_row + img_size
    max_col = min_col + img_size

    well_img = sub_img["aggs"][min_row:max_row, min_col:max_col]

    if mask_well:
        well_img_mask = sub_img["grid_binary"][min_row:max_row, min_col:max_col]
        well_img = well_img * well_img_mask

    # skip processing if well has incorrect shape
    if well_img.shape != (img_size, img_size):
        return -1

    if save_raw:
        if not test:
            skimage.io.imsave(
                outdir / "raw" / f"{image_label}.png",
                well_img,
                check_contrast=False,
            )
        else:
            skimage.io.imshow(well_img, cmap="magma")
            plt.show()
            plt.close()

    return well_img.reshape(1, img_size**2)


def save_peaks(img: np.ndarray, peaks: pd.DataFrame, out_dir: Path):
    plt.imshow(img)
    plt.autoscale(False)
    plt.plot(peaks["centroid-1"], peaks["centroid-0"], "r.", ms=0.2)
    for x, y, idx in zip(peaks["centroid-1"], peaks["centroid-0"], peaks.index):
        plt.text(
            x,
            y,
            idx,
            fontsize=0.1,
            horizontalalignment="center",
            verticalalignment="center",
        )
    plt.axis("off")
    plt.savefig(out_dir, dpi=1000, facecolor="white")
    plt.close()


def get_fiducial_center(grid_img: np.ndarray) -> Tuple[int, int]:
    """Find center point of chip fiducial

    Parameters
    ----------
    grid_img : numpy.ndarray
        Cropped image of microwell chip

    Returns
    -------
    out : tuple[int, int]
        Tuple of (x,y) coordinates
    """
    # these parameters should choose a relaxed window that
    # always contains the fiducial
    win_x_min = 9800
    win_x_max = 11500
    win_y_min = 3700
    win_y_max = 5500

    img = grid_img[win_y_min:win_y_max, win_x_min:win_x_max]

    if img.dtype != bool:
        img = get_binary_grid(img)

    xform = distance_transform_edt(~img)
    peaks = skimage.feature.peak_local_max(xform, threshold_rel=0.5)

    peak_x = int(np.mean(peaks[:, 1]).round(0)) + win_x_min
    peak_y = int(np.mean(peaks[:, 0]).round(0)) + win_y_min

    return peak_x, peak_y


def get_fiducial_wells(
    objects_df: pd.DataFrame, fid_x: int, fid_y: int
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Get two wells near fiducial for orientation

    Parameters
    ----------
    objects_df : pandas.DataFrame
        Dataframe of well objects found with `extract_wells_whole`
    fid_x : int
        X-coordinate (column) of fiducial center
    fid_y : int
        Y-coordinate (row) of fiducial center

    Returns
    -------
    out : Tuple[Tuple[int, int], Tuple[int, int]]
        A tuple of ((x,y), (x,y)) with locations of two wells,
        one above the fiducial center, the other below
    """
    well_1 = (
        objects_df.loc[
            (objects_df["centroid-1"] < (fid_x + 20))
            & (objects_df["centroid-1"] > (fid_x - 20))
            & (objects_df["centroid-0"] > fid_y)
        ]
        .sort_values("centroid-0")
        .iloc[0]
    )

    well_1_x = well_1["centroid-1"]
    well_1_y = well_1["centroid-0"]

    well_2 = (
        objects_df.loc[
            (objects_df["centroid-1"] < (fid_x + 20))
            & (objects_df["centroid-1"] > (fid_x - 20))
            & (objects_df["centroid-0"] < fid_y)
        ]
        .sort_values("centroid-0", ascending=False)
        .iloc[0]
    )

    well_2_x = well_2["centroid-1"]
    well_2_y = well_2["centroid-0"]

    return well_1_x, well_1_y, well_2_x, well_2_y
