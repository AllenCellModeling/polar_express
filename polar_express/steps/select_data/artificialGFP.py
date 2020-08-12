from pathlib import Path
import numpy as np

# Third party
from aicsimageio import imread
from skimage.morphology import ball, binary_closing
from scipy import ndimage
import pandas as pd
from tqdm import tqdm
import copy
import tifffile
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import cv2


def makeartificialGFP(
    selected_cell, artificial_cell_dir, vizcells, artificial_plot_dir
):
    """

    Method to create artificial GFP datasets based on 3D images of real cells

    Parameters
    ----------
    selected_cell : Pandas Series object with information about the selected cell
    artificial_cell_dir : Path to folder where the artificial GFP cells (tiff files)
        should be stored
    vizcells : List of cell IDs that will be visualized
    artificial_plot_dir : Path to folder with slice and stack tiffs of the artificial
        GFP cells

    Returns
    -------
    art_cells : Pandas DataFrame object with information about saved artificial cells


    """

    # image file
    file = selected_cell["Path"]
    cellid = selected_cell["CellId"]

    # visualize or not
    if cellid in vizcells:
        viz_flag = True
    else:
        viz_flag = False

    # read in image file
    imorg = imread(file)
    im = np.squeeze(imorg)

    # get the channel indices
    ch_struct = selected_cell["ch_struct"]
    ch_seg_nuc = selected_cell["ch_seg_nuc"]
    ch_seg_cell = selected_cell["ch_seg_cell"]

    # process channels
    seg_dna = np.squeeze(im[ch_seg_nuc, :, :, :])
    seg_dna[seg_dna > 0] = 255
    seg_dna = binary_closing(seg_dna.astype(bool), ball(7)).astype("uint8")
    seg_dna[seg_dna > 0] = 255
    seg_mem = np.squeeze(im[ch_seg_cell, :, :, :])
    seg_mem[seg_mem > 0] = 255
    seg_dna[(seg_mem == 0) & (seg_dna == 255)] = 0
    gfp = np.squeeze(im[ch_struct, :, :, :])
    seg_gfp = (seg_mem > 0) & (seg_dna <= 0)
    gfp = np.multiply(seg_gfp, gfp)

    # Center of nucleus and bottom of cell membrane
    center_of_nucleus = np.round(ndimage.center_of_mass(seg_dna)).astype(np.int)
    bottom_of_cell = np.min(np.argwhere(np.sum(seg_mem, axis=tuple([1, 2])) != 0))

    # %% Apply artificial dataset
    art_cells = pd.DataFrame()
    art_dataset_types = [
        "Uniform",
        "Uniform + noise",
        "Top Half Uniform",
        "Top Half Uniform + noise",
        "Tight around Nucleus",
        "Loose around Membrane",
        "Top Half Nucleus",
        "Bottom Membrane",
    ]
    for ii, adt in enumerate(art_dataset_types):

        if adt == "Uniform":
            gfp_art = copy.copy(gfp)
            gfp_art[seg_gfp > 0] = 255
        elif adt == "Uniform + noise":
            gfp_art = copy.copy(gfp)
            x = gfp_art[seg_gfp > 0]
            x = np.random.normal(128, 128, size=x.shape)
            x[x < 0] = 0
            x[x > 255] = 255
            x = x * (255 / np.max(x))
            gfp_art[seg_gfp > 0] = x
        elif adt == "Top Half Uniform":
            gfp_art = copy.copy(gfp)
            gfp_art[seg_gfp > 0] = 255
            gfp_art[0 : center_of_nucleus[0], :, :] = 0
        elif adt == "Top Half Uniform + noise":
            gfp_art = copy.copy(gfp)
            x = gfp_art[seg_gfp > 0]
            x = np.random.normal(128, 128, size=x.shape)
            x[x < 0] = 0
            x[x > 255] = 255
            x = x * (255 / np.max(x))
            gfp_art[seg_gfp > 0] = x
            gfp_art[0 : center_of_nucleus[0], :, :] = 0
        elif adt == "Tight around Nucleus":
            q = np.multiply(
                seg_gfp,
                ndimage.gaussian_filter(seg_dna, sigma=[3 * 0.108333333 / 0.29, 5, 5]),
            )
            q = q * (255 / np.max(q))
            gfp_art = copy.copy(q)
        elif adt == "Loose around Membrane":
            q = np.multiply(
                seg_gfp,
                ndimage.gaussian_filter(
                    255 - seg_mem, sigma=[6 * 0.108333333 / 0.29, 5, 5]
                ),
            )
            q = q * (255 / np.max(q))
            gfp_art = copy.copy(q)
        elif adt == "Top Half Nucleus":
            q = np.multiply(
                seg_gfp,
                ndimage.gaussian_filter(seg_dna, sigma=[3 * 0.108333333 / 0.29, 5, 5]),
            )
            q = q * (255 / np.max(q))
            gfp_art = copy.copy(q)
            gfp_art[0 : center_of_nucleus[0], :, :] = 0
        elif adt == "Bottom Membrane":
            q = np.multiply(
                seg_gfp,
                ndimage.gaussian_filter(
                    255 - seg_mem, sigma=[6 * 0.108333333 / 0.29, 5, 5]
                ),
            )
            q = q * (255 / np.max(q))
            gfp_art = copy.copy(q)
            gfp_art[bottom_of_cell + 5 :, :, :] = 0

        # Add to new tiff file
        imart = copy.deepcopy(imorg)
        imart[:, :, ch_struct, :, :, :] = gfp_art.astype("uint8")
        imart[:, :, ch_seg_nuc, :, :, :] = seg_dna
        imart[:, :, ch_seg_cell, :, :, :] = seg_mem
        tfile = artificial_cell_dir / f"cell_{cellid}_{adt}.tiff"
        tifffile.imwrite(tfile, imart)
        art_cells = art_cells.append(selected_cell, ignore_index=True)
        art_cells.loc[ii, "Path"] = tfile
        art_cells.loc[ii, "Structure"] = adt

        if viz_flag is True:
            sc = art_cells.iloc[ii]
            makeTIFFsliceandstack(sc, artificial_plot_dir)

    return art_cells


def makeTIFFsliceandstack(selected_cell, artificial_plot_dir):
    """

    Method to create a center slice PNG and a stacked tiff viewer for ImageJ

    Parameters
    ----------
    selected_cell : Pandas Series object with information about the selected cell
        including path to the tiff file
    artificial_plot_dir : Path to folder with slice and stack tiffs of the artificial
        GFP cells

    Returns
    -------

    """

    # read in image file
    file = selected_cell["Path"]
    cellid = selected_cell["CellId"]
    adt = selected_cell["Structure"]
    im = np.squeeze(imread(file))

    # get the channel indices
    ch_struct = int(selected_cell["ch_struct"])
    ch_seg_nuc = int(selected_cell["ch_seg_nuc"])
    ch_seg_cell = int(selected_cell["ch_seg_cell"])

    # process channels
    seg_dna = np.squeeze(im[ch_seg_nuc, :, :, :])
    seg_mem = np.squeeze(im[ch_seg_cell, :, :, :])
    seg_gfp = (seg_mem > 0) & (seg_dna <= 0)
    gfp_art = np.squeeze(im[ch_struct, :, :, :])

    # Color transform function
    DNA_color = [100, 100, 200]
    OUT_color = [150, 150, 150]
    GFP_colors = np.array([[0, 0, 0], [0, 255, 0]])
    seg_dna_un = np.unique(seg_dna)
    seg_mem_un = np.unique(seg_mem)
    gfp_minmax = [np.amin(gfp_art[seg_gfp > 0]), np.amax(gfp_art[seg_gfp > 0])]

    # Center of nucleus and bottom of cell membrane
    center_of_nucleus = np.round(ndimage.center_of_mass(seg_dna)).astype(np.int)

    # additional image information
    pixelScaleX = selected_cell["PixelScaleX"]
    pixelScaleZ = selected_cell["PixelScaleZ"]

    # dimensions
    zd, yd, xd = seg_mem.shape
    zde = int(round(zd * pixelScaleZ / pixelScaleX))

    # Parameters
    w1 = 20
    w2 = 20
    wb = 5
    w = w1 + wb + xd + w2 + wb + xd + w2 + wb + yd + w1
    h = 250
    hb = 5
    h12 = h - yd - hb
    h1 = np.floor(h12 / 2)
    # h2 = np.ceil(h12 / 2)
    h34 = h - zde - hb
    h3 = np.floor(h34 / 2)
    # h4 = np.ceil(h34 / 2)
    cmap = matplotlib.cm.get_cmap("YlOrRd")
    z_color = np.dot(255, cmap(0.15))
    y_color = np.dot(255, cmap(0.6))
    x_color = np.dot(255, cmap(0.8))

    wl = 30
    hl = 30
    xyzpic_root = Path("./polar_express/adddata")
    # x
    xpic = xyzpic_root / "x.tif"
    ximg = np.array(Image.open(xpic))
    ximg = np.squeeze(ximg[:, :, 0])
    ximg[ximg < 128] = 0
    ximg[ximg >= 128] = 255
    ximg = cv2.resize(ximg, dsize=(wl, hl), interpolation=cv2.INTER_NEAREST)
    xmat = np.zeros((hl, wl, 3), dtype="uint8")
    for yi in np.arange(hl):
        for xi in np.arange(wl):
            if ximg[yi, xi] == 0:
                xmat[yi, xi, :] = [150, 150, 150]
            elif ximg[yi, xi] == 255:
                xmat[yi, xi, :] = x_color[0:3]
    # y
    ypic = xyzpic_root / "y.tif"
    yimg = np.array(Image.open(ypic))
    yimg = np.squeeze(yimg[:, :, 0])
    yimg[yimg < 128] = 0
    yimg[yimg >= 128] = 255
    yimg = cv2.resize(yimg, dsize=(wl, hl), interpolation=cv2.INTER_NEAREST)
    ymat = np.zeros((hl, wl, 3), dtype="uint8")
    for yi in np.arange(hl):
        for xi in np.arange(wl):
            if yimg[yi, xi] == 0:
                ymat[yi, xi, :] = [150, 150, 150]
            elif yimg[yi, xi] == 255:
                ymat[yi, xi, :] = y_color[0:3]
    # z
    zpic = xyzpic_root / "z.tif"
    zimg = np.array(Image.open(zpic))
    zimg = np.squeeze(zimg[:, :, 0])
    zimg[zimg < 128] = 0
    zimg[zimg >= 128] = 255
    zimg = cv2.resize(zimg, dsize=(wl, hl), interpolation=cv2.INTER_NEAREST)
    zmat = np.zeros((hl, wl, 3), dtype="uint8")
    for yi in np.arange(hl):
        for xi in np.arange(wl):
            if zimg[yi, xi] == 0:
                zmat[yi, xi, :] = [150, 150, 150]
            elif zimg[yi, xi] == 255:
                zmat[yi, xi, :] = z_color[0:3]

    # XY array
    mem_XY_array = np.squeeze(seg_mem[center_of_nucleus[0], :, :])
    dna_XY_array = np.squeeze(seg_dna[center_of_nucleus[0], :, :])
    gfp_XY_array = np.squeeze(gfp_art[center_of_nucleus[0], :, :])

    XY_array = np.zeros((yd, xd, 3), dtype=np.uint8)
    XY_array[:, :, 0] = mem_XY_array
    XY_array[:, :, 1] = dna_XY_array
    XY_array[:, :, 2] = gfp_XY_array

    for yi in np.arange(yd):
        for xi in np.arange(xd):
            vec = XY_array[yi, xi, :]
            if (vec[0] == seg_mem_un[1]) & (vec[1] == seg_dna_un[1]):  # DNA
                XY_array[yi, xi, :] = DNA_color
            elif (vec[0] == seg_mem_un[0]) & (vec[1] == seg_dna_un[0]):  # outside
                XY_array[yi, xi, :] = OUT_color
            elif (vec[0] == seg_mem_un[1]) & (vec[1] == seg_dna_un[0]):  # cytoplasm
                val = XY_array[yi, xi, 2]
                XY_array[yi, xi, 0] = np.interp(val, gfp_minmax, GFP_colors[:, 0])
                XY_array[yi, xi, 1] = np.interp(val, gfp_minmax, GFP_colors[:, 1])
                XY_array[yi, xi, 2] = np.interp(val, gfp_minmax, GFP_colors[:, 2])
            else:
                1 / 0

    XY_array[center_of_nucleus[1], :, :] = XY_array[center_of_nucleus[1], :, :] + 25
    XY_array[:, center_of_nucleus[2], :] = XY_array[:, center_of_nucleus[2], :] + 25

    # XZ array
    mem_XZ_array = np.squeeze(seg_mem[:, center_of_nucleus[1], :])
    dna_XZ_array = np.squeeze(seg_dna[:, center_of_nucleus[1], :])
    gfp_XZ_array = np.squeeze(gfp_art[:, center_of_nucleus[1], :])

    XZ_array = np.zeros((zd, xd, 3), dtype=np.uint8)
    XZ_array[:, :, 0] = mem_XZ_array
    XZ_array[:, :, 1] = dna_XZ_array
    XZ_array[:, :, 2] = gfp_XZ_array

    for zi in np.arange(zd):
        for xi in np.arange(xd):
            vec = XZ_array[zi, xi, :]
            if (vec[0] == seg_mem_un[1]) & (vec[1] == seg_dna_un[1]):  # DNA
                XZ_array[zi, xi, :] = DNA_color
            elif (vec[0] == seg_mem_un[0]) & (vec[1] == seg_dna_un[0]):  # outside
                XZ_array[zi, xi, :] = OUT_color
            elif (vec[0] == seg_mem_un[1]) & (vec[1] == seg_dna_un[0]):  # cytoplasm
                val = XZ_array[zi, xi, 2]
                XZ_array[zi, xi, 0] = np.interp(val, gfp_minmax, GFP_colors[:, 0])
                XZ_array[zi, xi, 1] = np.interp(val, gfp_minmax, GFP_colors[:, 1])
                XZ_array[zi, xi, 2] = np.interp(val, gfp_minmax, GFP_colors[:, 2])
            else:
                print(seg_mem_un)
                print(seg_dna_un)
                print(vec)
                1 / 0

    XZ_array[center_of_nucleus[0], :, :] = XZ_array[center_of_nucleus[0], :, :] + 100

    XZ_array2 = np.zeros((zde, xd, 3), dtype=np.uint8)
    for i in np.arange(3):
        tm = np.squeeze(XZ_array[:, :, i]).T
        tm = cv2.resize(tm, dsize=(zde, xd), interpolation=cv2.INTER_NEAREST)
        XZ_array2[:, :, i] = tm.T

    # YZ
    mem_YZ_array = np.squeeze(seg_mem[:, :, center_of_nucleus[2]])
    dna_YZ_array = np.squeeze(seg_dna[:, :, center_of_nucleus[2]])
    gfp_YZ_array = np.squeeze(gfp_art[:, :, center_of_nucleus[2]])

    YZ_array = np.zeros((zd, yd, 3), dtype=np.uint8)
    YZ_array[:, :, 0] = mem_YZ_array
    YZ_array[:, :, 1] = dna_YZ_array
    YZ_array[:, :, 2] = gfp_YZ_array

    for zi in np.arange(zd):
        for yi in np.arange(yd):
            vec = YZ_array[zi, yi, :]
            if (vec[0] == seg_mem_un[1]) & (vec[1] == seg_dna_un[1]):  # DNA
                YZ_array[zi, yi, :] = DNA_color
            elif (vec[0] == seg_mem_un[0]) & (vec[1] == seg_dna_un[0]):  # outside
                YZ_array[zi, yi, :] = OUT_color
            elif (vec[0] == seg_mem_un[1]) & (vec[1] == seg_dna_un[0]):  # cytoplasm
                val = YZ_array[zi, yi, 2]
                YZ_array[zi, yi, 0] = np.interp(val, gfp_minmax, GFP_colors[:, 0])
                YZ_array[zi, yi, 1] = np.interp(val, gfp_minmax, GFP_colors[:, 1])
                YZ_array[zi, yi, 2] = np.interp(val, gfp_minmax, GFP_colors[:, 2])
            else:
                1 / 0

    YZ_array[center_of_nucleus[0], :, :] = YZ_array[center_of_nucleus[0], :, :] + 100

    YZ_array2 = np.zeros((zde, yd, 3), dtype=np.uint8)
    for i in np.arange(3):
        tm = np.squeeze(YZ_array[:, :, i]).T
        tm = cv2.resize(tm, dsize=(zde, yd), interpolation=cv2.INTER_NEAREST)
        YZ_array2[:, :, i] = tm.T

    plot_array = np.zeros((h, w, 3), dtype=np.uint8)
    plot_array[
        int(h1 + hb - 1) : int(h1 + hb + yd - 1),
        int(w1 + wb - 1) : int(w1 + wb + xd - 1),
        :,
    ] = XY_array
    plot_array[
        int(h3 + hb - 1) : int(h3 + hb + zde - 1),
        int(w1 + wb + xd + w2 + wb - 1) : int(w1 + wb + xd + w2 + wb + xd - 1),
        :,
    ] = XZ_array2
    plot_array[
        int(h3 + hb - 1) : int(h3 + hb + zde - 1),
        int(w1 + wb + xd + w2 + wb + xd + w2 + wb - 1) : int(
            w1 + wb + xd + w2 + wb + xd + w2 + wb + yd - 1
        ),
        :,
    ] = YZ_array2

    for i in np.arange(3):
        plot_array[
            int(h1 + hb - 1) : int(h1 + hb + yd - 1), int(w1 - 1) : int(w1 + wb - 1), i
        ] = y_color[i]
        plot_array[
            int(h1 - 1) : int(h1 + hb - 1), int(w1 + wb - 1) : int(w1 + wb + xd - 1), i
        ] = x_color[i]
        plot_array[
            int(h3 - 1) : int(h3 + hb - 1),
            int(w1 + wb + xd + w2 + wb - 1) : int(w1 + wb + xd + w2 + wb + xd - 1),
            i,
        ] = x_color[i]
        plot_array[
            int(h3 + hb - 1) : int(h3 + hb + zde - 1),
            int(w1 + wb + xd + w2 - 1) : int(w1 + wb + xd + w2 + wb - 1),
            i,
        ] = z_color[i]
        plot_array[
            int(h3 - 1) : int(h3 + hb - 1),
            int(w1 + wb + xd + w2 + wb + xd + w2 + wb - 1) : int(
                w1 + wb + xd + w2 + wb + xd + w2 + wb + yd - 1
            ),
            i,
        ] = y_color[i]
        plot_array[
            int(h3 + hb - 1) : int(h3 + hb + zde - 1),
            int(w1 + wb + xd + w2 + wb + xd + w2 - 1) : int(
                w1 + wb + xd + w2 + wb + xd + w2 + wb - 1
            ),
            i,
        ] = z_color[i]

    # x,y,z
    plot_array[
        int(h1 + hb + yd - 1) : int(h1 + hb + yd + hl - 1),
        int(w1 + wb + np.floor(xd / 2) - wl - 1) : int(w1 + wb + np.floor(xd / 2) - 1),
        :,
    ] = np.flip(xmat, axis=0)
    plot_array[
        int(h1 + hb + yd - 1) : int(h1 + hb + yd + hl - 1),
        int(w1 + wb + np.floor(xd / 2) - 1) : int(w1 + wb + np.floor(xd / 2) - 1) + wl,
        :,
    ] = np.flip(ymat, axis=0)
    plot_array[
        int(h3 + hb + zde - 1) : int(h3 + hb + zde + hl - 1),
        int(w1 + wb + xd + w2 + wb + np.floor(xd / 2) - wl - 1) : int(
            w1 + wb + xd + w2 + wb + np.floor(xd / 2) - 1
        ),
        :,
    ] = np.flip(xmat, axis=0)
    plot_array[
        int(h3 + hb + zde - 1) : int(h3 + hb + zde + hl - 1),
        int(w1 + wb + xd + w2 + wb + np.floor(xd / 2) - 1) : int(
            w1 + wb + xd + w2 + wb + np.floor(xd / 2) - 1
        )
        + wl,
        :,
    ] = np.flip(zmat, axis=0)

    plot_array[
        int(h3 + hb + zde - 1) : int(h3 + hb + zde + hl - 1),
        int(w1 + wb + xd + w2 + wb + xd + w2 + wb + np.floor(yd / 2) - wl - 1) : int(
            w1 + wb + xd + w2 + wb + xd + w2 + wb + np.floor(yd / 2) - 1
        ),
        :,
    ] = np.flip(ymat, axis=0)
    plot_array[
        int(h3 + hb + zde - 1) : int(h3 + hb + zde + hl - 1),
        int(w1 + wb + xd + w2 + wb + xd + w2 + wb + np.floor(yd / 2) - 1) : int(
            w1 + wb + xd + w2 + wb + xd + w2 + wb + np.floor(yd / 2) + wl - 1
        ),
        :,
    ] = np.flip(zmat, axis=0)

    plot_array = np.flip(plot_array, axis=0)

    fig, axes = plt.subplots()
    axes.imshow(plot_array)
    slicefile = artificial_plot_dir / f"CenterSlice_{cellid}_{adt}.png"
    fig.savefig(slicefile, format="png", dpi=1000)
    plt.close(fig)

    # slice builders

    # final plotting array
    Zplot_array = np.zeros((zd, h, w, 3), dtype=np.uint8)

    for slice in tqdm(np.arange(zd), "Slicing"):

        # XY array
        mem_XY_array = np.squeeze(seg_mem[slice, :, :])
        dna_XY_array = np.squeeze(seg_dna[slice, :, :])
        gfp_XY_array = np.squeeze(gfp_art[slice, :, :])

        XY_array = np.zeros((yd, xd, 3), dtype=np.uint8)
        XY_array[:, :, 0] = mem_XY_array
        XY_array[:, :, 1] = dna_XY_array
        XY_array[:, :, 2] = gfp_XY_array

        for yi in np.arange(yd):
            for xi in np.arange(xd):
                vec = XY_array[yi, xi, :]
                if (vec[0] == seg_mem_un[1]) & (vec[1] == seg_dna_un[1]):  # DNA
                    XY_array[yi, xi, :] = DNA_color
                elif (vec[0] == seg_mem_un[0]) & (vec[1] == seg_dna_un[0]):  # outside
                    XY_array[yi, xi, :] = OUT_color
                elif (vec[0] == seg_mem_un[1]) & (vec[1] == seg_dna_un[0]):  # cytoplasm
                    val = XY_array[yi, xi, 2]
                    XY_array[yi, xi, 0] = np.interp(val, gfp_minmax, GFP_colors[:, 0])
                    XY_array[yi, xi, 1] = np.interp(val, gfp_minmax, GFP_colors[:, 1])
                    XY_array[yi, xi, 2] = np.interp(val, gfp_minmax, GFP_colors[:, 2])
                else:
                    1 / 0

        XY_array[center_of_nucleus[1], :, :] = XY_array[center_of_nucleus[1], :, :] + 25
        XY_array[:, center_of_nucleus[2], :] = XY_array[:, center_of_nucleus[2], :] + 25

        # XZ array
        mem_XZ_array = np.squeeze(seg_mem[:, center_of_nucleus[1], :])
        dna_XZ_array = np.squeeze(seg_dna[:, center_of_nucleus[1], :])
        gfp_XZ_array = np.squeeze(gfp_art[:, center_of_nucleus[1], :])

        XZ_array = np.zeros((zd, xd, 3), dtype=np.uint8)
        XZ_array[:, :, 0] = mem_XZ_array
        XZ_array[:, :, 1] = dna_XZ_array
        XZ_array[:, :, 2] = gfp_XZ_array

        for zi in np.arange(zd):
            for xi in np.arange(xd):
                vec = XZ_array[zi, xi, :]
                if (vec[0] == seg_mem_un[1]) & (vec[1] == seg_dna_un[1]):  # DNA
                    XZ_array[zi, xi, :] = DNA_color
                elif (vec[0] == seg_mem_un[0]) & (vec[1] == seg_dna_un[0]):  # outside
                    XZ_array[zi, xi, :] = OUT_color
                elif (vec[0] == seg_mem_un[1]) & (vec[1] == seg_dna_un[0]):  # cytoplasm
                    val = XZ_array[zi, xi, 2]
                    XZ_array[zi, xi, 0] = np.interp(val, gfp_minmax, GFP_colors[:, 0])
                    XZ_array[zi, xi, 1] = np.interp(val, gfp_minmax, GFP_colors[:, 1])
                    XZ_array[zi, xi, 2] = np.interp(val, gfp_minmax, GFP_colors[:, 2])
                else:
                    1 / 0

        XZ_array[slice, :, :] = XZ_array[slice, :, :] + 100

        XZ_array2 = np.zeros((zde, xd, 3), dtype=np.uint8)
        for i in np.arange(3):
            tm = np.squeeze(XZ_array[:, :, i]).T
            tm = cv2.resize(tm, dsize=(zde, xd), interpolation=cv2.INTER_NEAREST)
            XZ_array2[:, :, i] = tm.T

        # YZ
        mem_YZ_array = np.squeeze(seg_mem[:, :, center_of_nucleus[2]])
        dna_YZ_array = np.squeeze(seg_dna[:, :, center_of_nucleus[2]])
        gfp_YZ_array = np.squeeze(gfp_art[:, :, center_of_nucleus[2]])

        YZ_array = np.zeros((zd, yd, 3), dtype=np.uint8)
        YZ_array[:, :, 0] = mem_YZ_array
        YZ_array[:, :, 1] = dna_YZ_array
        YZ_array[:, :, 2] = gfp_YZ_array

        for zi in np.arange(zd):
            for yi in np.arange(yd):
                vec = YZ_array[zi, yi, :]
                if (vec[0] == seg_mem_un[1]) & (vec[1] == seg_dna_un[1]):  # DNA
                    YZ_array[zi, yi, :] = DNA_color
                elif (vec[0] == seg_mem_un[0]) & (vec[1] == seg_dna_un[0]):  # outside
                    YZ_array[zi, yi, :] = OUT_color
                elif (vec[0] == seg_mem_un[1]) & (vec[1] == seg_dna_un[0]):  # cytoplasm
                    val = YZ_array[zi, yi, 2]
                    YZ_array[zi, yi, 0] = np.interp(val, gfp_minmax, GFP_colors[:, 0])
                    YZ_array[zi, yi, 1] = np.interp(val, gfp_minmax, GFP_colors[:, 1])
                    YZ_array[zi, yi, 2] = np.interp(val, gfp_minmax, GFP_colors[:, 2])
                else:
                    1 / 0

        YZ_array[slice, :, :] = YZ_array[slice, :, :] + 100

        YZ_array2 = np.zeros((zde, yd, 3), dtype=np.uint8)
        for i in np.arange(3):
            tm = np.squeeze(YZ_array[:, :, i]).T
            tm = cv2.resize(tm, dsize=(zde, yd), interpolation=cv2.INTER_NEAREST)
            YZ_array2[:, :, i] = tm.T

        plot_array = np.zeros((h, w, 3), dtype=np.uint8)
        plot_array[
            int(h1 + hb - 1) : int(h1 + hb + yd - 1),
            int(w1 + wb - 1) : int(w1 + wb + xd - 1),
            :,
        ] = XY_array
        plot_array[
            int(h3 + hb - 1) : int(h3 + hb + zde - 1),
            int(w1 + wb + xd + w2 + wb - 1) : int(w1 + wb + xd + w2 + wb + xd - 1),
            :,
        ] = XZ_array2
        plot_array[
            int(h3 + hb - 1) : int(h3 + hb + zde - 1),
            int(w1 + wb + xd + w2 + wb + xd + w2 + wb - 1) : int(
                w1 + wb + xd + w2 + wb + xd + w2 + wb + yd - 1
            ),
            :,
        ] = YZ_array2

        for i in np.arange(3):
            plot_array[
                int(h1 + hb - 1) : int(h1 + hb + yd - 1),
                int(w1 - 1) : int(w1 + wb - 1),
                i,
            ] = y_color[i]
            plot_array[
                int(h1 - 1) : int(h1 + hb - 1),
                int(w1 + wb - 1) : int(w1 + wb + xd - 1),
                i,
            ] = x_color[i]
            plot_array[
                int(h3 - 1) : int(h3 + hb - 1),
                int(w1 + wb + xd + w2 + wb - 1) : int(w1 + wb + xd + w2 + wb + xd - 1),
                i,
            ] = x_color[i]
            plot_array[
                int(h3 + hb - 1) : int(h3 + hb + zde - 1),
                int(w1 + wb + xd + w2 - 1) : int(w1 + wb + xd + w2 + wb - 1),
                i,
            ] = z_color[i]
            plot_array[
                int(h3 - 1) : int(h3 + hb - 1),
                int(w1 + wb + xd + w2 + wb + xd + w2 + wb - 1) : int(
                    w1 + wb + xd + w2 + wb + xd + w2 + wb + yd - 1
                ),
                i,
            ] = y_color[i]
            plot_array[
                int(h3 + hb - 1) : int(h3 + hb + zde - 1),
                int(w1 + wb + xd + w2 + wb + xd + w2 - 1) : int(
                    w1 + wb + xd + w2 + wb + xd + w2 + wb - 1
                ),
                i,
            ] = z_color[i]

        # x,y,z
        plot_array[
            int(h1 + hb + yd - 1) : int(h1 + hb + yd + hl - 1),
            int(w1 + wb + np.floor(xd / 2) - wl - 1) : int(
                w1 + wb + np.floor(xd / 2) - 1
            ),
            :,
        ] = np.flip(xmat, axis=0)
        plot_array[
            int(h1 + hb + yd - 1) : int(h1 + hb + yd + hl - 1),
            int(w1 + wb + np.floor(xd / 2) - 1) : int(w1 + wb + np.floor(xd / 2) - 1)
            + wl,
            :,
        ] = np.flip(ymat, axis=0)
        plot_array[
            int(h3 + hb + zde - 1) : int(h3 + hb + zde + hl - 1),
            int(w1 + wb + xd + w2 + wb + np.floor(xd / 2) - wl - 1) : int(
                w1 + wb + xd + w2 + wb + np.floor(xd / 2) - 1
            ),
            :,
        ] = np.flip(xmat, axis=0)
        plot_array[
            int(h3 + hb + zde - 1) : int(h3 + hb + zde + hl - 1),
            int(w1 + wb + xd + w2 + wb + np.floor(xd / 2) - 1) : int(
                w1 + wb + xd + w2 + wb + np.floor(xd / 2) - 1
            )
            + wl,
            :,
        ] = np.flip(zmat, axis=0)

        plot_array[
            int(h3 + hb + zde - 1) : int(h3 + hb + zde + hl - 1),
            int(
                w1 + wb + xd + w2 + wb + xd + w2 + wb + np.floor(yd / 2) - wl - 1
            ) : int(w1 + wb + xd + w2 + wb + xd + w2 + wb + np.floor(yd / 2) - 1),
            :,
        ] = np.flip(ymat, axis=0)
        plot_array[
            int(h3 + hb + zde - 1) : int(h3 + hb + zde + hl - 1),
            int(w1 + wb + xd + w2 + wb + xd + w2 + wb + np.floor(yd / 2) - 1) : int(
                w1 + wb + xd + w2 + wb + xd + w2 + wb + np.floor(yd / 2) + wl - 1
            ),
            :,
        ] = np.flip(zmat, axis=0)

        plot_array = np.flip(plot_array, axis=0)

        Zplot_array[slice, :, :, :] = plot_array

    Zplot_array.shape = 1, zd, 1, h, w, 3  # dimensions in TZCYXS order
    stackfile = artificial_plot_dir / f"Stack_{cellid}_{adt}.tiff"
    tifffile.imwrite(stackfile, Zplot_array, imagej=True)
