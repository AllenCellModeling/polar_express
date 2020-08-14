#!/usr/bin/env python
# -*- coding: utf-8 -*-

from aicsimageio import imread
import numpy as np
import pickle

from .applySegmentationMasks import applySegmentationMasks
from .findVerticalCutoffs import findVerticalCutoffs
from .FoldChangeFunctions import findFoldChange_AB
from .FoldChangeFunctions import findFoldChange_Angular
from .computeVoxelMatrix import compute_voxel_matrix

from polar_express.definitions import ROOT_DIR


def computeMetricsDict(
    selected_cell, cell_metrics_dir, AB_mode, num_angular_compartments
):
    """
    Method to compute and save cell metrics dictionary for 3D input cell image

    Parameters
    ----------
    selected_cell : Pandas Series object with information about the selected cell
    cell_metrics_dir : Path to folder where the computed cell metrics dictionary
        should be stored
    AB_mode : str
        "quadrants" if AB compartments should split the cell into quadrants,
        "hemispheres" if AB compartments should split the cell into halves.
    num_angular_compartments : int
        The number of equal-size angles the cell should be split into for the
        angular compartment analysis.

    Returns
    -------
    pfile: Path
        Path to the pickle file of the computed cell metrics dictionary
    """

    # image file
    file = selected_cell["Path"]
    cellid = selected_cell["CellId"]

    # read in image file
    try:
        im = np.squeeze(imread(file))  # provided absolute path
    except FileNotFoundError:
        try:
            rel_path = ROOT_DIR + file  # provided relative path
            im = np.squeeze(imread(rel_path))
        except FileNotFoundError:
            raise

    # additional image information
    pixelScaleX = selected_cell["PixelScaleX"]
    pixelScaleY = selected_cell["PixelScaleY"]
    pixelScaleZ = selected_cell["PixelScaleZ"]
    vol_scale_factor = pixelScaleX * pixelScaleY * pixelScaleZ
    # pixel scale factors stored in (z,y,x) order
    scale_factors = np.array([pixelScaleZ, pixelScaleY, pixelScaleX])

    # get the channel indices
    ch_dna = selected_cell["ch_dna"]
    ch_memb = selected_cell["ch_memb"]
    ch_struct = selected_cell["ch_struct"]
    ch_seg_nuc = selected_cell["ch_seg_nuc"]
    ch_seg_cell = selected_cell["ch_seg_cell"]

    channel_indices = {
        "ch_dna": ch_dna,
        "ch_memb": ch_memb,
        "ch_struct": ch_struct,
        "ch_seg_nuc": ch_seg_nuc,
        "ch_seg_cell": ch_seg_cell,
    }

    # Get the segmentation channels
    (seg_dna, seg_mem, seg_gfp, dna, mem, gfp) = applySegmentationMasks(
        im, channel_indices
    )

    masked_channels = {
        "seg_dna": seg_dna,
        "seg_mem": seg_mem,
        "seg_gfp": seg_gfp,
        "dna": dna,
        "mem": mem,
        "gfp": gfp,
    }

    # compute z metrics
    (
        bot_of_cell,
        bot_of_nucleus,
        centroid_of_nucleus,
        top_of_nucleus,
        top_of_cell,
    ) = findVerticalCutoffs(im, masked_channels)

    z_metrics = {
        "bot_of_cell": bot_of_cell,
        "bot_of_nucleus": bot_of_nucleus,
        "centroid_of_nucleus": centroid_of_nucleus,
        "top_of_nucleus": top_of_nucleus,
        "top_of_cell": top_of_cell,
    }

    # compute fold change metrics
    (AB_fold_changes, AB_cyto_vol, AB_gfp_intensities) = findFoldChange_AB(
        masked_channels, z_metrics, vol_scale_factor, mode=AB_mode, silent=True
    )
    (Ang_fold_changes, Ang_cyto_vol, Ang_gfp_intensities,) = findFoldChange_Angular(
        masked_channels,
        z_metrics,
        vol_scale_factor,
        num_sections=num_angular_compartments,
        silent=True,
    )

    # compute (nx4) voxel matrix
    voxel_matrix = compute_voxel_matrix(
        scale_factors, centroid_of_nucleus, masked_channels
    )

    # store metrics
    metric = {
        "structure": selected_cell["Structure"],
        "vol_cell": np.sum(seg_mem > 0) * vol_scale_factor,
        "height_cell": (top_of_cell - bot_of_cell) * pixelScaleZ,
        "vol_nucleus": np.sum(seg_dna > 0) * vol_scale_factor,
        "height_nucleus": ((top_of_nucleus - bot_of_nucleus) * pixelScaleZ),
        "min_z_dna": bot_of_nucleus,
        "max_z_dna": top_of_nucleus,
        "min_z_cell": bot_of_cell,
        "max_z_cell": top_of_cell,
        "nuclear_centroid": centroid_of_nucleus,
        "total_dna_intensity": np.sum(dna),
        "total_mem_intensity": np.sum(mem),
        "total_gfp_intensity": np.sum(gfp),
        "AB_mode": AB_mode,
        "AB_fold_changes": AB_fold_changes,
        "AB_cyto_vol": AB_cyto_vol,
        "AB_gfp_intensities": AB_gfp_intensities,
        "num_angular_compartments": num_angular_compartments,
        "Ang_fold_changes": Ang_fold_changes,
        "Ang_cyto_vol": Ang_cyto_vol,
        "Ang_gfp_intensities": Ang_gfp_intensities,
        "x_dim": seg_dna.shape[2],
        "y_dim": seg_dna.shape[1],
        "z_dim": seg_dna.shape[0],
        "scale_factors": scale_factors,
        "voxel_matrix": voxel_matrix,
        "channels": masked_channels,
    }

    # save metric
    pfile = cell_metrics_dir / f"cell_{cellid}.pickle"
    if pfile.is_file():
        pfile.unlink()
    with open(pfile, "wb") as f:
        pickle.dump(metric, f)

    return pfile
