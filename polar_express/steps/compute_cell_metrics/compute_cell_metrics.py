#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
from tqdm import tqdm
from aicsimageio import AICSImage, imread
import numpy as np
import pickle

from datastep import Step, log_run_params

from ..select_data import SelectData
from .applySegmentationMasks import applySegmentationMasks
from .findVerticalCutoffs import findVerticalCutoffs
from .FoldChangeFunctions import findFoldChange_AB
from .FoldChangeFunctions import findFoldChange_Angular

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class ComputeCellMetrics(Step):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [SelectData],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(
        self,
        selected_cells_manifest: Optional[Path] = None,
        filepath_column: str = "filepath",
        AB_mode = "quadrants",
        num_angular_compartments = 8,
        **kwargs
    ):
        """
        Compute cell metrics

        Protected Parameters
        --------------------
        distributed_executor_address: Optional[str]
            An optional executor address to pass to some computation engine.
        clean: bool
            Should the local staging directory be cleaned prior to this run.
            Default: False (Do not clean)
        debug: bool
            A debug flag for the developer to use to manipulate how much data runs,
            how it is processed, etc.
            Default: False (Do not debug)

        Parameters
        ----------
        selected_cells_manifest: Optional[Path]
            Path to manifest file that contains a path to a CSV file with annotation of selected cells
            Default: self.step_local_staging_dir.parent / selectdata / manifest.csv
        filepath_column: str
            If providing a path to a csv manifest, the column to use for matrices.
            Default: "filepath"

        Returns
        -------
        cell_metrics_manifest: Path
            Path to manifest file that contains paths to compute metrics for each of the selected cells
        """

        # Directory assignments
        cell_metrics_dir = self.step_local_staging_dir / "metrics"
        cell_metrics_dir.mkdir(exist_ok=True)

        # Manifest from previous step
        if selected_cells_manifest is None:
            selected_cells_manifest = (
                self.step_local_staging_dir.parent / "selectdata" / "manifest.csv"
            )

        # Load manifest (from Path to Dataframe)
        selected_cells_manifest = pd.read_csv(selected_cells_manifest)

        # Load selected cells (from Path to Dataframe)
        selected_cells = selected_cells_manifest[filepath_column].iloc[0]
        selected_cells = pd.read_csv(selected_cells)

        no_of_cells = len(selected_cells)

        # Configure manifest dataframe for storage tracking
        self.manifest = pd.DataFrame(index=range(no_of_cells), columns=["filepath"])

        # Main loop to create
        for i in tqdm(range(no_of_cells), desc="Computing metrics for cells"):
        #for i in tqdm(range(2), desc="Computing metrics for cells"):
            # image file
            file   = selected_cells['Path'].iloc[i]
            cellid = selected_cells['CellId'].iloc[i]

            # read in image file
            im = np.squeeze(imread(file))

            # additional image information
            pixelScaleX = selected_cells['PixelScaleX'].iloc[i]
            pixelScaleY = selected_cells['PixelScaleY'].iloc[i]
            pixelScaleZ = selected_cells['PixelScaleZ'].iloc[i]
            vol_scale_factor = pixelScaleX * pixelScaleY * pixelScaleZ

            # get the channel indices
            ch_dna = selected_cells['ch_dna'].iloc[i]
            ch_memb = selected_cells['ch_memb'].iloc[i]
            ch_struct = selected_cells['ch_struct'].iloc[i]
            ch_seg_nuc = selected_cells['ch_seg_nuc'].iloc[i]
            ch_seg_cell = selected_cells['ch_seg_cell'].iloc[i]

            channel_indices = {"ch_dna" : ch_dna,
                               "ch_memb" : ch_memb,
                               "ch_struct" : ch_struct,
                               "ch_seg_nuc" : ch_seg_nuc,
                               "ch_seg_cell" : ch_seg_cell}

            # Get the segmentation channels
            seg_dna, seg_mem, seg_gfp, dna, mem, gfp = applySegmentationMasks(im, channel_indices)

            masked_channels = {"seg_dna" : seg_dna,
                               "seg_mem" : seg_mem,
                               "seg_gfp" : seg_gfp,
                               "dna" : dna,
                               "mem" : mem,
                               "gfp" : gfp}

            # compute nucleus metrics
            bot_of_cell, bot_of_nucleus, centroid_of_nucleus, top_of_nucleus, top_of_cell = findVerticalCutoffs(im, masked_channels)

            nucleus_metrics = {"bot_of_cell" : bot_of_cell,
                               "bot_of_nucleus" : bot_of_nucleus,
                               "centroid_of_nucleus" : centroid_of_nucleus,
                               "top_of_nucleus" : top_of_nucleus,
                               "top_of_cell" : top_of_cell}

            # compute fold change metrics
            AB_fold_changes, AB_cyto_vol, AB_gfp_intensities = findFoldChange_AB(im, masked_channels, nucleus_metrics,
                                                                                 vol_scale_factor, mode=AB_mode)
            Ang_fold_changes, Ang_cyto_vol, Ang_gfp_intensities = findFoldChange_Angular(im, masked_channels, nucleus_metrics,
                                                                                         vol_scale_factor, num_sections=num_angular_compartments)

            # store metrics
            metric = {"structure" : selected_cells['Structure'].iloc[i],
                      "vol_cell" : np.sum(seg_mem) * vol_scale_factor,
                      "height_cell" : (top_of_cell - bot_of_cell) * pixelScaleZ,
                      "vol_nucleus" : np.sum(seg_dna) * vol_scale_factor,
                      "height_nucleus" : (top_of_nucleus - bot_of_nucleus) * pixelScaleZ,
                      "min_z_dna" : bot_of_nucleus,
                      "max_z_dna" : top_of_nucleus,
                      "min_z_cell" : bot_of_cell,
                      "max_z_cell" : top_of_cell,
                      "nuclear_centroid" : centroid_of_nucleus,
                      "total_dna_intensity" : np.sum(dna),
                      "total_mem_intensity" : np.sum(mem),
                      "total_gfp_intensity" : np.sum(gfp),
                      "AB_mode" : AB_mode,
                      "AB_fold_changes" : AB_fold_changes,
                      "AB_cyto_vol" : AB_cyto_vol,
                      "AB_gfp_intensities" : AB_gfp_intensities,
                      "num_angular_compartments" : num_angular_compartments,
                      "Ang_fold_changes" : Ang_fold_changes,
                      "Ang_cyto_vol" : Ang_cyto_vol,
                      "Ang_gfp_intensities" : Ang_gfp_intensities
                      }

            # save metric
            pfile = cell_metrics_dir / f"cell_{cellid}.pickle"
            if pfile.is_file():
                pfile.unlink()
            with open(pfile, "wb") as f:
                pickle.dump(metric, f)
            # Add the path to the manifest
            self.manifest.at[i, "filepath"] = pfile

        # Save the manifest
        manifest_file = self.step_local_staging_dir / "manifest.csv"
        self.manifest.to_csv(manifest_file, index=False)

        return manifest_file
