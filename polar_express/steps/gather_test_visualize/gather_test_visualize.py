#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle

from datastep import Step, log_run_params

from ..compute_cell_metrics import ComputeCellMetrics
from .violinPlotHelper import makeViolinPlot

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class GatherTestVisualize(Step):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [ComputeCellMetrics],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(
        self,
        cell_metrics_manifest: Optional[Path] = None,
        filepath_column: str = "filepath",
        **kwargs
    ):
        """
        Gather, test, visualize

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
        cell_metrics_manifest: Optional[Path]
            Path to manifest file that contains a path to pickle files with the stored
            metrics for selected cells
            Default: self.step_local_staging_dir.parent / computecellmetrics
                     / manifest.csv
        filepath_column: str
            If providing a path to a csv manifest, the column to use for matrices.
            Default: "filepath"

        Returns
        -------
        visualizations_manifest: Path
            Path to manifest file that contains paths to image files of the generated
            visualizations
        """

        # Directory assignments
        vis_dir = self.step_local_staging_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)

        # Manifest from previous step
        if cell_metrics_manifest is None:
            cell_metrics_manifest = (
                self.step_local_staging_dir.parent / "computecellmetrics"
                / "manifest.csv"
            )

        # Load manifest (from Path to Dataframe)
        cell_metrics_manifest = pd.read_csv(cell_metrics_manifest)
        cell_pickles = cell_metrics_manifest[filepath_column]
        no_of_cells = len(cell_pickles)

        with (open(cell_pickles.iloc[0], "rb")) as openfile:
            curr_metrics = pickle.load(openfile)

        structure = curr_metrics['structure']

        if (curr_metrics['AB_mode'] == 'quadrants'):
            num_AB_compartments = 4
        elif (curr_metrics['AB_mode'] == 'hemispheres'):
            num_AB_compartments = 2
        else:
            raise Exception('Invalid mode entered. Use \'quadrants\' '
                            + 'or \'hemispheres\'.')

        num_angular_compartments = curr_metrics['num_angular_compartments']

        # Create storage data matrices for AB compartments
        AB_fc_storage = np.zeros([no_of_cells, num_AB_compartments])
        AB_cyto_storage = np.zeros([no_of_cells, num_AB_compartments])
        AB_gfp_storage = np.zeros([no_of_cells, num_AB_compartments])

        # Create storage data matrix for Angular compartments
        Angular_fc_storage = np.zeros([no_of_cells, num_angular_compartments])
        Angular_cyto_storage = np.zeros([no_of_cells, num_angular_compartments])
        Angular_gfp_storage = np.zeros([no_of_cells, num_angular_compartments])

        for index in tqdm(range(no_of_cells), desc="Generating Visualizations"):

            with (open(cell_pickles.iloc[index], "rb")) as openfile:
                curr_metrics = pickle.load(openfile)

            AB_fc_storage[index, :] = curr_metrics['AB_fold_changes']
            AB_cyto_storage[index, :] = curr_metrics['AB_cyto_vol']
            AB_gfp_storage[index, :] = curr_metrics['AB_gfp_intensities']

            Angular_fc_storage[index, :] = curr_metrics['Ang_fold_changes']
            Angular_cyto_storage[index, :] = curr_metrics['Ang_cyto_vol']
            Angular_gfp_storage[index, :] = curr_metrics['Ang_gfp_intensities']

        num_plots = 6

        # Configure manifest dataframe for storage tracking
        self.manifest = pd.DataFrame(index=range(num_plots), columns=["filepath"])

        i = 0

        for mode in ['AB', 'Angular']:
            for plot in ['FC', 'Cyto', 'GFP']:

                if (mode == 'AB'):
                    path = vis_dir / (mode + "_" + str(num_AB_compartments) + "_"
                                      + structure + "_" + plot + ".png")
                    if (plot == 'FC'):
                        makeViolinPlot(mode, num_AB_compartments, AB_fc_storage,
                                       'Fold Change (GFP to Cytoplasm Volume)',
                                       structure, no_of_cells, path)
                    elif (plot == 'Cyto'):
                        makeViolinPlot(mode, num_AB_compartments, AB_cyto_storage,
                                       'Cytoplasm Volume', structure, no_of_cells, path)
                    elif (plot == 'GFP'):
                        makeViolinPlot(mode, num_AB_compartments, AB_gfp_storage,
                                       'GFP Intensity', structure, no_of_cells, path)

                elif (mode == 'Angular'):
                    path = vis_dir / (mode + "_" + str(num_angular_compartments) + "_"
                                      + structure + "_" + plot + ".png")
                    if (plot == 'FC'):
                        makeViolinPlot(mode, num_angular_compartments,
                                       Angular_fc_storage,
                                       'Fold Change (GFP to Cytoplasm Volume)',
                                       structure, no_of_cells, path)
                    elif (plot == 'Cyto'):
                        makeViolinPlot(mode, num_angular_compartments,
                                       Angular_cyto_storage, 'Cytoplasm Volume',
                                       structure, no_of_cells, path)
                    elif (plot == 'GFP'):
                        makeViolinPlot(mode, num_angular_compartments,
                                       Angular_gfp_storage, 'GFP Intensity', structure,
                                       no_of_cells, path)

                # Add the path to the manifest
                self.manifest.at[i, "filepath"] = path
                i += 1

        # Save the manifest
        manifest_file = self.step_local_staging_dir / "manifest.csv"
        self.manifest.to_csv(manifest_file, index=False)

        return manifest_file
