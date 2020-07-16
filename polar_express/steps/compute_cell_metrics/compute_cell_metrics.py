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
            # image file
            file   = selected_cells['Path'].iloc[i]
            cellid = selected_cells['CellId'].iloc[i]
            # read in image file
            im = np.squeeze(imread(file))
            # compute metric
            metric = np.sum(im)
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
