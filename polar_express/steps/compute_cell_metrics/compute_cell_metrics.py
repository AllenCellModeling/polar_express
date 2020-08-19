#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
from aics_dask_utils import DistributedHandler

from datastep import Step, log_run_params

from ..select_data import SelectData
from .metricsDict import computeMetricsDict

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

    @staticmethod
    def _compute_save_metrics(
        row_index: int,
        selectedcell: pd.Series,
        cell_metrics_dir: Path,
        AB_mode,
        num_angular_compartments,
    ) -> Path:
        # Compute cell metrics
        cellMetricsPath = computeMetricsDict(
            selectedcell, cell_metrics_dir, AB_mode, num_angular_compartments
        )

        return cellMetricsPath

    @log_run_params
    def run(
        self,
        selected_cells_manifest: Optional[Path] = None,
        filepath_column: str = "filepath",
        AB_mode="quadrants",
        num_angular_compartments=8,
        cell_metrics_dir=None,
        distributed_executor_address: Optional[str] = None,
        batch_size: Optional[int] = None,
        **kwargs,
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
            Path to manifest file that contains a path to a CSV file with annotation of
            selected cells
            Default: self.step_local_staging_dir.parent / selectdata / manifest.csv

        filepath_column: str
            If providing a path to a csv manifest, the column to use for matrices.
            Default: "filepath"

        AB_mode: str
            "quadrants" if AB compartments should split the cell into quadrants,
            "hemispheres" if AB compartments should split the cell into halves.

        num_angular_compartments : int
            The number of equal-size angles the cell should be split into for the
            angular compartment analysis.

        distributed_executor_address: Optional[str]
            An optional executor address to pass to some computation engine.
            Default: None

        batch_size: Optional[int]
            An optional batch size to process n cells at a time.
            Default: None (Process all at once)

        Returns
        -------
        cell_metrics_manifest: Path
            Path to manifest file that contains paths to compute metrics for each of the
            selected cells
        """

        # Directory assignments
        if cell_metrics_dir is None:
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

        print("Starting distributed run")
        # Process each row
        with DistributedHandler(distributed_executor_address) as handler:
            # Start processing
            results = handler.batched_map(
                self._compute_save_metrics,
                # Convert dataframe iterrows into two lists of items to iterate over
                # One list will be row index
                # One list will be the pandas series of every row
                *zip(*list(selected_cells.iterrows())),
                # Pass the other parameters as list of the same thing for each
                # mapped function call
                [cell_metrics_dir for i in range(no_of_cells)],
                [AB_mode for i in range(no_of_cells)],
                [num_angular_compartments for i in range(no_of_cells)],
                batch_size=batch_size,
            )

        # Gather paths to computed metrics dictionaries
        for index, result in enumerate(results):
            # Add the path to the manifest
            self.manifest.at[index, "filepath"] = result

        # Save the manifest
        manifest_file = self.step_local_staging_dir / "manifest.csv"
        self.manifest.to_csv(manifest_file, index=False)

        return manifest_file
