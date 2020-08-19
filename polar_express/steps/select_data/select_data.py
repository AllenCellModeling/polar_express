#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
from aics_dask_utils import DistributedHandler

from datastep import Step, log_run_params
from .artificialGFP import makeartificialGFP

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class SelectData(Step):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @staticmethod
    def _create_art_cells(
        row_index: int,
        selectedcell: pd.Series,
        artificial_cell_dir: Path,
        vizcells: List,
        artificial_plot_dir: Path,
    ) -> pd.DataFrame:
        # Create artificial cells
        art_cells = makeartificialGFP(
            selectedcell, artificial_cell_dir, vizcells, artificial_plot_dir
        )

        return art_cells

    @log_run_params
    def run(
        self,
        dataset="/allen/aics/modeling/theok/Projects/Data/Org3Dcells",
        artflag=False,
        distributed_executor_address: Optional[str] = None,
        batch_size: Optional[int] = None,
        **kwargs,
    ):
        """
        Select cells from annotation table

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
        dataset: str
            Path to the directory containing the .tiff files to be passed into the
            datastep workflow. The directory should also contain a sub-directory
            "Annotation" which holds ann.csv

        artflag: Optional[Bool]
            Flag indicating to run artificial GFP generation in cells
            Default: False

        distributed_executor_address: Optional[str]
            An optional executor address to pass to some computation engine.
            Default: None

        batch_size: Optional[int]
            An optional batch size to process n cells at a time.
            Default: None (Process all at once)

        Returns
        -------
        selected_cells_manifest: Path
            Path to manifest file that contains a path to a CSV file with annotation of
            selected cells
        """

        # Directory assignments
        cell_annotation_dir = self.step_local_staging_dir / "annotation"
        cell_annotation_dir.mkdir(exist_ok=True)

        # Point to master annotation file
        cell3D_root = Path(dataset)
        csvfile = cell3D_root / "Annotation" / "ann.csv"
        cells = pd.read_csv(csvfile)

        if artflag is True:

            # Additional directory assignments
            artificial_cell_dir = self.step_local_staging_dir / "artdata"
            artificial_cell_dir.mkdir(exist_ok=True)
            artificial_plot_dir = self.step_local_staging_dir / "artplots"
            artificial_plot_dir.mkdir(exist_ok=True)

            # Select artificial cells
            N = 5
            selectedcells = cells.sample(n=N, random_state=1)
            Nex = 2
            vizcells = list(selectedcells["CellId"].sample(n=Nex, random_state=1))

            print("Starting distributed run")

            # Process each row
            with DistributedHandler(distributed_executor_address) as handler:
                # Start processing
                results = handler.batched_map(
                    self._create_art_cells,
                    # Convert dataframe iterrows into two lists of items to iterate over
                    # One list will be row index
                    # One list will be the pandas series of every row
                    *zip(*list(selectedcells.iterrows())),
                    # Pass the other parameters as list of the same thing for each
                    # mapped function call
                    [artificial_cell_dir for i in range(len(dataset))],
                    [vizcells for i in range(len(dataset))],
                    [artificial_plot_dir for i in range(len(dataset))],
                    batch_size=batch_size,
                )

            # Generate features paths rows
            art_cells_compiled = pd.DataFrame()
            for result in results:
                art_cells_compiled = art_cells_compiled.append(
                    result, ignore_index=True
                )

            selected_cell_csv = cell_annotation_dir / "ann_sc.csv"
            art_cells_compiled.to_csv(selected_cell_csv)
            log.info(f"{len(selectedcells)} Art cells are selected")

            # Handling of the manifest
            # Configure manifest dataframe for storage tracking
            self.manifest = pd.DataFrame(index=range(1), columns=["filepath"])
            # Add the path to the manifest
            self.manifest.at[0, "filepath"] = selected_cell_csv
            # Save the manifest
            manifest_file = self.step_local_staging_dir / "manifest.csv"
            self.manifest.to_csv(manifest_file, index=False)

        else:

            # Select ER cells
            # Load in and select ER cells in interphase (stage = 0)
            selectedcells = cells[
                (cells["Interphase and Mitotic Stages (stage)"] == 0)
                & (cells[("Structure")] == "Endoplasmic reticulum")
            ]
            # Save selected cells
            selected_cell_csv = cell_annotation_dir / "ann_sc.csv"
            selectedcells.to_csv(selected_cell_csv)
            log.info(f"{len(selectedcells)} ER cells in interphase are selected")

            # Handling of the manifest
            # Configure manifest dataframe for storage tracking
            self.manifest = pd.DataFrame(index=range(1), columns=["filepath"])
            # Add the path to the manifest
            self.manifest.at[0, "filepath"] = selected_cell_csv
            # Save the manifest
            manifest_file = self.step_local_staging_dir / "manifest.csv"
            self.manifest.to_csv(manifest_file, index=False)

        return manifest_file
