#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd

from datastep import Step, log_run_params

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

    @log_run_params
    def run(
        self,
        dataset='/allen/aics/modeling/theok/Projects/Data/Org3Dcells',
        **kwargs
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

        Returns
        -------
        selected_cells_manifest: Path
            Path to manifest file that contains a path to a CSV file with annotation of
            selected cells
        """

        # Directory assignments
        cell_annotation_dir = self.step_local_staging_dir / "annotation"
        cell_annotation_dir.mkdir(exist_ok=True)

        # Select ER cells
        # Point to master annotation file
        cell3D_root = Path(dataset)
        csvfile = cell3D_root / 'Annotation' / 'ann.csv'
        cells = pd.read_csv(csvfile)
        # Load in and select ER cells in interphase (stage = 0)
        selectedcells = cells[(cells['Interphase and Mitotic Stages (stage)'] == 0)
                              & (cells[('Structure')] == 'Endoplasmic reticulum')]
        # Save selected cells
        selected_cell_csv = cell_annotation_dir / 'ann_sc.csv'
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
