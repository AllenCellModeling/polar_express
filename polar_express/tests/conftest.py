#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
from aicsimageio import imread
import pickle
import pytest
import dask.dataframe as dd
import pandas as pd

from polar_express.steps import SelectData
from polar_express.steps import ComputeCellMetrics

###############################################################################


@pytest.fixture
def data_dir() -> Path:
    return Path(__file__).parent / "test_data"


@pytest.fixture
def data_metrics_dir() -> Path:
    # where the ground truth metrics for cell 30827 are stored as pickle files
    return Path(__file__).parent / "test_data_metrics"


@pytest.fixture
def new_data_metrics_dir() -> Path:
    # where the newly computed metrics for cell 30827 are stored and should be
    # accessed for testing
    return Path(__file__).parent / "new_data_metrics"


@pytest.fixture
def test_image(data_dir):
    # read in image file
    file = data_dir / "cell_30827.tiff"
    im = np.squeeze(imread(file))
    return im


@pytest.fixture
def true_image_metrics(data_metrics_dir):
    # read in ground truth metrics dictionary
    metrics_file = data_metrics_dir / "cell_30827.pickle"
    with (open(metrics_file, "rb")) as openfile:
        metrics = pickle.load(openfile)
    return metrics


@pytest.fixture
def test_image_metrics(new_data_metrics_dir):
    # read in computed metrics dictionary
    cell_30827 = new_data_metrics_dir / "cell_30827.pickle"
    with (open(cell_30827, "rb")) as openfile:
        metrics = pickle.load(openfile)
    return metrics


# Execute this once before running any tests
@pytest.fixture(scope="session", autouse=True)
def execute_before_any_test():

    # fixture with 'session' scope cannot take fixtures with 'function' scope
    data_dir = Path(__file__).parent / "test_data"
    data_metrics_dir = Path(__file__).parent / "test_data_metrics"
    new_data_metrics_dir = Path(__file__).parent / "new_data_metrics"

    # Initialize step
    step = SelectData()

    # Ensure that it still runs
    select_cells_manifest = step.run(data_dir)
    select_cells_manifest = dd.read_csv(select_cells_manifest)

    # Initialize step
    step = ComputeCellMetrics()

    # Ensure that it still runs
    cell_metrics_manifest = step.run(
        AB_mode="quadrants",
        num_angular_compartments=8,
        cell_metrics_dir=new_data_metrics_dir,
    )

    # Load manifest (from Path to Dataframe)
    cell_metrics_manifest = pd.read_csv(cell_metrics_manifest)

    # Save the manifest
    manifest_file = data_metrics_dir / "cell_metrics_manifest.csv"
    cell_metrics_manifest.to_csv(manifest_file, index=False)
