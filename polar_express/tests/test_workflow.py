#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pathlib import Path
import dask.dataframe as dd

from polar_express.steps import SelectData
from polar_express.steps import ComputeCellMetrics
from polar_express.steps import GatherTestVisualize

#######################################################################################


@pytest.fixture
def select_data_manifest(data_dir):
    # Initialize step
    step = SelectData()

    # Ensure that it still runs
    output_manifest = step.run(data_dir)
    output_manifest = dd.read_csv(output_manifest)
    return output_manifest


@pytest.fixture
def art_select_data_manifest(data_dir):
    # Initialize step
    step = SelectData()

    # Run with artificial cell setting
    output_manifest = step.run(dataset=data_dir, artflag=True)
    output_manifest = dd.read_csv(output_manifest)
    return output_manifest


@pytest.fixture
def cell_metrics_manifest_1(select_data_manifest):
    # Initialize step
    step = ComputeCellMetrics()

    # Ensure that it still runs
    output_manifest = step.run(AB_mode="hemispheres", num_angular_compartments=2)
    output_manifest = dd.read_csv(output_manifest)
    return output_manifest


@pytest.fixture
def cell_metrics_manifest_2(select_data_manifest):
    # Initialize step
    step = ComputeCellMetrics()

    # Ensure that it still runs
    output_manifest = step.run(AB_mode="quadrants", num_angular_compartments=8)
    output_manifest = dd.read_csv(output_manifest)
    return output_manifest


@pytest.fixture
def gtv_manifest(cell_metrics_manifest_1):
    # Initialize step
    step = GatherTestVisualize()

    # Ensure that it still runs
    output_manifest = step.run()
    output_manifest = dd.read_csv(output_manifest)
    return output_manifest


def test_selectData(data_dir, select_data_manifest):

    # Run asserts

    # Check expected columns
    assert all(
        expected_col in select_data_manifest.columns for expected_col in ["filepath"]
    )

    # Check output length
    assert len(select_data_manifest) == 1

    # Check all expected files exist
    assert all(Path(f).resolve(strict=True) for f in select_data_manifest["filepath"])


def test_selectArtificialData(art_select_data_manifest):

    # Run asserts

    # Check expected columns
    assert all(
        expected_col in art_select_data_manifest.columns
        for expected_col in ["filepath"]
    )

    # Check output length
    assert len(art_select_data_manifest) == 1

    # Check all expected files exist
    assert all(
        Path(f).resolve(strict=True) for f in art_select_data_manifest["filepath"]
    )


# AB compartments: hemispheres, Angular compartments: 2
# We expect results to be the same, to a certain degree of significance
def test_computeCellMetrics_setting1(cell_metrics_manifest_1):

    # Run asserts

    # Check expected columns
    assert all(
        expected_col in cell_metrics_manifest_1.columns for expected_col in ["filepath"]
    )

    # Check output length
    assert len(cell_metrics_manifest_1) == 3

    # Check all expected files exist
    assert all(
        Path(f).resolve(strict=True) for f in cell_metrics_manifest_1["filepath"]
    )


# AB compartments: quadrants, Angular compartments: 8
# We do not expect results to be the same
def test_computeCellMetrics_setting2(cell_metrics_manifest_2):

    # Run asserts

    # Check expected columns
    assert all(
        expected_col in cell_metrics_manifest_2.columns for expected_col in ["filepath"]
    )

    # Check output length
    assert len(cell_metrics_manifest_2) == 3

    # Check all expected files exist
    assert all(
        Path(f).resolve(strict=True) for f in cell_metrics_manifest_2["filepath"]
    )


def test_gatherTestVisualize(gtv_manifest):

    # Run asserts

    # Check expected columns
    assert all(expected_col in gtv_manifest.columns for expected_col in ["filepath"])

    # Check output length
    assert len(gtv_manifest) == 6  # The expected number of visualizations generated

    # Check all expected files exist
    assert all(Path(f).resolve(strict=True) for f in gtv_manifest["filepath"])
