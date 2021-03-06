#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script will run all tasks in a prefect Flow.
When you add steps to you step workflow be sure to add them to the step list
and configure their IO in the `run` function.
"""

import logging

import psutil
from distributed import LocalCluster
from prefect import Flow
from prefect.engine.executors import DaskExecutor, LocalExecutor
from polar_express import steps

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class All:
    def __init__(self):
        """
        Set all of your available steps here.
        This is only used for data logging operations, not computation purposes.
        """
        self.step_list = [
            steps.SelectData(),
            steps.ComputeCellMetrics(),
            steps.GatherTestVisualize(),
        ]

    def run(
        self,
        clean: bool = False,
        debug: bool = False,
        **kwargs,
    ):
        """
        Run a flow with your steps.
        Parameters
        ----------
        clean: bool
            Should the local staging directory be cleaned prior to this run.
            Default: False (Do not clean)
        debug: bool
            A debug flag for the developer to use to manipulate how much data runs,
            how it is processed, etc.
            Default: False (Do not debug)
        Notes
        -----
        Documentation on prefect:
        https://docs.prefect.io/core/
        Basic prefect example:
        https://docs.prefect.io/core/
        """
        # Initalize steps
        select_data = steps.SelectData()
        compute_cell_metrics = steps.ComputeCellMetrics()
        gather_test_visualize = steps.GatherTestVisualize()

        # Choose executor
        if debug:
            exe = LocalExecutor()
        else:

            # Create local cluster
            log.info("Creating LocalCluster")
            current_mem_gb = psutil.virtual_memory().available / 2 ** 30
            n_workers = int(current_mem_gb // 4)
            cluster = LocalCluster(n_workers=n_workers)
            log.info("Created LocalCluster")

            # Set distributed_executor_address
            distributed_executor_address = cluster.scheduler_address

            # Batch size on local cluster
            batch_size = int(psutil.cpu_count() // n_workers)

            # Log dashboard URI
            log.info(f"Dask dashboard available at: {cluster.dashboard_link}")

            # Use dask cluster
            exe = DaskExecutor(distributed_executor_address)

        # Configure your flow
        with Flow("polar_express") as flow:
            # If you want to clean the local staging directories pass clean
            # If you want to utilize some debugging functionality pass debug
            # If you don't utilize any of these, just pass the parameters you need.

            # step 1: select cells and store in annotation file
            selected_cells_manifest = select_data(
                clean=clean,
                debug=debug,
                distributed_executor_address=distributed_executor_address,
                batch_size=batch_size,
                **kwargs,  # Allows us to pass `--n {some integer}` or other params
            )

            # step 2: compute metrics for each of the cells
            cell_metrics_manifest = compute_cell_metrics(
                selected_cells_manifest,
                clean=clean,
                debug=debug,
                distributed_executor_address=distributed_executor_address,
                batch_size=batch_size,
                **kwargs,  # Allows us to pass `--n {some integer}` or other params
            )

            # step 3: gather the computed metrics and create visualizations
            gather_test_visualize(
                cell_metrics_manifest,
                clean=clean,
                debug=debug,
                **kwargs,  # Allows us to pass `--n {some integer}` or other params
            )

        # Run flow and get ending state
        state = flow.run(executor=exe)

        # Get and display any outputs you want to see on your local terminal
        log.info(select_data.get_result(state, flow))
        log.info(compute_cell_metrics.get_result(state, flow))
        log.info(gather_test_visualize.get_result(state, flow))

    def pull(self):
        """
        Pull all steps.
        """
        for step in self.step_list:
            step.pull()

    def checkout(self):
        """
        Checkout all steps.
        """
        for step in self.step_list:
            step.checkout()

    def push(self):
        """
        Push all steps.
        """
        for step in self.step_list:
            step.push()

    def clean(self):
        """
        Clean all steps.
        """
        for step in self.step_list:
            step.clean()
