#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

#######################################################################################


def testVoxelMatricesEqual(test_image_metrics, true_image_metrics):

    test_matrix = test_image_metrics["voxel_matrix"]
    true_matrix = true_image_metrics["voxel_matrix"]

    assert test_matrix.shape == true_matrix.shape

    matrix_diff = np.max(np.abs(test_matrix - true_matrix))

    assert matrix_diff == 0
