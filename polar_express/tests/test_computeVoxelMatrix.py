#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

#######################################################################################


def testMatricesEqual(test_image_metrics, true_image_metrics):

    assert np.alltrue(test_image_metrics['voxel_matrix']
                      == true_image_metrics['voxel_matrix'])
