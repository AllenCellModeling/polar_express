#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

#######################################################################################


def testIsEqualAB(test_image_metrics, true_image_metrics):
    assert np.alltrue(test_image_metrics['AB_fold_changes']
                      == true_image_metrics['AB_fold_changes'])
    assert np.alltrue(test_image_metrics['AB_cyto_vol']
                      == true_image_metrics['AB_cyto_vol'])
    assert np.alltrue(test_image_metrics['AB_gfp_intensities']
                      == true_image_metrics['AB_gfp_intensities'])


def testIsEqualAngular(test_image_metrics, true_image_metrics):
    assert np.alltrue(test_image_metrics['Ang_fold_changes']
                      == true_image_metrics['Ang_fold_changes'])
    assert np.alltrue(test_image_metrics['Ang_cyto_vol']
                      == true_image_metrics['Ang_cyto_vol'])
    assert np.alltrue(test_image_metrics['Ang_gfp_intensities']
                      == true_image_metrics['Ang_gfp_intensities'])


def testNonNegativeAB(test_image_metrics, true_image_metrics):
    assert np.alltrue(test_image_metrics['AB_fold_changes'] >= 0)
    assert np.alltrue(test_image_metrics['AB_cyto_vol'] >= 0)
    assert np.alltrue(test_image_metrics['AB_gfp_intensities'] >= 0)


def testNonNegativeAngular(test_image_metrics, true_image_metrics):
    assert np.alltrue(test_image_metrics['Ang_fold_changes'] >= 0)
    assert np.alltrue(test_image_metrics['Ang_cyto_vol'] >= 0)
    assert np.alltrue(test_image_metrics['Ang_gfp_intensities'] >= 0)
