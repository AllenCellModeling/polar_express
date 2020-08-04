#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pathlib import Path
import numpy as np
from aicsimageio import imread
import pickle

#######################################################################################

def testChannelsEqual(test_image_metrics, true_image_metrics):

    assert np.alltrue(test_image_metrics['channels']['seg_dna']
                      == true_image_metrics['channels']['seg_dna'])
    assert np.alltrue(test_image_metrics['channels']['seg_mem']
                      == true_image_metrics['channels']['seg_mem'])
    assert np.alltrue(test_image_metrics['channels']['seg_gfp']
                      == true_image_metrics['channels']['seg_gfp'])
    assert np.alltrue(test_image_metrics['channels']['dna']
                      == true_image_metrics['channels']['dna'])
    assert np.alltrue(test_image_metrics['channels']['mem']
                      == true_image_metrics['channels']['mem'])
    assert np.alltrue(test_image_metrics['channels']['gfp']
                      == true_image_metrics['channels']['gfp'])

def testValuesNonNegative(test_image_metrics):

    assert np.alltrue(test_image_metrics['channels']['seg_dna'] >= 0)
    assert np.alltrue(test_image_metrics['channels']['seg_mem'] >= 0)
    assert np.alltrue(test_image_metrics['channels']['seg_gfp'] >= 0)
    assert np.alltrue(test_image_metrics['channels']['dna'] >= 0)
    assert np.alltrue(test_image_metrics['channels']['mem'] >= 0)
    assert np.alltrue(test_image_metrics['channels']['gfp'] >= 0)