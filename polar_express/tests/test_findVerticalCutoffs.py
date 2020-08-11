#!/usr/bin/env python
# -*- coding: utf-8 -*-


#######################################################################################


def testVerticalCutoffsEqual(test_image_metrics, true_image_metrics):

    assert test_image_metrics["min_z_dna"] == true_image_metrics["min_z_dna"]
    assert test_image_metrics["max_z_dna"] == true_image_metrics["max_z_dna"]
    assert test_image_metrics["min_z_cell"] == true_image_metrics["min_z_cell"]
    assert test_image_metrics["max_z_cell"] == true_image_metrics["max_z_cell"]
    assert (
        test_image_metrics["nuclear_centroid"] == true_image_metrics["nuclear_centroid"]
    )
