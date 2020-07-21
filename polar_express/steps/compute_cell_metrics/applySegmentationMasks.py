from pathlib import Path
import numpy as np

# Third party
from aicsimageio import AICSImage, imread

def applySegmentationMasks(im, channel_indices):
    """

    Applies each availible segmentation mask to the corresponding channel.
    Takes an image (in the form of a 4-dimensional matrix) as input and returns the dna, membrane,
    and gfp channels after the segmentation has been applied to the image.

    Parameters
    ----------
    imgPath : Path object pointing to the image to be segmented
    annPath: Path object pointing to the annotation file of the image

    Returns
    -------
    seg_dna : segmented DNA channel
    seg_mem : segmented cell membrane channel
    seg_gfp : segmented GFP channel
    dna : unsegmented DNA channel
    mem : unsegmented cell membrane channel
    gfp : unsegmented GFP channel

    """

    #im = np.squeeze(imread(imgPath))

    # unpack channel indices
    ch_dna = channel_indices["ch_dna"]
    ch_memb = channel_indices["ch_memb"]
    ch_struct = channel_indices["ch_struct"]
    ch_seg_nuc = channel_indices["ch_seg_nuc"]
    ch_seg_cell = channel_indices["ch_seg_cell"]

    # dna channel
    dna = np.squeeze(im[ch_dna, :, :, :])
    # membrane channel
    mem = np.squeeze(im[ch_memb, :, :, :])
    # GFP channel
    gfp = np.squeeze(im[ch_struct, :, :, :])

    # get rid of intensity values outside of segmentation
    seg_dna = np.squeeze(im[ch_seg_nuc, :, :, :])
    dna = np.multiply(seg_dna > 0, dna)
    seg_mem = np.squeeze(im[ch_seg_cell, :, :, :])
    mem = np.multiply(seg_mem > 0, mem)
    seg_gfp = (seg_mem > 0) & (seg_dna <= 0)
    gfp = np.multiply(seg_gfp, gfp)

    # Pay attention to the order of return values
    return seg_dna, seg_mem, seg_gfp, dna, mem, gfp