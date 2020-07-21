from pathlib import Path
import numpy as np

# Third party
from aicsimageio import AICSImage, imread

def findVerticalCutoffs(im, masked_channels):
    """

    Helper method that finds the z-coordinate (height level) of the top of the cell, top of the nucleus, centroid
    of the nucleus, bottom of the nucleus, and bottom of the cell. Takes an image as a 4-dimensional matrix as
    input, and returns these values.

    Parameters
    ----------
    imgPath : Path object pointing to the image

    Returns
    -------
    bot_of_cell : z-coordinate of bottom-most cell membrane voxel
    bot_of_nucleus : z-coordinate of bottom-most nucleus voxel
    centroid_of_nucleus : z-coordinate of centroid of nucleus
    top_of_nucleus : z-coordinate of top-most nucleus voxel
    top_of_cell : z-coordinate of top-most cell membrane voxel

    """

    #from .applySegmentationMasks import applySegmentationMasks

    #im = np.squeeze(imread(imgPath))

    # z-stack indices of the 4 sections of the cell
    top_of_cell = None
    top_of_nucleus = None
    centroid_of_nucleus = np.zeros(3)
    bot_of_nucleus = None
    bot_of_cell = None

    # Get the segmentation channels
    #seg_dna, seg_mem, seg_gfp, dna, mem, gfp = applySegmentationMasks(im, channel_dict)

    # unpack masked channels
    seg_dna = masked_channels["seg_dna"]
    seg_mem = masked_channels["seg_mem"]
    seg_gfp = masked_channels["seg_gfp"]
    dna = masked_channels["dna"]
    mem = masked_channels["mem"]
    gfp = masked_channels["gfp"]

    num_z_stacks = im.shape[1]

    # Find where the cell membrane starts and ends
    for stack in range(1, num_z_stacks):
        prev_stack = stack - 1

        if (prev_stack == 0 and np.sum(seg_mem[prev_stack,:,:]) != 0): # The bottom of cell membrane is at the bottom of image
            bot_of_cell = prev_stack
        if (stack == num_z_stacks - 1 and np.sum(seg_mem[stack,:,:]) != 0): # The top of cell membrance is at the top of image
            top_of_cell = stack
        if (np.sum(seg_mem[prev_stack,:,:]) == 0 and np.sum(seg_mem[stack,:,:]) != 0): # The bottom layer of the cell membrane
            bot_of_cell = stack
        if (np.sum(seg_mem[prev_stack,:,:]) != 0 and np.sum(seg_mem[stack,:,:]) == 0): # The top layer of the cell membrane
            top_of_cell = prev_stack

    # Find where the nucleus starts and ends
    for stack in range(1, num_z_stacks):
        prev_stack = stack - 1
        if (np.sum(seg_dna[prev_stack,:,:]) == 0 and np.sum(seg_dna[stack,:,:]) != 0): # The bottom layer of the nucleus
            bot_of_nucleus = stack
        if (np.sum(seg_dna[prev_stack,:,:]) != 0 and np.sum(seg_dna[stack,:,:]) == 0): # The top layer of the nucleus
            top_of_nucleus = prev_stack

    # Find the centroid of the nucleus
    voxel_count = 0 # count the number of voxels belonging to the nucleus

    for stack in range(bot_of_nucleus, top_of_nucleus): # Iterate through the z-stacks that belong to the nucleus
        x_indices, y_indices = np.where(seg_dna[stack] != 0) # Get the x and y coordinates of the nucleus voxels of the current z-stack
        num_voxels_in_stack = x_indices.size
        voxel_count += num_voxels_in_stack # increment counter by the number of voxels in the current stack

        # Increment each centroid coordinate by the sum of the corresponding coordinates of this stack
        centroid_of_nucleus[0] += np.sum(x_indices)
        centroid_of_nucleus[1] += np.sum(y_indices)
        centroid_of_nucleus[2] += num_voxels_in_stack * stack

    # Find the mean of each dimension
    centroid_of_nucleus = centroid_of_nucleus / voxel_count

    # Check for unexpected positions of cell sections
    if (top_of_nucleus > top_of_cell or bot_of_nucleus < bot_of_cell):
        print('Nucleus exceeds boundaries of cell membrane!')
    if (centroid_of_nucleus[2] > top_of_nucleus or centroid_of_nucleus[2] < bot_of_nucleus):
        print('Centroid of nucleus exceeds boundaries of nucleus!')
    if (bot_of_cell > top_of_cell or bot_of_nucleus > top_of_nucleus):
        print('Bottom layer of organelle exceeds top layer!')
    if (any(elem is None for elem in [bot_of_cell, bot_of_nucleus, centroid_of_nucleus, top_of_nucleus, top_of_cell])):
        print('No value assigned to a vertical cutoff!')

    print('Empirical nuclear voxel count: ' + str(voxel_count))

    ### Temporary code to find volume of cell
    cell_voxel_count = 0

    for stack in range(bot_of_cell, top_of_cell): # Iterate through the z-stacks that belong to the cell
        x_indices, y_indices = np.where(seg_mem[stack] != 0) # Get the x and y coordinates of the nucleus voxels of the current z-stack
        num_voxels_in_stack = x_indices.size
        cell_voxel_count += num_voxels_in_stack # increment counter by the number of voxels in the current stack

    print('Empirical cell voxel count: ' + str(cell_voxel_count))
    ###


    # Return the results
    return bot_of_cell, bot_of_nucleus, centroid_of_nucleus, top_of_nucleus, top_of_cell