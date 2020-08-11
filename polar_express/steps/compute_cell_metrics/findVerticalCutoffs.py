import numpy as np
from scipy import ndimage


def findVerticalCutoffs(im, masked_channels):
    """

    Helper method that finds the z-coordinate (height level) of the top of the cell, top
    of the nucleus, centroid of the nucleus, bottom of the nucleus, and bottom of the
    cell. Takes an image as a 4-dimensional matrix as input, and returns these values.

    Parameters
    ----------
    im : image of the cell to be analyzed, in the form of a 4D numpy array.
    masked_channels : dictionary containing the dna, membrane, and gfp channels, as
        well as their segmentation channels.

    Returns
    -------
    bot_of_cell : z-coordinate of bottom-most cell membrane voxel
    bot_of_nucleus : z-coordinate of bottom-most nucleus voxel
    centroid_of_nucleus : tuple respresenting coordinates of centroid of nucleus in the
        form (z,y,x)
    top_of_nucleus : z-coordinate of top-most nucleus voxel
    top_of_cell : z-coordinate of top-most cell membrane voxel

    """

    # z-stack indices of the 4 sections of the cell
    top_of_cell = None
    top_of_nucleus = None
    centroid_of_nucleus = None  # np.zeros(3)
    bot_of_nucleus = None
    bot_of_cell = None

    # unpack masked channels
    seg_dna = masked_channels["seg_dna"]
    seg_mem = masked_channels["seg_mem"]

    num_z_stacks = im.shape[1]

    # Find where the cell membrane starts and ends
    for stack in range(1, num_z_stacks):

        prev_stack = stack - 1

        # The bottom of cell membrane is at the bottom of image
        if prev_stack == 0 and np.sum(seg_mem[prev_stack, :, :]) != 0:
            bot_of_cell = prev_stack

        # The top of cell membrance is at the top of image
        if stack == num_z_stacks - 1 and np.sum(seg_mem[stack, :, :]) != 0:
            top_of_cell = stack

        # The bottom layer of the cell membrane
        if np.sum(seg_mem[prev_stack, :, :]) == 0 and np.sum(seg_mem[stack, :, :]) != 0:
            bot_of_cell = stack

        # The top layer of the cell membrane
        if np.sum(seg_mem[prev_stack, :, :]) != 0 and np.sum(seg_mem[stack, :, :]) == 0:
            top_of_cell = prev_stack

    # Find where the nucleus starts and ends
    for stack in range(1, num_z_stacks):

        prev_stack = stack - 1

        # The bottom layer of the nucleus
        if np.sum(seg_dna[prev_stack, :, :]) == 0 and np.sum(seg_dna[stack, :, :]) != 0:
            bot_of_nucleus = stack

        # The top layer of the nucleus
        if np.sum(seg_dna[prev_stack, :, :]) != 0 and np.sum(seg_dna[stack, :, :]) == 0:
            top_of_nucleus = prev_stack

    # Find the centroid of the nucleus
    centroid_of_nucleus = ndimage.measurements.center_of_mass(seg_dna)

    # Check for unexpected positions of cell sections
    if top_of_nucleus > top_of_cell or bot_of_nucleus < bot_of_cell:
        print("Nucleus exceeds boundaries of cell membrane!")

    if (
        centroid_of_nucleus[0] > top_of_nucleus
        or centroid_of_nucleus[0] < bot_of_nucleus
    ):
        print("Centroid of nucleus exceeds boundaries of nucleus!")

    if bot_of_cell > top_of_cell or bot_of_nucleus > top_of_nucleus:
        print("Bottom layer of organelle exceeds top layer!")

    if any(
        elem is None
        for elem in [
            bot_of_cell,
            bot_of_nucleus,
            centroid_of_nucleus,
            top_of_nucleus,
            top_of_cell,
        ]
    ):
        print("No value assigned to a vertical cutoff!")

    # Return the results
    return bot_of_cell, bot_of_nucleus, centroid_of_nucleus, top_of_nucleus, top_of_cell
