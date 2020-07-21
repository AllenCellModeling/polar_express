import numpy as np

# Helper method that finds and returns, for each of the cell sections, the fold change of the proportion of gfp intensity
# to cytoplasm volume, as well as the corresponding cytoplasm volume and gfp intensity data. Takes the image with which to
# find the fold change as input. If mode = 'quadrants', makes computations based on the cell being divided into 4 vertical
# quadrants. If mode = 'hemispheres', makes computations based on the cell being divided into 2 vertical quadrants.
def findFoldChange_AB(im, masked_channels, nucleus_metrics, vol_scale_factor, mode='quadrants'):
    if (mode == 'quadrants'):
        num_compartments = 4
    elif (mode == 'hemispheres'):
        num_compartments = 2
    else:
        raise Exception('Invalid mode entered. Please use \'quadrants\' or \'hemispheres\'.')

    cyto_vol = np.zeros(num_compartments) # cyto_vol[i] is the total number of voxels between dna_seq and mem_seq for section "i"
    gfp_intensities = np.zeros(num_compartments) # gfp_intensities[i] is the total summed intensity of gfp channel for section"i"
    top_stack = None # Keeps track of top layer of the current section
    bot_stack = None # Keeps track of bottom layer of the current section

    # Unpack masked segmentation channels
    seg_dna = masked_channels["seg_dna"]
    seg_mem = masked_channels["seg_mem"]
    seg_gfp = masked_channels["seg_gfp"]
    dna = masked_channels["dna"]
    mem = masked_channels["mem"]
    gfp = masked_channels["gfp"]

    # Unpack nucleus metrics
    bot_of_cell = nucleus_metrics["bot_of_cell"]
    bot_of_nucleus = nucleus_metrics["bot_of_nucleus"]
    centroid_of_nucleus = nucleus_metrics["centroid_of_nucleus"]
    top_of_nucleus = nucleus_metrics["top_of_nucleus"]
    top_of_cell = nucleus_metrics["top_of_cell"]

    # Find top and bottom of current z-stack
    for section in range(num_compartments):
        if (mode == 'quadrants'):
            if (section == 0): # top quarter of cell
                top_stack = top_of_cell
                bot_stack = top_of_nucleus
            elif (section == 1): # top half of nucleus
                top_stack = top_of_nucleus
                bot_stack = int(centroid_of_nucleus[2])
            elif (section == 2): # bottom half of nucleus
                top_stack = int(centroid_of_nucleus[2])
                bot_stack = bot_of_nucleus
            else: # bottom quarter of cell
                top_stack = bot_of_nucleus
                bot_stack = bot_of_cell
        else: # Mode == 'hemispheres'
            if (section == 0): # top half of cell
                top_stack = top_of_cell
                bot_stack = int(centroid_of_nucleus[2])
            else: # bottom half of cell
                top_stack = int(centroid_of_nucleus[2])
                bot_stack = bot_of_cell

        for stack in range(bot_stack, top_stack):
            # Increment by number of voxels between cell membrane and nucleus
            cyto_vol[section] += np.count_nonzero(seg_mem[stack,:,:] > 0) - np.count_nonzero(seg_dna[stack,:,:] > 0)
            # Increment by total gfp intensity of this stack
            gfp_intensities[section] += np.sum(gfp[stack,:,:])

    # Normalize
    cyto_vol = cyto_vol / np.sum(cyto_vol) 
    cyto_vol = cyto_vol * vol_scale_factor # Scale volume by pixel scale factor
    gfp_intensities = gfp_intensities / np.sum(gfp_intensities)

    # Calculate the Fold Change for each Section
    fold_changes = np.zeros(num_compartments)

    for section in range(num_compartments):
        if (cyto_vol[section] != 0 and gfp_intensities[section] != 0): # Some sections have cytoplasm volume of 0
            fold_changes[section] = np.log2(gfp_intensities[section] / cyto_vol[section])
        elif (cyto_vol[section] == 0):
            print('Cytoplasm volume of 0 detected')
        elif (gfp_intensities[section] == 0):
            print('GFP intensity of 0 detected')

    return fold_changes, cyto_vol, gfp_intensities


def findFoldChange_Angular(im, masked_channels, nucleus_metrics, vol_scale_factor, num_sections):
    # Unpack masked segmentation channels
    seg_dna = masked_channels["seg_dna"]
    seg_mem = masked_channels["seg_mem"]
    seg_gfp = masked_channels["seg_gfp"]
    dna = masked_channels["dna"]
    mem = masked_channels["mem"]
    gfp = masked_channels["gfp"]

    # Unpack nucleus metrics
    bot_of_cell = nucleus_metrics["bot_of_cell"]
    bot_of_nucleus = nucleus_metrics["bot_of_nucleus"]
    centroid_of_nucleus = nucleus_metrics["centroid_of_nucleus"]
    top_of_nucleus = nucleus_metrics["top_of_nucleus"]
    top_of_cell = nucleus_metrics["top_of_cell"]

    x = im[0,:,:,:].shape[2] #104  # x-dimension of 3d cell image
    y = im[0,:,:,:].shape[1] #168  # y-dimension of 3d cell image
    z = im[0,:,:,:].shape[0] #64  # z-dimension of 3d cell image

    # Make 3d matrices for x,y,z channels

    # Make x shape
    xvec = 1 + np.arange(x)
    xmat = np.repeat(xvec[:, np.newaxis], y, axis=1)
    xnd = np.repeat(xmat[:, :, np.newaxis], z, axis=2)
    # Make y shape
    yvec = 1 + np.arange(y)
    ymat = np.repeat(yvec[np.newaxis, :], x, axis=0)
    ynd = np.repeat(ymat[:, :, np.newaxis], z, axis=2)

    # Make z shape
    zvec = 1 + np.arange(z)
    zmat = np.repeat(zvec[np.newaxis, :], y, axis=0)
    znd = np.repeat(zmat[np.newaxis, :, :], x, axis=0)

    xnd = xnd - centroid_of_nucleus[0]
    ynd = ynd - centroid_of_nucleus[1]
    znd = znd - centroid_of_nucleus[2]

    res = znd / np.sqrt(xnd ** 2 + ynd ** 2 + znd ** 2)

    cyto_vol = np.zeros(num_sections) # cyto_vol[i] is the total number of voxels between dna_sq and mem_seq for section "i"
    gfp_intensities = np.zeros(num_sections) # gfp_intensities[i] is the total summed intensity of gfp channel for section"i"

    theta = np.pi / num_sections # The angle that each section captures, in radians

    count = 0

    for section in range(num_sections):
        upper_bound_angle = section * theta
        lower_bound_angle = (1 + section) * theta

        # Find cosine of angle in the case of unit circle
        cos_upper_bound_angle = np.cos(upper_bound_angle)
        cos_lower_bound_angle = np.cos(lower_bound_angle)

        mask = (res > cos_lower_bound_angle) * (res <= cos_upper_bound_angle) #(res > cos_lower_bound_angle * res <= cos_upper_bound_angle)
        mask = np.swapaxes(mask, 0, 2)

        masked_seg_mem = mask * seg_mem
        masked_seg_dna = mask * seg_dna

        cyto_vol[section] += np.count_nonzero(masked_seg_mem) - np.count_nonzero(masked_seg_dna) # Increment by number of voxels between cell membrane and nucleus
        gfp_intensities[section] += np.sum(gfp * mask)

        count += np.sum(mask)

    # Normalize
    cyto_vol = cyto_vol / np.sum(cyto_vol)
    cyto_vol = cyto_vol * vol_scale_factor # Scale volume by pixel scale factor
    gfp_intensities = gfp_intensities / np.sum(gfp_intensities)

    # Calculate the Fold Change for each Section
    fold_changes = np.zeros(num_sections)

    for section in range(num_sections):
        if (cyto_vol[section] != 0 and gfp_intensities[section] != 0): # Some sections have cytoplasm volume of 0
            fold_changes[section] = np.log2(gfp_intensities[section] / cyto_vol[section])
        elif (cyto_vol[section] == 0):
            print('Cytoplasm volume of 0 detected')
        elif (gfp_intensities[section] == 0):
            print('GFP intensity of 0 detected')

    return fold_changes, cyto_vol, gfp_intensities