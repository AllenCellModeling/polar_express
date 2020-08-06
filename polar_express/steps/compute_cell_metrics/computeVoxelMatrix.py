import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import distance


def compute_voxel_matrix(scale_factors, centroid, masked_channels):

    # Unpack masked segmentation channels
    seg_gfp = masked_channels["seg_gfp"]
    seg_dna = masked_channels["seg_dna"]
    seg_mem = masked_channels["seg_mem"]
    gfp = masked_channels["gfp"]

    cyto_indices = np.where(seg_gfp != 0)
    cyto_indices = np.array(cyto_indices).T
    num_voxels = cyto_indices.shape[0]
    cyto_tuples = [tuple(cyto_indices[i, :]) for i in range(num_voxels)]

    # column 1 of the matrix
    cyto_intensities = [gfp[i] for i in cyto_tuples]

    cyto_indices = cyto_indices * scale_factors  # apply pixel scale
    centroid = centroid * scale_factors  # apply pixel scale
    unit_z_vector = np.array([1, 0, 0])

    # column 2 of the matrix
    angles = [np.arccos(np.dot(unit_z_vector, cyto_indices[i, :] - centroid)
                        / (np.dot(cyto_indices[i, :] - centroid,
                                  cyto_indices[i, :] - centroid)
                        ** 0.5)) for i in range(num_voxels)]

    seg_dna_shell = ndi.distance_transform_cdt(seg_dna == 0, 'taxicab') == 1
    nucleus_shell_indices = np.where(seg_dna_shell != 0)
    nucleus_shell_indices = np.array(nucleus_shell_indices).T
    nucleus_shell_indices = nucleus_shell_indices * scale_factors  # apply pixel scale

    # column 3 of the matrix
    nucleus_dists = [np.min(distance.cdist(np.reshape(cyto_indices[i, :],
                     (1, 3)), nucleus_shell_indices, 'euclidean'))
                     for i in range(num_voxels)]

    seg_mem_shell = ndi.distance_transform_cdt(seg_mem == 0, 'taxicab') == 1
    mem_shell_indices = np.where(seg_mem_shell != 0)
    mem_shell_indices = np.array(mem_shell_indices).T
    mem_shell_indices = mem_shell_indices * scale_factors  # apply pixel scale

    # column 4 of the matrix
    mem_dists = [np.min(distance.cdist(np.reshape(cyto_indices[i, :],
                 (1, 3)), mem_shell_indices, 'euclidean')) for i in range(num_voxels)]

    return np.array([cyto_intensities, angles, nucleus_dists, mem_dists]).T
