import numpy as np
from scipy import sparse
import scipy 
import open3d as o3d
from scipy.spatial.distance import cdist
from point_cloud_utils import remove_isolated_points, get_subpcd, \
        get_statistical_inlier_indices, kDTree_1NN_feature_reprojection,generate_random_colors

def ncuts_chunk(pcd_chunk,pcd_ground_chunk,chunk_major,ncuts_threshold=0.03,split_lim=0.075,alpha=1.0,proximity_threshold=1.0):
    points_major = np.asarray(chunk_major.points)
    num_points_major = points_major.shape[0]   

    spatial_distance = cdist(points_major, points_major)
    mask = np.where(spatial_distance <= proximity_threshold, 1, 0)

    if alpha:
                    spatial_edge_weights = mask * np.exp(-alpha * spatial_distance)
    else: 
                    spatial_edge_weights = mask

    A = spatial_edge_weights
    print("Adjacency Matrix built")

    chunk_major, A = remove_isolated_points(chunk_major, A)
    print(num_points_major - np.asarray(chunk_major.points).shape[0], "isolated points removed")
    num_points_major = np.asarray(chunk_major.points).shape[0]

    A = scipy.sparse.csr_matrix(A)
    grouped_labels = normalized_cut(A,num_points_major, np.arange(num_points_major), T = ncuts_threshold,split_lim=split_lim)


    num_groups = len(grouped_labels)

    print("There are", num_groups, "cut regions")

    random_colors = generate_random_colors(600)

    pcd_color = np.zeros((num_points_major, 3))
    for i, s in enumerate(grouped_labels):
                    for j in s:
                                    pcd_color[j] = np.array(random_colors[i]) / 255

    pcd_chunk.paint_uniform_color([0, 0, 0])
    colors = kDTree_1NN_feature_reprojection(np.asarray(pcd_chunk.colors), pcd_chunk, pcd_color, chunk_major)
    pcd_chunk.colors = o3d.utility.Vector3dVector(colors)

    ##only remove some obstacle points which are falsely associated to the ground
    inliers = get_statistical_inlier_indices(pcd_ground_chunk)
    ground_inliers = get_subpcd(pcd_ground_chunk, inliers)
    mean_hight = np.mean(np.asarray(ground_inliers.points)[:,2])
    in_idcs = np.where(np.asarray(ground_inliers.points)[:,2] < (mean_hight + 0.2))[0]
    cut_hight = get_subpcd(ground_inliers, in_idcs)
    cut_hight.paint_uniform_color([0, 0, 0])
    merged_chunk = pcd_chunk + cut_hight

    return merged_chunk

def cut_cost(W, mask):
    return (np.sum(W) - np.sum(W[mask][:, mask]) - np.sum(W[~mask][:, ~mask])) / 2

def ncut_cost(W, D, cut):
    cost = cut_cost(W, cut)
    assoc_a = D.todense()[cut].sum() # Anastasiia: this also can be optimized in the future
    assoc_b = D.todense()[~cut].sum()
    return (cost / assoc_a) + (cost / assoc_b)

def get_min_ncut(ev, d, w, num_cuts):
    mcut = np.inf
    mn = ev.min()
    mx = ev.max()

    # If all values in `ev` are equal, it implies that the graph can't be
    # further sub-divided. In this case the bi-partition is the the graph
    # itself and an empty set.
    min_mask = np.zeros_like(ev, dtype=bool)
    if np.allclose(mn, mx):
        return min_mask, mcut

    # Refer Shi & Malik 2001, Section 3.1.3, Page 892
    # Perform evenly spaced n-cuts and determine the optimal one.
    for t in np.linspace(mn, mx, num_cuts, endpoint=False):
        mask = ev > t
        cost = ncut_cost(w,d, mask)
        if cost < mcut:
            min_mask = mask
            mcut = cost

    return min_mask, mcut


def normalized_cut(w, num_points_orig,labels, T = 0.01,split_lim=0.01):
    W = w + sparse.identity(w.shape[0])
    split_percentage = labels.shape[0] / (num_points_orig + 1e-8) 
    if W.shape[0] > 2 and split_percentage > split_lim:

        d = np.array(W.sum(axis=0))[0]
        d2 = np.reciprocal(np.sqrt(d))
        D = sparse.diags(d)
        D2 = sparse.diags(d2)

        A = D2 * (D - W) * D2

        eigvals, eigvecs = sparse.linalg.eigsh(A, 2, sigma = 1e-10, which='LM')

        index2 = np.argsort(eigvals)[1]

        ev = eigvecs[:, index2]
        mask, mcut = get_min_ncut(ev, D, w, 10)
    
        if mcut < T :
                labels1 = normalized_cut(w[mask][:, mask], num_points_orig,labels[mask], T=T)
                labels2 = normalized_cut(w[~mask][:, ~mask], num_points_orig,labels[~mask], T=T)
                return labels1 + labels2
        else:
            return [labels]
    else:
        return [labels]