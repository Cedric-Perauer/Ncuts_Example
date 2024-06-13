# Transformation on point clouds
import numpy as np
import open3d as o3d
import copy
from tqdm import tqdm 
import random


def generate_random_colors(N, seed=0):
    colors = set()  # Use a set to store unique colors
    while len(colors) < N:  # Keep generating colors until we have N unique ones
        # Generate a random color and add it to the set
        colors.add((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    return list(colors)  # Convert the set to a list before returning


def get_pcd(points: np.array):
    """
    Convert numpy array to open3d point cloud
    Args:
        points: 3D points in camera coordinate [npoints, 3] or [npoints, 4]
    Returns:
        pcd: open3d point cloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    return pcd

def transform_pcd(points, T):
    """
    Apply the perspective projection
    Args:
        points:              3D points in coordinate system 1 [npoints, 3]
        T:                   Transformation matrix in homogeneous coordinates [4, 4]
    Returns:
        points:              3D points in coordinate system 2 [npoints, 3]
    """
    pcd = get_pcd(points)
    transformed_pcd = pcd.transform(T)
    return np.asarray(transformed_pcd.points)

def filter_points_from_dict(points, filter_dict):
    """
    Filter points based on a dict
    Args:
        points:      3D points in camera coordinate [npoints, 3]
        filter_dict: dict that maps point indices to pixel coordinates
    Returns:
        points:    3D points in camera coordinate within image FOV [npoints, 3]
    """

    inds = np.array(list(filter_dict.keys()))
    return points[inds]

def filter_points_from_list(points, filter_list):
    """
    Filter points based on a dict
    Args:
        points:      3D points in camera coordinate [npoints, 3]
        filter_dict: dict that maps point indices to pixel coordinates
    Returns:
        points:    3D points in camera coordinate within image FOV [npoints, 3]
    """

    inds = np.array(list(filter_list))
    return points[inds]

def point_to_label(point_to_pixel: dict, label_map, label_is_color=True, label_is_instance=False):
    '''
    Args:
        point_to_pixel: dict that maps point indices to pixel coordinates
        label_map: label map of image
    Returns:
        point_to_label: dict that maps point indices to labels
    '''
    point_to_label_dict = {}

    for index, point_data in point_to_pixel.items():
        pixel = point_data['pixels']

        if label_is_color:
            color = label_map[pixel[1], pixel[0]]
            if color.tolist() == [70,70,70]: # ignore unlabeled pixels
                continue
            point_to_label_dict[index] = tuple((color / 255))

        elif label_is_instance:
            instance_label = int(label_map[pixel[1], pixel[0]])
            if instance_label: # ignore unlabeled pixels
                point_to_label_dict[index] = instance_label
            else:
                continue
            
        else: # label is continuous feature vector
            point_to_label_dict[index] = label_map[pixel[1], pixel[0]]
    
    return point_to_label_dict

def change_point_indices(point_to_X: dict, indices: list):
    '''
    Args:
        point_to_X: dict that maps point indices to X
        indices: list of indices
    Returns:
        point_to_X: dict that maps point indices to X
    '''
    point_to_X_new = {}
    for old_index in point_to_X.keys():
        new_index = indices[old_index]
        point_to_X_new[new_index] = point_to_X[old_index]
    
    return point_to_X_new

def transformation_matrix(rotation: np.array, translation: np.array):
    """
    Create a transformation matrix from rotation and translation
    Args:
        rotation:    Rotation matrix [3, 3]
        translation: Translation vector [3, 1]
    Returns:
        T:           Transformation matrix in homogeneous coordinates [4, 4]
    """

    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T

def kDTree_1NN_feature_reprojection(features_to, pcd_to, features_from, pcd_from, max_radius=None, no_feature_label=[1,0,0]):
    '''
    Args:
        pcd_from: point cloud to be projected
        pcd_to: point cloud to be projected to
        search_method: search method ("radius", "knn")
        search_param: search parameter (radius or k)
    Returns:
        features_to: features projected on pcd_to
    '''
    from_tree = o3d.geometry.KDTreeFlann(pcd_from)

    for i, point in enumerate(np.asarray(pcd_to.points)):

        [_, idx, _] = from_tree.search_knn_vector_3d(point, 1)
        if max_radius is not None:
            if np.linalg.norm(point - np.asarray(pcd_from.points)[idx[0]]) > max_radius:
                features_to[i,:] = no_feature_label
            else:
                features_to[i,:] = features_from[idx[0]]
        else:
            features_to[i,:] = features_from[idx[0]]
    
    return features_to

def intersect(pred_indices, gt_indices):
        intersection = np.intersect1d(pred_indices, gt_indices)
        return intersection.size / pred_indices.shape[0]

def remove_isolated_points(pcd, adjacency_matrix):

    isolated_mask = ~np.all(adjacency_matrix == 0, axis=1)
    adjacency_matrix = adjacency_matrix[isolated_mask][:, isolated_mask]
    pcd = pcd.select_by_index(np.where(isolated_mask == True)[0])

    return pcd, adjacency_matrix


def get_statistical_inlier_indices(pcd, nb_neighbors=20, std_ratio=2.0):
    _, inlier_indices = copy.deepcopy(pcd).remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return inlier_indices

def get_subpcd(pcd, indices, colors=False, normals=False):
    subpcd = o3d.geometry.PointCloud()
    subpcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[indices])
    if colors:
        subpcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[indices])
    if normals:
        subpcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[indices])
    return subpcd
