import numpy as np
import os

from read_write_model import read_model

# Prepare .npz file with scene information (image_paths, depth_paths, intrinsics, poses, pairs, overlaps)
def prepare_scene_info(folder, resultName):
    directory = os.path.join("data", folder)
    cameras, images, points3d = read_model(directory, ext = ".bin")

    idx = 0
    idx_to_id = {}
    ids = sorted(images.keys())
    image_paths = []
    depth_paths = []
    intrinsics = []
    poses = []

    for id in ids:
        idx_to_id[idx] = id
        fileName = images[id].name
        image_paths.append(os.path.join(folder, "images", fileName))
        depth_paths.append(os.path.join(folder, "depth_maps", fileName.split('.')[0] + ".h5"))
        
        fx, fy, cx, cy = cameras[images[id].camera_id].params
        K = np.zeros([3, 3])
        K[0, 0] = fx
        K[0, 2] = cx
        K[1, 1] = fy
        K[1, 2] = cy
        K[2, 2] = 1
        intrinsics.append(K)

        pose = np.zeros([4, 4])
        pose[:3, :3] = images[id].qvec2rotmat()
        pose[:3, 3] = images[id].tvec
        pose[3, 3] = 1
        poses.append(pose)

        idx += 1

    pairs = []
    overlaps = []
    for i in range(len(image_paths)):
        for j in range(i + 1, len(image_paths)):
            id1 = idx_to_id[i]
            id2 = idx_to_id[j]

            keypoints1 = set(images[id1].point3D_ids)
            keypoints2 = set(images[id2].point3D_ids)
            matches = keypoints1 & keypoints2

            pairs.append([i, j])
            overlaps.append(len(matches) / len(keypoints1))

            pairs.append([j, i])
            overlaps.append(len(matches) / len(keypoints2))

    np.savez(
        resultName,
        image_paths=image_paths,
        depth_paths=depth_paths,
        intrinsics=intrinsics,
        poses=poses,
        pairs=pairs,
        overlaps=overlaps
    )

prepare_scene_info("dawn_ceres/2015293_c6_orbit125", "dawn_ceres.npz")