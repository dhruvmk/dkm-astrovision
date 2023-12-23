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
    intrinsics = np.array([None for _ in range(len(ids))], dtype = object)
    poses = np.array([None for _ in range(len(ids))], dtype = object)

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
        intrinsics[idx] = K

        pose = np.zeros([4, 4])
        qvec = images[id].qvec
        qvec = qvec / np.linalg.norm(qvec)
        w, x, y, z = qvec
        R = np.array([
            [
                1 - 2 * y * y - 2 * z * z,
                2 * x * y - 2 * z * w,
                2 * x * z + 2 * y * w
            ],
            [
                2 * x * y + 2 * z * w,
                1 - 2 * x * x - 2 * z * z,
                2 * y * z - 2 * x * w
            ],
            [
                2 * x * z - 2 * y * w,
                2 * y * z + 2 * x * w,
                1 - 2 * x * x - 2 * y * y
            ]
        ])
        pose[: 3, : 3] = R
        pose[: 3, 3] = images[id].tvec
        pose[3, 3] = 1
        poses[idx] = pose

        idx += 1

    pairs = []
    overlaps = []
    for i in range(len(image_paths)):
        for j in range(i + 1, len(image_paths)):
            id1 = idx_to_id[i]
            id2 = idx_to_id[j]

            points1 = set(images[id1].point3D_ids)
            points2 = set(images[id2].point3D_ids)
            matches = points1 & points2

            if len(matches) == 0:
                continue

            pairs.append([i, j])
            overlaps.append(len(matches) / len(points1))

            pairs.append([j, i])
            overlaps.append(len(matches) / len(points2))

    resultDict = {
        "image_paths": np.array(image_paths, dtype = object),
        "depth_paths": np.array(depth_paths, dtype = object),
        "intrinsics": intrinsics,
        "poses": poses,
        "pairs": np.array(pairs, dtype = object),
        "overlaps": np.array(overlaps, dtype = object) 
    }
    print(resultDict)
    np.save(
        resultName,
        resultDict
    )
