"""Reproduce Lego camera measurements: python3 measure.py > manifest-copy.json (from any directory)."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parents[3]
scene = root / "data/raw/nerf_synthetic/lego"
flip = np.diag([1, -1, -1, 1])
sift = cv2.SIFT_create(nfeatures=1600)
matcher = cv2.BFMatcher(cv2.NORM_L2)


def features(path):
    with Image.open(path) as image:
        rgba = np.asarray(image)
    gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
    return sift.detectAndCompute(gray, np.uint8(rgba[:, :, 3] > 240) * 255)


def match_pair(paths, views, K, i, j):
    key1, desc1 = features(paths[i])
    key2, desc2 = features(paths[j])
    candidates = matcher.knnMatch(desc1, desc2, k=2)
    matches = [a for a, b in candidates if a.distance < 0.70 * b.distance]
    x1 = np.array([key1[m.queryIdx].pt for m in matches])
    x2 = np.array([key2[m.trainIdx].pt for m in matches])
    relative = views[j] @ np.linalg.inv(views[i])
    tx, ty, tz = relative[:3, 3]
    skew = np.array([[0, -tz, ty], [tz, 0, -tx], [-ty, tx, 0]])
    inv = np.linalg.inv(K)
    F = inv.T @ skew @ relative[:3, :3] @ inv
    h1 = np.column_stack([x1, np.ones(len(x1))])
    h2 = np.column_stack([x2, np.ones(len(x2))])
    lines2, lines1 = h1 @ F.T, h2 @ F
    error = abs((h2 * lines2).sum(axis=1)) / np.sqrt(
        (lines2[:, :2]**2 + lines1[:, :2]**2).sum(axis=1))
    close = error <= 1
    X = cv2.triangulatePoints(K @ views[i, :3], K @ views[j, :3], x1[close].T, x2[close].T)
    X /= X[3]
    positive = ((views[i] @ X)[2] > 0) & ((views[j] @ X)[2] > 0)
    return dict(first=paths[i].name, second=paths[j].name, matches=len(matches),
                within_1px=int(close.sum()), positive_depth=int(positive.sum()),
                median_sampson_root_px=float(np.median(error)))


report = dict(run_id="inspection_001", scene=str(scene.relative_to(root)),
              numpy=np.__version__, opencv=cv2.__version__, splits={})
all_poses = []
all_paths = []
for split, expected in (("train", 100), ("val", 100), ("test", 200)):
    metadata = json.loads((scene / f"transforms_{split}.json").read_text())
    paths = [scene / (item["file_path"] + ".png") for item in metadata["frames"]]
    assert len(paths) == len(set(paths)) == expected
    assert set(paths) == {p for p in (scene / split).glob("*.png") if p.stem.removeprefix("r_").isdigit()}
    sizes = set()
    for path in paths:
        with Image.open(path) as image:
            image.load()
            sizes.add((image.size, image.mode, image.format))
    assert sizes == {((800, 800), "RGBA", "PNG")}
    (w, h), mode, _ = sizes.pop()
    fov = metadata["camera_angle_x"]
    f = w / (2 * np.tan(fov / 2))
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    pose = np.array([item["transform_matrix"] for item in metadata["frames"]])
    all_poses.extend(pose)
    all_paths.extend(p.relative_to(scene).as_posix() for p in paths)
    assert pose.shape == (expected, 4, 4) and np.isfinite(pose).all()
    assert np.allclose(pose[:, 3], [0, 0, 0, 1], atol=1e-6, rtol=0)
    R, C = pose[:, :3, :3], pose[:, :3, 3]
    ortho = float(abs(R.transpose(0, 2, 1) @ R - np.eye(3)).max())
    det_error = float(abs(np.linalg.det(R) - 1).max())
    assert max(ortho, det_error) < 1e-5
    views = np.linalg.inv(pose @ flip)
    origin = views[:, :3, 3]
    assert np.all(origin[:, 2] > 0)
    origin_offset = np.linalg.norm(f * origin[:, :2] / origin[:, 2:3], axis=1)
    radius = np.linalg.norm(C, axis=1)
    direction = C / radius[:, None]
    pairs = []
    for i in (0, expected // 3, 2 * expected // 3):
        angles = np.degrees(np.arccos(np.clip(direction @ direction[i], -1, 1)))
        j = int(abs(angles - 8).argmin())
        pairs.append(match_pair(paths, views, K, i, j))
    report["splits"][split] = dict(frames=len(paths), size=[w, h], mode=mode, fov_x_rad=fov,
        fov_x_deg=float(np.degrees(fov)), K=K.tolist(), max_orthogonality_error=ortho,
        max_determinant_error=det_error, min_radius=float(radius.min()), max_radius=float(radius.max()),
        max_origin_offset_px=float(origin_offset.max()), example=paths[0].name,
        example_center=C[0].tolist(), example_world_forward=(-R[0, :, 2]).tolist(),
        example_w2c=views[0].tolist(), image_pair_checks=pairs)
poses = np.stack(all_poses)
report["near_pose_pairs_at_max_abs_difference_1e-5"] = []
for i, pose in enumerate(poses):
    differences = np.max(abs(poses[:i] - pose), axis=(1, 2))
    for j in np.flatnonzero(differences <= 1e-5):
        report["near_pose_pairs_at_max_abs_difference_1e-5"].append(
            [all_paths[j], all_paths[i], float(differences[j])])
print(json.dumps(report, indent=2))

