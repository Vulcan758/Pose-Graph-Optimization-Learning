import numpy as np
from scipy.spatial.transform import Rotation as R
rng = np.random.default_rng(42)

def T(t, q):  # q = [qx,qy,qz,qw]
    M = np.eye(4); M[:3,:3] = R.from_quat(q).as_matrix(); M[:3,3] = t; return M
def tq(M):
    return M[:3,3], R.from_matrix(M[:3,:3]).as_quat()

# Ground truth: 3 laps of a 10x6 m rectangle, climbing 0.5 m per lap
gt = []
side_pts = [(10,0),(0,6),(-10,0),(0,-6)]
pose = np.eye(4); step = 1.0
for lap in range(3):
    for dx, dy in side_pts:
        n = int(round(max(abs(dx),abs(dy))/step))
        for _ in range(n):
            gt.append(pose.copy())
            mv = np.eye(4); mv[0,3] = step; mv[2,3] = 0.5/32
            pose = pose @ mv
        rot = np.eye(4); rot[:3,:3] = R.from_euler('z', 90, degrees=True).as_matrix()
        pose = pose @ rot
gt.append(pose.copy())
N = len(gt)

sig_t, sig_r = 0.05, np.deg2rad(1.0)
info = np.diag([1/sig_t**2]*3 + [1/sig_r**2]*3)
info_str = " ".join(f"{info[i,j]:.4f}" for i in range(6) for j in range(i,6))

def noisy(M):
    t, q = tq(M)
    t = t + rng.normal(0, sig_t, 3)
    r = R.from_rotvec(rng.normal(0, sig_r, 3)) * R.from_quat(q)
    return T(t, r.as_quat())

edges = []
for i in range(N-1):
    edges.append((i, i+1, noisy(np.linalg.inv(gt[i]) @ gt[i+1])))
# loop closures: same spot on consecutive laps (32 poses per lap)
for i in range(N):
    j = i + 32
    if j < N and i % 4 == 0:
        edges.append((i, j, noisy(np.linalg.inv(gt[i]) @ gt[j])))

# Initial guess: dead reckoning from noisy odometry
init = [np.eye(4)]
for i, j, Z in edges[:N-1]:
    init.append(init[-1] @ Z)

def write(fn, poses, with_edges):
    with open(fn, "w") as f:
        for k, P in enumerate(poses):
            t, q = tq(P)
            f.write(f"VERTEX_SE3:QUAT {k} " + " ".join(f"{v:.6f}" for v in [*t, *q]) + "\n")
        if with_edges:
            for i, j, Z in edges:
                t, q = tq(Z)
                f.write(f"EDGE_SE3:QUAT {i} {j} " + " ".join(f"{v:.6f}" for v in [*t, *q]) + f" {info_str}\n")

write("dummy_se3.g2o", init, True)
write("dummy_se3_groundtruth.g2o", gt, False)
print(N, "vertices,", len(edges), "edges (", len(edges)-(N-1), "loop closures)")