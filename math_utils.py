import numpy as np
from geometry_msgs.msg import PoseStamped, TransformStamped

def quat_to_mat(q):
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1 - 2*(yy+zz),     2*(xy-wz),       2*(xz+wy)],
        [2*(xy+wz),         1 - 2*(xx+zz),   2*(yz-wx)],
        [2*(xz-wy),         2*(yz+wx),       1 - 2*(xx+yy)]
    ], dtype=np.float64)
    return R

def mat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2,1] - R[1,2]) / s
        qy = (R[0,2] - R[2,0]) / s
        qz = (R[1,0] - R[0,1]) / s
    else:
        i = int(np.argmax([R[0,0], R[1,1], R[2,2]]))
        if i == 0:
            s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            qw = (R[2,1] - R[1,2]) / s
            qx = 0.25 * s
            qy = (R[0,1] + R[1,0]) / s
            qz = (R[0,2] + R[2,0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            qw = (R[0,2] - R[2,0]) / s
            qx = (R[0,1] + R[1,0]) / s
            qy = 0.25 * s
            qz = (R[1,2] + R[2,1]) / s
        else:
            s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            qw = (R[1,0] - R[0,1]) / s
            qx = (R[0,2] + R[2,0]) / s
            qy = (R[1,2] + R[2,1]) / s
            qz = 0.25 * s

    q = np.array([qx,qy,qz,qw], dtype=np.float64)
    q /= (np.linalg.norm(q) + 1e-12)
    return q

def mat_to_rpy_zyx(R):
    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    # pitch = asin(-R[2,0])
    if abs(R[2,0]) < 1.0:
        pitch = np.arcsin(-R[2,0])
        roll  = np.arctan2(R[2,1], R[2,2])
        yaw   = np.arctan2(R[1,0], R[0,0])
    else:
        # Gimbal lock: pitch = +/-90 deg
        pitch = np.pi/2 if R[2,0] <= -1.0 else -np.pi/2
        roll = 0.0
        yaw = np.arctan2(-R[0,1], R[1,1])
    return roll, pitch, yaw

def mat_to_rotvec(R):
    # axis-angle: rotvec = axis * angle
    tr = np.trace(R)
    cosang = (tr - 1.0) / 2.0
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.arccos(cosang)
    if ang < 1e-9:
        return np.zeros(3, dtype=np.float64)
    axis = np.array([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0],
        R[1,0] - R[0,1],
    ], dtype=np.float64) / (2.0*np.sin(ang))
    return axis * ang

def rotvec_to_mat(rv):
    ang = np.linalg.norm(rv)
    if ang < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rv / ang
    x,y,z = axis
    K = np.array([[0,-z,y],[z,0,-x],[-y,x,0]], dtype=np.float64)
    R = np.eye(3) + np.sin(ang)*K + (1-np.cos(ang))*(K@K)
    return R


def tf_to_T(tf: TransformStamped) -> np.ndarray:
    t = tf.transform.translation
    r = tf.transform.rotation
    R = quat_to_mat([r.x, r.y, r.z, r.w])
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3,3] = [t.x, t.y, t.z]
    return T

def pose_to_T(p: PoseStamped) -> np.ndarray:
    q = p.pose.orientation
    t = p.pose.position
    R = quat_to_mat([q.x,q.y,q.z,q.w])
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3,3] = [t.x, t.y, t.z]
    return T


