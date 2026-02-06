#!/usr/bin/env python3
"""UR5 RTDE bridge

이 노드는 아래 topic interface들을 통해 다른 노드와 UR5를 연결해줍니다.
-------------------
Subscribes (commands):
  - /ur5/goal_tcp_pose     : absolute TCP pose (moveL)         [geometry_msgs/PoseStamped]
  - /ur5/goal_tcp_pose_r   : relative TCP delta (moveL)        [geometry_msgs/PoseStamped]
  - /ur5/goal_joint        : absolute joint target (moveJ)     [sensor_msgs/JointState]
  - /ur5/goal_joint_r      : relative joint delta (moveJ)      [sensor_msgs/JointState]
  - /ur5/cmd  : where/list/save/go                [std_msgs/String]
    - where : log current TCP pose + joint angles
    - list  : log saved pose names
    - save <name> : save current joint angles as <name>
    - go <name>   : moveJ(Joint-based move) to saved joint angles <name>

Publishes (state):
  - /ur5/tcp_pose     : current TCP pose                          [geometry_msgs/PoseStamped]
  - /ur5/status  : "IDLE" or "MOVING"                        [std_msgs/String]
    - status is protected by a lock AND we publish status immediately on transitions
     (so very short motions still produce at least one MOVING + one IDLE message so that
     other nodes can recognize this node has finished its work).

Services:
  - /ur5/stop : stop motion                                       [std_srvs/Trigger]

Pose Database format:
    {
      "home": {"type": "joint", "q": [q0,q1,q2,q3,q4,q5]},
      ...
    }
"""

import json
import os
import threading
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from math_utils import quat_to_mat, mat_to_quat, mat_to_rotvec, rotvec_to_mat

class UR5RTDEBridge(Node):
    def __init__(self):
        super().__init__("ur5_rtde_bridge")

        # ---- params ----
        self.robot_ip = self.declare_parameter("robot_ip", "192.168.0.43").value

        # moveL params (TCP)
        self.speed_l = float(self.declare_parameter("speed_l", 0.10).value)   # m/s
        self.accel_l = float(self.declare_parameter("accel_l", 0.25).value)   # m/s^2

        # moveJ params (joint). If not set, these defaults are reasonable.
        self.speed_j = float(self.declare_parameter("speed_j", 1.0).value)    # rad/s
        self.accel_j = float(self.declare_parameter("accel_j", 1.0).value)    # rad/s^2

        self.publish_rate = float(self.declare_parameter("publish_rate", 30.0).value)

        # ---- RTDE ----
        import rtde_control, rtde_receive
        self.rtde_c = rtde_control.RTDEControlInterface(self.robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.robot_ip)

        # ---- state (thread-safe) ----
        self._state_lock = threading.Lock()
        self._moving: bool = False
        self._status: str = "IDLE"  # only IDLE / MOVING

        # ---- pubs ----
        self.pub_tcp_pose = self.create_publisher(PoseStamped, "/ur5/tcp_pose", 10)
        self.pub_status = self.create_publisher(String, "/ur5/status", 10)

        # ---- subs: moveL ----
        # /ur5/goal_tcp_pose 이름으로 발행된 topic이 들어올 때마다 _on_tcp_abs 함수가 불립니다.
        # 이 topic에는 목표 위치가 담겨져 있으며, _on_tcp_abs는 이 목표 위치를 parameter로 받아 이동 명령을 실행합니다.
        # 또한 /ur5/goal_tcp_pose_r 토픽도 구독하여 상대 이동 명령을 처리합니다.
        self.sub_tcp_abs = self.create_subscription(PoseStamped, "/ur5/goal_tcp_pose", self._on_tcp_abs, 10)
        self.sub_tcp_rel = self.create_subscription(PoseStamped, "/ur5/goal_tcp_pose_r", self._on_tcp_rel, 10)

        # ---- subs: moveJ ----
        # /ur5/goal_joint 이름으로 발행된 topic이 들어올 때마다 _on_joint_abs 함수가 불립니다.
        # 이 topic에는 목표 관절 각도가 담겨져 있으며, _on_joint_abs는 이 목표 각도를 parameter로 받아 이동 명령을 실행합니다.
        # 또한 /ur5/goal_joint_r 토픽도 구독하여 상대 관절 이동 명령을 처리합니다.
        self.sub_joint_abs = self.create_subscription(JointState, "/ur5/goal_joint", self._on_joint_abs, 10)
        self.sub_joint_rel = self.create_subscription(JointState, "/ur5/goal_joint_r", self._on_joint_rel, 10)

        # ---- service ----
        # /stop 서비스가 호출되면 _on_stop 함수가 실행되어 로봇의 움직임을 멈춥니다.
        self.srv_stop = self.create_service(Trigger, "/ur5/stop", self._on_stop)

        # ---- command topic + pose DB ----
        # json 파일로 저장된 pose들을 불러옵니다.
        self.pose_db_path = Path(__file__).resolve().parent / "ur5_saved_poses.json"
        self.pose_db: Dict[str, Dict] = {}
        self._load_pose_db()

        # /ur5/cmd 토픽을 구독하여 _on_cmd 함수에서 명령어를 처리합니다.
        self.sub_cmd = self.create_subscription(String, "/ur5/cmd", self._on_cmd, 10)

        # ---- periodic publish ----
        # rate가 50이라면 1초에 50번 _publish_state 함수가 호출되어 현재 TCP 위치와 실행 상태를 발행합니다.
        if self.publish_rate > 0:
            self.create_timer(1.0 / self.publish_rate, self._publish_state)

        self.get_logger().info(
            f"UR5 RTDE bridge connected to {self.robot_ip}. "
            f"moveL: speed_l={self.speed_l}, accel_l={self.accel_l} | "
            f"moveJ: speed_j={self.speed_j}, accel_j={self.accel_j}"
        )
        self.get_logger().info("Subscribing: /ur5/goal_tcp_pose, /ur5/goal_tcp_pose_r, /ur5/goal_joint, /ur5/goal_joint_r")
        self.get_logger().info("Cmd topic: /ur5/cmd (where/list/save/go)")
        self.get_logger().info("Publishing: /ur5/tcp_pose, /ur5/status")

    # --------------------
    # status helpers
    # --------------------
    def _get_status(self) -> str:
        with self._state_lock:
            return self._status

    def _set_status(self, status: str):
        with self._state_lock:
            self._status = status
        self._publish_status(status)

    def _publish_status(self, status: Optional[str] = None):
        s = String()
        s.data = (status or self._get_status())
        try:
            self.pub_status.publish(s)
        except Exception:
            # If middleware is not ready, just skip.
            pass

    # --------------------
    # periodic state publish
    # --------------------
    def _publish_state(self):
        # publish current TCP pose + current status at a fixed rate
        try:
            pose = self.rtde_r.getActualTCPPose()  # [x,y,z, rx,ry,rz]
            x, y, z, rx, ry, rz = [float(v) for v in pose]
            R = rotvec_to_mat(np.array([rx, ry, rz], dtype=np.float64))
            qx, qy, qz, qw = mat_to_quat(R)

            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.position.x = x
            msg.pose.position.y = y
            msg.pose.position.z = z
            msg.pose.orientation.x = float(qx)
            msg.pose.orientation.y = float(qy)
            msg.pose.orientation.z = float(qz)
            msg.pose.orientation.w = float(qw)
            self.pub_tcp_pose.publish(msg)

            self._publish_status()
        except Exception:
            # Don't spam logs if robot is not ready / connection hiccups
            pass

    # --------------------
    # motion runner
    # --------------------
    def _start_motion(self, worker_fn: Callable[[], object], busy_msg: str) -> bool:
        with self._state_lock:
            if self._moving:
                self.get_logger().warn(busy_msg)
                return False
            self._moving = True

        # Make "acceptance" visible immediately.
        self._set_status("MOVING")

        def runner():
            try:
                worker_fn()
            except Exception as e:
                self.get_logger().error(f"Motion failed: {e}")
            finally:
                with self._state_lock:
                    self._moving = False
                self._set_status("IDLE")

        threading.Thread(target=runner, daemon=True).start()
        return True

    # --------------------
    # cmd: where/list/save/go
    # --------------------
    def _on_cmd(self, msg: String):
        line = (msg.data or "").strip()
        if not line:
            return

        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip().lower() if len(parts) == 2 else None

        if cmd == "where":
            tcp = self.rtde_r.getActualTCPPose()
            q = self.rtde_r.getActualQ()
            self.get_logger().info(f"CURRENT TCP: {tcp}")
            self.get_logger().info(f"CURRENT  Q : {q}")
            return

        if cmd == "list":
            keys = sorted(self.pose_db.keys())
            self.get_logger().info(f"SAVED: {keys}")
            return

        if cmd == "save":
            if not arg:
                self.get_logger().warn("Usage: save <name>")
                return
            q = self.rtde_r.getActualQ()
            entry = {"type": "joint", "q": [float(v) for v in q]}
            self.pose_db[arg] = entry
            self._save_pose_db()
            self.get_logger().info(f"SAVED '{arg}' (joint q)")
            return

        if cmd == "go":
            if not arg:
                self.get_logger().warn("Usage: go <name>")
                return
            self._go_saved_joint(arg)
            return

        self.get_logger().warn("Unknown command. supported: where, list, save <name>, go <name>")

    def _go_saved_joint(self, name: str):
        key = name.lower()
        if key not in self.pose_db:
            self.get_logger().warn(f"No saved pose: '{key}'. Use 'list' or 'save {key}'.")
            return

        entry = self.pose_db[key]
        if not isinstance(entry, dict) or entry.get("type") != "joint" or "q" not in entry:
            self.get_logger().warn(f"Saved entry '{key}' is not a joint pose. (file might be legacy?)")
            return

        q = entry["q"]
        if not (isinstance(q, list) and len(q) == 6):
            self.get_logger().warn(f"Saved joint pose '{key}' is invalid.")
            return

        self.get_logger().info(f"GO '{key}' (moveJ)")
        self._start_motion(lambda: self.rtde_c.moveJ(q, speed=self.speed_j, acceleration=self.accel_j),
                           busy_msg="Robot is moving. Ignore go command.")

    # --------------------
    # pose db
    # --------------------
    def _load_pose_db(self):
        try:
            if not self.pose_db_path.exists():
                self.get_logger().info(f"Pose DB not found: {self.pose_db_path} (starting empty)")
                return

            with open(self.pose_db_path, "r") as f:
                data = json.load(f) or {}

            if not isinstance(data, dict):
                self.get_logger().warn("Pose DB is not a dict; ignoring.")
                return

            loaded = {}

            for k, v in data.items():
                name = str(k).strip().lower()
                if not name:
                    continue

                loaded[name] = {"type": "joint", "q": [float(x) for x in v["q"]]}
                continue

            self.pose_db = loaded
            self.get_logger().info(f"Loaded pose DB: {self.pose_db_path} (keys={list(self.pose_db.keys())})")
        except Exception as e:
            self.get_logger().warn(f"Failed to load pose DB: {e}")

    def _save_pose_db(self):
        try:
            os.makedirs(self.pose_db_path.parent, exist_ok=True)
            with open(self.pose_db_path, "w") as f:
                json.dump(self.pose_db, f, indent=2)
        except Exception as e:
            self.get_logger().warn(f"Failed to save pose DB: {e}")

    # --------------------
    # stop service
    # --------------------
    def _on_stop(self, req, resp):
        # Best-effort stop. We do NOT force status to IDLE here if a motion thread is running.
        try:
            # stopL is used to stop linear motion (best-effort).
            self.rtde_c.stopL(0.5)
            # Some setups support stopJ; call if present.
            if hasattr(self.rtde_c, "stopJ"):
                try:
                    self.rtde_c.stopJ(0.5)
                except Exception:
                    pass

            resp.success = True
            resp.message = "stop called"
        except Exception as e:
            resp.success = False
            resp.message = str(e)

        # publish whatever the current status is
        self._publish_status()
        return resp

    # --------------------
    # moveL callbacks
    # --------------------
    @staticmethod
    def _pose_to_rtde_target(msg: PoseStamped):
        # (x, y, z, quaternion) 형식으로 들어온 pose를 ur5가 쓰는 (x, y, z, rx, ry, rz) 형식으로 변환합니다.
        p = msg.pose.position
        q = msg.pose.orientation
        R = quat_to_mat([q.x, q.y, q.z, q.w])
        rv = mat_to_rotvec(R)
        return [float(p.x), float(p.y), float(p.z), float(rv[0]), float(rv[1]), float(rv[2])]

    @staticmethod
    def _unwrap_rotvec_near(rotvec: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Return an equivalent rotation-vector close to `reference`.

        Rotation-vector (axis-angle) representation is *not unique*:
        the same rotation can be represented as `v + 2πk * axis` (k ∈ ℤ).

        UR's moveL interpolates linearly in the 6D pose vector space.
        If rotvec suddenly "wraps" (e.g., near ±π), the robot may spin in an
        unexpected direction even though the *true* rotation is almost the same.
        This helper picks a representation that stays close to the current rotvec.
        """

        v = np.asarray(rotvec, dtype=np.float64).reshape(3)
        ref = np.asarray(reference, dtype=np.float64).reshape(3)

        theta = float(np.linalg.norm(v))
        if theta < 1e-12:
            return v

        axis = v / theta

        # Choose integer k that minimizes || (v + 2πk axis) - ref ||.
        # Derivation: only the component along `axis` changes with k.
        k0 = int(np.round((float(np.dot(axis, ref)) - theta) / (2.0 * np.pi)))

        best = v
        best_norm = float(np.linalg.norm(v - ref))
        for k in (k0 - 1, k0, k0 + 1):
            cand = v + (2.0 * np.pi * float(k)) * axis
            n = float(np.linalg.norm(cand - ref))
            if n < best_norm:
                best = cand
                best_norm = n

        return best

    def _on_tcp_abs(self, msg: PoseStamped):
        target = self._pose_to_rtde_target(msg)
        self._start_motion(lambda: self.rtde_c.moveL(target, speed=self.speed_l, acceleration=self.accel_l),
                           busy_msg="Robot is moving. Ignore /ur5/goal_tcp_pose.")

    def _on_tcp_rel(self, msg: PoseStamped):
        """Relative TCP delta (moveL).

        **중요**: UR의 TCP 자세는 (x,y,z, rx,ry,rz)에서 회전(rx,ry,rz)이 *rotvec(axis-angle)*입니다.
        rotvec은 벡터처럼 더하면(+) 올바른 회전 합성이 되지 않습니다.

        우리가 원하는 동작:
          - TCP 위치는 고정(또는 dx,dy,dz만큼 평행이동)
          - 회전은 "base 좌표계"의 x/y/z 축을 기준으로 상대 회전

        구현:
          - 현재 rotvec -> 회전행렬 R_cur
          - delta quaternion -> 회전행렬 R_delta
          - base 기준 회전이므로 **좌측 곱**: R_target = R_delta @ R_cur
            (참고: tool 기준 회전이면 R_target = R_cur @ R_delta)
          - R_target -> rotvec로 변환 후 moveL 타겟에 적용
        """

        # ---- delta translation (base frame) ----
        p = msg.pose.position
        delta_xyz = np.array([p.x, p.y, p.z], dtype=np.float64)

        # ---- delta rotation (quaternion) ----
        q = msg.pose.orientation
        q_delta = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
        qn = float(np.linalg.norm(q_delta))
        if qn < 1e-12:
            # invalid quaternion -> treat as identity
            q_delta = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        else:
            q_delta = q_delta / qn

        R_delta = quat_to_mat([float(q_delta[0]), float(q_delta[1]), float(q_delta[2]), float(q_delta[3])])

        def worker():
            pose = self.rtde_r.getActualTCPPose()  # [x,y,z, rx,ry,rz]
            x, y, z, rx, ry, rz = [float(v) for v in pose]

            # translation delta in base
            x_t = x + float(delta_xyz[0])
            y_t = y + float(delta_xyz[1])
            z_t = z + float(delta_xyz[2])

            # current orientation
            rv_cur = np.array([rx, ry, rz], dtype=np.float64)
            R_cur = rotvec_to_mat(rv_cur)

            # base-frame relative rotation
            R_target = R_delta @ R_cur
            rv_target = mat_to_rotvec(R_target)

            # keep representation continuous near current rotvec (avoid sudden 2π wrap)
            rv_target = self._unwrap_rotvec_near(rv_target, rv_cur)

            target = [
                float(x_t), float(y_t), float(z_t),
                float(rv_target[0]), float(rv_target[1]), float(rv_target[2])
            ]
            return self.rtde_c.moveL(target, speed=self.speed_l, acceleration=self.accel_l)

        self._start_motion(worker, busy_msg="Robot is moving. Ignore /ur5/goal_tcp_pose_r.")

    # --------------------
    # moveJ callbacks
    # --------------------
    @staticmethod
    def _joint_from_msg(msg: JointState) -> Optional[Sequence[float]]:
        if not msg.position:
            return None
        if len(msg.position) < 6:
            return None
        return [float(x) for x in msg.position[:6]]

    def _on_joint_abs(self, msg: JointState):
        q = self._joint_from_msg(msg)
        if q is None:
            self.get_logger().warn("/ur5/goal_joint requires JointState.position with 6 values.")
            return
        self._start_motion(lambda: self.rtde_c.moveJ(q, speed=self.speed_j, acceleration=self.accel_j),
                           busy_msg="Robot is moving. Ignore /ur5/goal_joint.")

    def _on_joint_rel(self, msg: JointState):
        dq = self._joint_from_msg(msg)
        if dq is None:
            self.get_logger().warn("/ur5/goal_joint_r requires JointState.position with 6 values.")
            return

        def worker():
            q = self.rtde_r.getActualQ()
            q2 = [float(q[i]) + float(dq[i]) for i in range(6)]
            return self.rtde_c.moveJ(q2, speed=self.speed_j, acceleration=self.accel_j)

        self._start_motion(worker, busy_msg="Robot is moving. Ignore /ur5/goal_joint_r.")


def main():
    rclpy.init()
    node = UR5RTDEBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

