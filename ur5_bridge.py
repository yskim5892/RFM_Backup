#!/usr/bin/env python3
"""
UR5 Bridge Node (Simplified & Parameterized)

[Namespace 규칙]
- 이 노드는 relative 이름("~/...")을 사용합니다.
- node_name 이 "ur5_bridge"이면 실제 이름은 "/ur5_bridge/..."가 됩니다.

[Interfaces: Publish Topics]
1) ~/tcp_pose  (geometry_msgs/PoseStamped)
- 현재 TCP pose publish
- msg.pose.position: x, y, z (m)
- msg.pose.orientation: quaternion x, y, z, w

2) ~/status  (std_msgs/String)
- "IDLE" / "MOVING" 상태 publish

[Interfaces: Actions]
1) ~/move_tcp  (rfm/action/MoveTcp)
- Goal:
  target_pose: geometry_msgs/PoseStamped
  relative: bool
- Result:
  success: bool
  message: string
- Feedback:
  current_pose: geometry_msgs/PoseStamped
  status: string

2) ~/move_joint  (rfm/action/MoveJoint)
- Goal:
  target_joint: sensor_msgs/JointState (position[0:6] 사용)
  relative: bool
- Result:
  success: bool
  message: string
- Feedback:
  current_joint: sensor_msgs/JointState
  status: string

3) ~/move_saved  (rfm/action/MoveSaved)
- Goal:
  pose_name: string
- Result:
  success: bool
  message: string
- Feedback:
  status: string

[Interfaces: Services]
1) ~/manage_pose  (rfm/srv/ManagePose)
- Request:
  cmd: string  ("save" | "list" | "delete")
  name: string
- Response:
  success: bool
  message: string
  pose_names: string[]  (list에서 사용)

2) ~/stop  (std_srvs/srv/Trigger)
- Request: (empty)
- Response:
  success: bool
  message: string

[CLI Examples (node_name=ur5_bridge)]
- 상태 확인:
  ros2 topic echo /ur5_bridge/status
  ros2 topic echo /ur5_bridge/tcp_pose

- move_tcp (absolute):
  ros2 action send_goal /ur5_bridge/move_tcp rfm/action/MoveTcp \
  "{target_pose: {pose: {position: {x: 0.3, y: -0.2, z: 0.2}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}, relative: false}"

- move_tcp (relative):
  ros2 action send_goal /ur5_bridge/move_tcp rfm/action/MoveTcp \
  "{target_pose: {pose: {position: {x: 0.0, y: 0.0, z: 0.05}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}, relative: true}"

- move_joint:
  ros2 action send_goal /ur5_bridge/move_joint rfm/action/MoveJoint \
  "{target_joint: {position: [0.0, -1.57, 1.57, 0.0, 1.57, 0.0]}, relative: false}"

- move_saved:
  ros2 action send_goal /ur5_bridge/move_saved rfm/action/MoveSaved \
  "{pose_name: home}"

- manage_pose:
  ros2 service call /ur5_bridge/manage_pose rfm/srv/ManagePose "{cmd: save, name: home}"
  ros2 service call /ur5_bridge/manage_pose rfm/srv/ManagePose "{cmd: list, name: ''}"
  ros2 service call /ur5_bridge/manage_pose rfm/srv/ManagePose "{cmd: delete, name: home}"

- stop:
  ros2 service call /ur5_bridge/stop std_srvs/srv/Trigger "{}"

[Python Examples]
- ActionClient (MoveTcp):
  from rclpy.action import ActionClient
  from rfm.action import MoveTcp
  from geometry_msgs.msg import PoseStamped

  self.act_tcp = ActionClient(self, MoveTcp, "/ur5_bridge/move_tcp")
  goal = MoveTcp.Goal()
  goal.relative = True
  goal.target_pose = PoseStamped()
  goal.target_pose.pose.position.z = 0.05
  goal.target_pose.pose.orientation.w = 1.0
  future = self.act_tcp.send_goal_async(goal)

- ActionClient (MoveJoint):
  from rfm.action import MoveJoint

  self.act_joint = ActionClient(self, MoveJoint, "/ur5_bridge/move_joint")
  goal = MoveJoint.Goal()
  goal.relative = False
  goal.target_joint.position = [0.0, -1.57, 1.57, 0.0, 1.57, 0.0]
  future = self.act_joint.send_goal_async(goal)

- Service client (ManagePose / Stop):
  from rfm.srv import ManagePose
  from std_srvs.srv import Trigger

  cli_manage = self.create_client(ManagePose, "/ur5_bridge/manage_pose")
  req = ManagePose.Request()
  req.cmd = "list"
  req.name = ""
  future = cli_manage.call_async(req)

  cli_stop = self.create_client(Trigger, "/ur5_bridge/stop")
  future = cli_stop.call_async(Trigger.Request())

- Topic subscribe:
  from geometry_msgs.msg import PoseStamped
  from std_msgs.msg import String
  self.create_subscription(PoseStamped, "/ur5_bridge/tcp_pose", cb_tcp, 10)
  self.create_subscription(String, "/ur5_bridge/status", cb_status, 10)
"""

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R_sci

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from std_srvs.srv import Trigger
from rfm.action import MoveTcp, MoveJoint, MoveSaved
from rfm.srv import ManagePose

class UR5Bridge(Node):
    def __init__(self, node_name="ur5"):
        super().__init__(node_name)
        self.declare_parameters("", [
            ("robot_ip", "192.168.0.43"), 
            ("speed_l", 0.1), ("accel_l", 0.25), 
            ("speed_j", 0.5), ("accel_j", 0.5), 
            ("publish_rate", 30.0)
        ])
        
        import rtde_control, rtde_receive
        self.rtde_c = rtde_control.RTDEControlInterface(self.get_parameter("robot_ip").value)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.get_parameter("robot_ip").value)

        self._state_lock = threading.Lock()
        self._status = "IDLE"
        self.pose_db = {}
        self.pose_db_path = Path(__file__).parent / "ur5_saved_poses.json"
        self._load_db()

        # node_name 기반 상대경로 topic 사용
        self.pub_tcp = self.create_publisher(PoseStamped, "~/tcp_pose", 10)
        self.pub_status = self.create_publisher(String, "~/status", 10)
        
        cb = ReentrantCallbackGroup()
        
        # Actions
        self._action_tcp = ActionServer(self, MoveTcp, "~/move_tcp", self._cb_tcp, callback_group=cb, cancel_callback=self._cb_cancel)
        self._action_joint = ActionServer(self, MoveJoint, "~/move_joint", self._cb_joint, callback_group=cb, cancel_callback=self._cb_cancel)
        self._action_saved = ActionServer(self, MoveSaved, "~/move_saved", self._cb_saved, callback_group=cb, cancel_callback=self._cb_cancel)
        
        self.create_service(ManagePose, "~/manage_pose", self._cb_manage)
        self.create_service(Trigger, "~/stop", self._cb_stop)

        if self.get_parameter("publish_rate").value > 0:
            self.create_timer(1.0 / self.get_parameter("publish_rate").value, self._timer_cb)
        
        self.get_logger().info(f"UR5 Bridge Started as '{node_name}'")

    def _set_status(self, status):
        with self._state_lock: self._status = status
        self.pub_status.publish(String(data=status))

    def _cb_cancel(self, _): return CancelResponse.ACCEPT

    def _wait_motion(self, gh, abort_func):
        self._set_status("MOVING")
        try:
            while rclpy.ok():
                if gh.is_cancel_requested:
                    abort_func()
                    gh.canceled()
                    return False, "Canceled"
                if self.rtde_c.getAsyncOperationProgress() < 0:
                    break
                time.sleep(0.05)
            gh.succeed()
            return True, "Arrived"
        except Exception as e:
            gh.abort()
            return False, str(e)
        finally:
            self._set_status("IDLE")

    def _cb_tcp(self, gh):
        goal = gh.request
        try:
            p, q = goal.target_pose.pose.position, goal.target_pose.pose.orientation
            if goal.relative:
                curr = self.rtde_r.getActualTCPPose()
                x, y, z, rx, ry, rz = [float(v) for v in curr]
                
                # Pos
                tgt_p = [x + p.x, y + p.y, z + p.z]
                
                # Rot
                R_cur = R_sci.from_rotvec([rx, ry, rz]).as_matrix()
                q_d = [q.x, q.y, q.z, q.w]
                if np.linalg.norm(q_d) < 1e-9: q_d = [0,0,0,1]
                R_delta = R_sci.from_quat(q_d).as_matrix()
                rv_tgt = R_sci.from_matrix(R_delta @ R_cur).as_rotvec()
                
                rv_tgt = self._unwrap_rotvec_near(rv_tgt, [rx, ry, rz])
                target = tgt_p + rv_tgt.tolist()
            else:
                target = [p.x, p.y, p.z] + R_sci.from_quat([q.x, q.y, q.z, q.w]).as_rotvec().tolist()

            self.rtde_c.moveL(target, speed=self.get_parameter("speed_l").value, acceleration=self.get_parameter("accel_l").value, asynchronous=True)
            success, msg = self._wait_motion(gh, self.rtde_c.stopL)
            return MoveTcp.Result(success=success, message=msg)
        except Exception as e:
            gh.abort()
            return MoveTcp.Result(success=False, message=str(e))

    def _cb_joint(self, gh):
        goal = gh.request
        try:
            tgt = list(goal.target_joint.position)[:6]
            if len(tgt) < 6: raise ValueError("Need 6 joints")
            
            if goal.relative:
                curr = self.rtde_r.getActualQ()
                tgt = [c + t for c, t in zip(curr, tgt)]
            
            self.rtde_c.moveJ(tgt, speed=self.get_parameter("speed_j").value, acceleration=self.get_parameter("accel_j").value, asynchronous=True)
            success, msg = self._wait_motion(gh, self.rtde_c.stopJ)
            return MoveJoint.Result(success=success, message=msg)
        except Exception as e:
            gh.abort()
            return MoveJoint.Result(success=False, message=str(e))

    def _cb_saved(self, gh):
        name = gh.request.pose_name.lower()
        if name not in self.pose_db:
            gh.abort()
            return MoveSaved.Result(success=False, message="Unknown pose")
        
        entry = self.pose_db[name]
        if "joints" in entry: q = entry["joints"]
        elif "q" in entry: q = entry["q"]
        else:
            gh.abort()
            return MoveSaved.Result(success=False, message="Invalid pose data")

        self.rtde_c.moveJ(q, speed=self.get_parameter("speed_j").value, acceleration=self.get_parameter("accel_j").value, asynchronous=True)
        success, msg = self._wait_motion(gh, self.rtde_c.stopJ)
        return MoveSaved.Result(success=success, message=msg)

    def _cb_manage(self, req, resp):
        cmd, name = req.cmd.lower(), req.name.lower()
        resp.success = True
        if cmd == "save" and name:
            self.pose_db[name] = {
                "joints": [float(x) for x in self.rtde_r.getActualQ()],
                "tcp": [float(x) for x in self.rtde_r.getActualTCPPose()]
            }
            self._save_db()
            resp.message = f"Saved {name}"
        elif cmd == "list":
            resp.pose_names = sorted(list(self.pose_db.keys()))
        elif cmd == "delete" and name in self.pose_db:
            del self.pose_db[name]
            self._save_db()
            resp.message = f"Deleted {name}"
        else:
            resp.success, resp.message = False, "Invalid command"
        return resp

    def _cb_stop(self, req, resp):
        self.rtde_c.stopL()
        self.rtde_c.stopJ()
        return Trigger.Response(success=True, message="Stopped")

    def _timer_cb(self):
        try:
            pose = self.rtde_r.getActualTCPPose()
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base"
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pose[:3]
            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = \
                R_sci.from_rotvec(pose[3:]).as_quat()
            self.pub_tcp.publish(msg)
            self.pub_status.publish(String(data=self._status))
        except: pass

    def _unwrap_rotvec_near(self, rv, ref):
        v, r = np.array(rv), np.array(ref)
        th = np.linalg.norm(v)
        if th < 1e-9: return v
        k = int(np.round((np.dot(v/th, r) - th) / (2*np.pi)))
        return v + 2*np.pi*k*(v/th)

    def _load_db(self):
        if self.pose_db_path.exists():
            with open(self.pose_db_path) as f: self.pose_db = json.load(f)
    def _save_db(self):
        with open(self.pose_db_path, "w") as f: json.dump(self.pose_db, f, indent=2)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--node_name", default="ur5", help="Name of this ROS 2 node")
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = UR5Bridge(node_name=args.node_name)
    rclpy.spin(node, executor=MultiThreadedExecutor())
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__": main()
