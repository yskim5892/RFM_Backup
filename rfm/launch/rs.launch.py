import yaml
from ament_index_python.packages import get_package_share_directory
import os
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription([
        DeclareLaunchArgument("wrist_cam", default_value="false"),
        DeclareLaunchArgument("top_cam", default_value="false"),
        DeclareLaunchArgument("both_cam", default_value="false"),

        OpaqueFunction(function=launch_setup),
    ])


def launch_setup(context, *args, **kwargs):
    wrist = LaunchConfiguration("wrist_cam").perform(context) == "true"
    top   = LaunchConfiguration("top_cam").perform(context) == "true"
    both  = LaunchConfiguration("both_cam").perform(context) == "true"

    if both:
        wrist = True
        top = True

    share = Path(get_package_share_directory('rfm'))
    yaml_path = share/ 'configs/camera_serials.yaml'
    with open(yaml_path, "r") as f:
        serials = yaml.safe_load(f)

    nodes = []

    if wrist:
        nodes.append(Node(
            package="realsense2_camera",
            executable="realsense2_camera_node",
            namespace="wrist_cam",
            name="camera",
            output="screen",
            parameters=[{
                "serial_no": serials["wrist_cam"]["serial"],
                "enable_color": True,
                "enable_depth": True,
                "align_depth.enable": True,
                "rgb_camera.profile": "640x480x15",
                "depth_module.profile": "640x480x15",
                #"hole_filling_filter.enable": True,
                #"temporal_filter.enable": True,
                #"spatial_filter.enable": True,
            }]
        ))

    if top:
        nodes.append(Node(
            package="realsense2_camera",
            executable="realsense2_camera_node",
            namespace="top_cam",
            name="camera",
            output="screen",
            parameters=[{
                "serial_no": serials["top_cam"]["serial"],
                "enable_color": True,
                "enable_depth": True,
                "align_depth.enable": True,
                "rgb_camera.profile": "640x480x15",
                "depth_module.profile": "640x480x15",
                #"hole_filling_filter.enable": True,
                #"temporal_filter.enable": True,
                #"spatial_filter.enable": True,
            }]
        ))

    return nodes

