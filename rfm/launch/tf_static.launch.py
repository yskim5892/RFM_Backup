from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='tool0_to_cam',
            arguments=[
                '-0.04', '0.09', '0.03',
                '0.0', '-0.25881905', '0.96592583', '0.0',
                'tool0', 'camera_color_optical_frame'
            ],
            output='screen'
        ),
    ])

