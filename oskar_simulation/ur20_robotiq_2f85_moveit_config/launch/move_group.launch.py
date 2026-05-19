from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("ur20_robotiq_2f85", package_name="ur20_robotiq_2f85_moveit_config").to_moveit_configs()
    return generate_move_group_launch(moveit_config)
