#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive


class GroundPlanePublisher(Node):
    def __init__(self) -> None:
        super().__init__("add_ground_plane")
        self._client = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self._timer = self.create_timer(0.5, self._publish_ground_plane)
        self._sent_request = False

    def _publish_ground_plane(self) -> None:
        if self._sent_request:
            return

        if not self._client.wait_for_service(timeout_sec=0.1):
            self.get_logger().info("Waiting for /apply_planning_scene service...")
            return

        collision_object = CollisionObject()
        collision_object.header.frame_id = "world"
        collision_object.id = "ground_plane"

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [4.0, 4.0, 0.1]

        pose = Pose()
        pose.orientation.w = 1.0
        # Keep the floor slightly below z=0 so the base resting on the ground is
        # not treated as a start-state collision from numerical contact at exactly 0.
        pose.position.z = -0.055

        collision_object.primitives.append(primitive)
        collision_object.primitive_poses.append(pose)
        collision_object.operation = CollisionObject.ADD

        planning_scene = PlanningScene()
        planning_scene.is_diff = True
        planning_scene.world.collision_objects.append(collision_object)

        request = ApplyPlanningScene.Request()
        request.scene = planning_scene

        future = self._client.call_async(request)
        future.add_done_callback(self._handle_response)
        self._sent_request = True
        self._timer.cancel()

    def _handle_response(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:  # pragma: no cover - ROS service exceptions are runtime-only
            self.get_logger().error(f"Failed to apply planning scene: {exc}")
            return

        if response.success:
            self.get_logger().info("Ground plane added to planning scene.")
        else:
            self.get_logger().error("MoveIt rejected the ground plane update.")


def main() -> None:
    rclpy.init()
    node = GroundPlanePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
