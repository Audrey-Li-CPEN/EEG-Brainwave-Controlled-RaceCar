#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
import asyncio
import websockets

class SignalControlNode(Node):
    def __init__(self):
        super().__init__('signal_control_node')
        self.publisher_ = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.websocket_url = "ws://localhost:5000/signal"
        self.signal = None
        self.get_logger().info("Signal Control Node Initialized.")

        asyncio.get_event_loop().run_until_complete(self.start_websocket())

        self.timer = self.create_timer(0.02, self.publish_drive_command)  # 50Hz

    async def start_websocket(self):
        try:
            async with websockets.connect(self.websocket_url) as websocket:
                self.get_logger().info(f"Connected to WebSocket server at {self.websocket_url}")
                while True:
                    message = await websocket.recv()  
                    self.handle_signal(message)
        except Exception as e:
            self.get_logger().error(f"WebSocket error: {e}")

    def handle_signal(self, message):
        try:
            import json
            data = json.loads(message)
            self.signal = data.get("direction")                
        except json.JSONDecodeError as e:
            self.get_logger().warn(f"Invalid message received: {message}")

    def publish_drive_command(self):
        if not self.signal:
            return 

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = 0.5 

        if self.signal == "left":
            drive_msg.drive.steering_angle = 0.2  
        elif self.signal == "right":
            drive_msg.drive.steering_angle = -0.2  
        elif self.signal == "stop":
            drive_msg.drive.speed = 0.0  
            drive_msg.drive.steering_angle = 0.0
        elif self.signal == "rest":
            return  
        else:
            self.get_logger().warn(f"Invalid signal: {self.signal}")
            return

        self.publisher_.publish(drive_msg)
        self.get_logger().info(f"Published control command: {self.signal}")

def main(args=None):
    rclpy.init(args=args)
    node = SignalControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
