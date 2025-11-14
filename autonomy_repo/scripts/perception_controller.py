#!/usr/bin/env python3 
import numpy as np 
import rclpy 
from asl_tb3_lib.control import BaseHeadingController 
from asl_tb3_lib.math_utils import wrap_angle 
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState 
from std_msgs.msg import Bool
 
class PerceptionController(BaseHeadingController): 
    def __init__(self): 
        super().__init__("perception_controller")
        self.kp = 10.0 
        self.declare_parameter("active", True)
 
        # Task 4.1 - Create Boolean variable image_detected
        self.image_detected = False
 
        # Task 4.1 - Create subscriber to /detector_bool topic
        self.detector_sub = self.create_subscription(
            Bool,
            '/detector_bool',
            self.detector_callback,
            10
        )
 
    @property
    def active(self) -> bool:
        return self.get_parameter("active").value
 
    # Task 4.1 - Callback function for detector
    def detector_callback(self, msg: Bool):
        """Callback that sets image_detected when target object is detected"""
        self.image_detected = msg.data
        if self.image_detected:
            self.get_logger().info("Target object detected! Stopping rotation.")
        else:
            self.get_logger().info("No target object detected. Continuing rotation.")
 
    # Task 4.2 - Updated compute_control_with_goal method
    def compute_control_with_goal(self, currState: TurtleBotState, goalState: TurtleBotState) -> TurtleBotControl: 
        controlMsg = TurtleBotControl() 
 
        # Set omega to 0.2 if image NOT detected, 0 if image IS detected
        if not self.image_detected:
            controlMsg.omega = 0.2
        else:
            controlMsg.omega = 0.0
 
        return controlMsg 
 
 
if __name__=="__main__": 
    rclpy.init() 
    perceptionController = PerceptionController() 
    try: 
        # Spin the node (process callbacks) 
        rclpy.spin(perceptionController) 
    except KeyboardInterrupt: 
 
            rclpy.shutdown()

     

