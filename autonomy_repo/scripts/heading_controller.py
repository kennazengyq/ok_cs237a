#!/usr/bin/env python3
import numpy as np
import rclpy
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

class HeadingController(BaseHeadingController):
    def __init__(self):
        super().__init__()
        self.kp = 10.0

    def compute_control_with_goal(self, currState: TurtleBotState, goalState: TurtleBotState) -> TurtleBotControl:
        error = goalState.theta - currState.theta
        wrappedDifference = wrap_angle(error)
        w = self.kp * wrappedDifference
        controlMsg = TurtleBotControl()
        controlMsg.omega = w
        return controlMsg
    

if __name__=="__main__":
    rclpy.init()
    headingController = HeadingController()
    try:
        # Spin the node (process callbacks)
        rclpy.spin(headingController)
    except KeyboardInterrupt:
        rclpy.shutdown()
