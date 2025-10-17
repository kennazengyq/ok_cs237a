#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

# import the message type to use
from std_msgs.msg import Int64, Bool, String
from geometry_msgs.msg import Twist

class ConstantControl(Node):
    def __init__(self) -> None:
    # initialize base class (must happen before everything else)
        super().__init__("control")	
        
	    # create publisher with: self.create_publisher(<msg type>, <topic>, <qos>)
        self.control_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        
       # create a timer with: self.create_timer(<second>, <callback>)
        self.timer = self.create_timer(0.2, self.control_callback)

        self.subscription = self.create_subscription(Bool, "/kill", self.kill_callback, 10)

    def control_callback(self) -> None:
        """
        Control callback triggered by the timer
        """
       # construct seinding constant control message
        self.msg = Twist()
        self.msg.linear.x = 1.0
        # self.msg.angular.z = 2.0

       # publish message
        self.control_pub.publish(self.msg)

    def kill_callback(self, msg: Bool) -> None:
        if msg:
            self.timer.cancel()
            self.msg.linear.x = 0.0
            self.msg.angular.z = 0.0
            self.control_pub.publish(self.msg)



if __name__ == "__main__":
    rclpy.init()        # initialize ROS2 context (must run before any other rclpy call)
    node = ConstantControl()  # instantiate the ConstantControl node
    rclpy.spin(node)    # Use ROS2 built-in schedular for executing the node
    rclpy.shutdown()    # cleanly shutdown ROS2 context