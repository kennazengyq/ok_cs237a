#!/usr/bin/env python3
from asl_tb3_msgs.msg import TurtleBotState, TurtleBotControl
from asl_tb3_lib.grids import StochOccupancyGrid2D, snap_to_grid
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool
import rclpy                    # ROS2 client library
from rclpy.node import Node     # ROS2 node baseclass
import numpy as np
from scipy.signal import convolve2d
import typing as T

class Frontier(Node):
    def __init__(self):
        super().__init__("frontier_node")
        self.occupancy: T.Optional[StochOccupancyGrid2D] = None
        self.state: T.Optional[TurtleBotState] = None

        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, 10)
        self.state_sub = self.create_subscription(TurtleBotState, "/state", self.state_callback, 10)
        self.nav_success_sub = self.create_subscription(Bool, "/nav_success", self.nav_success_callback, 10)

        self.cmd_nav_pub = self.create_publisher(TurtleBotState, "/cmd_nav", 10)


    def map_callback(self, msg: OccupancyGrid) -> None:
        """ Callback triggered when the map is updated

        Args:
            msg (OccupancyGrid): updated map message
        """
        self.occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=9,
            probs=msg.data,
        )

    def state_callback(self, msg: TurtleBotState) -> None:
        """ Callback triggered when the robot state is updated

        Args:
            msg (TurtleBotState): updated robot state message
        """
        self.state = msg

    def nav_success_callback(self, msg: Bool) -> None:
        """ Triggers search for next frontier point

        Args:
            msg (Bool): navigation success message
        """
        if msg.data and self.occupancy is not None and self.state is not None:
            frontier_states = self.explore(self.occupancy)
            if frontier_states.shape[0] > 0:
                closest_frontier = frontier_states[np.argmin(np.linalg.norm(frontier_states - np.array([self.state.x, self.state.y]), axis=1))]
                self.get_logger().info(f"Next frontier to explore: {closest_frontier}")
                cmd_nav_msg = TurtleBotControl()
                cmd_nav_msg.x = closest_frontier[0]
                cmd_nav_msg.y = closest_frontier[1]
                self.cmd_nav_pub.publish(cmd_nav_msg)
            else:
                self.get_logger().info("No frontiers found to explore.")

    def explore(self, occupancy):
        """ returns potential states to explore
        Args:
            occupancy (StochasticOccupancyGrid2D): Represents the known, unknown, occupied, and unoccupied states. See class in first section of notebook.

        Returns:
            frontier_states (np.ndarray): state-vectors in (x, y) coordinates of potential states to explore. Shape is (N, 2), where N is the number of possible states to explore.

        HINTS:
        - Function `convolve2d` may be helpful in producing the number of unknown, and number of occupied states in a window of a specified cell
        - Note the distinction between physical states and grid cells. Most operations can be done on grid cells, and converted to physical states at the end of the function with `occupancy.grid2state()`
        """
        '''
            1. The percentage of unknown cells surrounding a cell in some window should be greater than or equal
            to 20% of the surrounding cells.
            2. The number of known, occupied cells surrounding a cell in some window should be 0.
            3. The percentage of known, unoccupied cells surrounding a cell in some window should be greater than
            or equal to 30% of the surrounding cells.
        '''

        window_size = 13    # defines the window side-length for neighborhood of cells to consider for heuristics
        ########################### Code starts here ###########################
        unknown_mask = (occupancy.probs == -1.0)
        occupied_mask = (occupancy.probs >= 0.5)
        unoccupied_mask = (occupancy.probs >= 0.0) & (occupancy.probs < 0.5)
        kernel = np.ones((window_size, window_size))

        unknown_counts = convolve2d(unknown_mask, kernel, mode='same', boundary='fill', fillvalue=0)
        occupied_counts = convolve2d(occupied_mask, kernel, mode='same', boundary='fill', fillvalue=0)
        unoccupied_counts = convolve2d(unoccupied_mask, kernel, mode='same', boundary='fill', fillvalue=0)
        
        unknown_thresh = 0.2 * (window_size * window_size)
        unoccupied_thresh = 0.3 * (window_size * window_size)
        frontier_states = []
        for x in range(occupancy.probs.shape[0]):
            for y in range(occupancy.probs.shape[1]):
                if (unknown_counts[x, y] >= unknown_thresh and occupied_counts[x, y] == 0 and unoccupied_counts[x, y] >= unoccupied_thresh):
                    frontier_states.append(occupancy.grid2state(np.array([x, y])))

        ########################### Code ends here ###########################
        return np.array(frontier_states)

def main(args=None):
    rclpy.init(args=args)
    node = Frontier()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Exploration interrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()