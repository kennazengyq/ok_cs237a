#!/usr/bin/env python3
from asl_tb3_msgs.msg import TurtleBotState, TurtleBotControl
from asl_tb3_lib.grids import StochOccupancyGrid2D, snap_to_grid
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool
import rclpy                    # ROS2 client library
from rclpy.node import Node     # ROS2 node baseclass
import numpy as np
from scipy.signal import convolve2d
import typing as Ts

class Frontier(Node):
    def __init__(self):
        super().__init__("frontier_node")
        self.occupancy: T.Optional[StochOccupancyGrid2D] = None
        self.state: T.Optional[TurtleBotState] = None

        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self.map_callback, 10)
        self.state_sub = self.create_subscription(TurtleBotState, "/state", self.state_callback, 10)
        self.nav_success_sub = self.create_subscription(Bool, "/nav_success", self.nav_success_callback, 10)

        self.cmd_nav_pub = self.create_publisher(TurtleBotState, "/cmd_nav", 10)        
        self.started = False

        self.last_stop_sign_time = None

        self.paused_pub = self.create_publisher(Bool, "/paused_topic", 10)
        
        self.paused = False
        
        self.detector_sub = self.create_subscription(
            Bool,
            "/detector_bool",
            self.detector_callback,
            10,
        )
        

    def init_callback(self) -> None:
        self.get_logger().info("Frontier node initialized, starting exploration...")
        self.command_next_frontier()

    def detector_callback(self, msg: Bool) -> None:
        """ Callback triggered when the stop sign detector updates

        Args:
            msg (Bool): stop sign detected message
        """
        if not msg.data:
            # self.get_logger().info("Stop sign not detected, navigation.")

            return
        
        curr_time = self.get_clock().now()

        msg = "last stop time is " + str(self.last_stop_sign_time)

        self.get_logger().info(msg)

        if self.last_stop_sign_time is None or (curr_time - self.last_stop_sign_time).nanoseconds / 1e9 > 7.0:
            self.last_stop_sign_time = curr_time
            self.get_logger().info("Stop sign detected, stopping navigation.")
            self.paused = True

            paused_msg = Bool()
            paused_msg.data = True
            self.paused_pub.publish(paused_msg)
            self.resume_timer = self.create_timer(5.0, self.resume_navigation)
        else:
            self.get_logger().info("Stop sign detected recently, ignoring.")

    def resume_navigation(self) -> None:
        self.get_logger().info("Resuming navigation after stop sign.")
        # self.command_next_frontier()
        self.paused = False
        self.paused_pub.publish(Bool(data=False))
        self.resume_timer.cancel()
        

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

        if not self.started and self.state is not None:
            self.get_logger().info("Map received, starting exploration...")
            self.command_next_frontier()
            self.started = True

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
        if msg.data and not self.paused:
            self.get_logger().info("Navigation to frontier succeeded, searching for next frontier...")
            self.command_next_frontier()
            
        else:
            self.get_logger().info("Nav unsuccesful")   
                

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
                    frontier_states.append(occupancy.grid2state(np.array([y, x])))
        ########################### Code ends here ###########################
        return np.array(frontier_states)
        occupancy_occupied = np.array(occupancy.probs) >= 0.5
        occupancy_unknown = np.array(occupancy.probs) < 0.0
        occupancy_unoccupied = (np.array(occupancy.probs) >= 0.0) & (np.array(occupancy.probs) < 0.5)

        kernel = np.ones((window_size, window_size))
        mid_point = window_size // 2
        kernel[mid_point, mid_point] = 0
        occupied_neighbors = convolve2d(occupancy_occupied, kernel, mode='same')
        unknown_neighbors = convolve2d(occupancy_unknown, kernel, mode='same')
        unoccupied_neighbors = convolve2d(occupancy_unoccupied, kernel, mode='same')

        frontier_cells = []

        for x in range(occupancy_occupied.shape[0]):
            for y in range(occupancy_occupied.shape[1]):
                num_neighbors = unknown_neighbors[y, x] + occupied_neighbors[y, x] + unoccupied_neighbors[y, x]
                if num_neighbors == 0:
                    continue

                unknown_perc = unknown_neighbors[y, x] / num_neighbors
                occupied_perc = occupied_neighbors[y, x] / num_neighbors
                unoccupied_perc = unoccupied_neighbors[y, x] / num_neighbors

                if occupied_perc == 0 and unknown_perc >= 0.2 and unoccupied_perc >= 0.3:
                    frontier_cells.append(np.array([x, y]))

        frontier_states = []
        for cell in frontier_cells:
            frontier_states.append(occupancy.grid2state(cell))
        if not frontier_states:
            return np.array([]).reshape(0, 2) # Return empty (0,2) array if no frontiers
        frontier_states = np.array(frontier_states)
        
    
    def command_next_frontier(self):
        frontier_states = self.explore(self.occupancy)
        if len(frontier_states) == 0:
            self.get_logger().info("No frontiers found to explore.")
            return

        distances = np.linalg.norm(frontier_states - np.array([self.state.x, self.state.y]), axis=1)
        closest_frontier = frontier_states[np.argmin(distances)]
        cmd_msg = TurtleBotState()
        cmd_msg.x = closest_frontier[0]
        cmd_msg.y = closest_frontier[1]
        cmd_msg.theta = 0.0  
        self.cmd_nav_pub.publish(cmd_msg)

        self.get_logger().info(f"Commanding navigation to frontier at: {closest_frontier}")

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