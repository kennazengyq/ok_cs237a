#!/usr/bin/env python3
from asl_tb3_msgs.msg import TurtleBotState, TurtleBotControl
from asl_tb3_lib.grids import StochOccupancyGrid2D, snap_to_grid
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool
import rclpy
from rclpy.node import Node
import numpy as np
from scipy.signal import convolve2d
import typing as T


class Frontier(Node):
    def __init__(self):
        super().__init__("frontier_node")
        self.occupancy: T.Optional[StochOccupancyGrid2D] = None
        self.state: T.Optional[TurtleBotState] = None
        self.is_navigating = False
        self.exploration_complete = False
        self.visited_frontiers = []  # Track visited locations
        
        # Parameters
        self.declare_parameter("window_size", 13)
        self.declare_parameter("unknown_thresh", 0.2)
        self.declare_parameter("unoccupied_thresh", 0.3)
        self.declare_parameter("exploration_complete_thresh", 0.05)
        self.declare_parameter("min_frontier_distance", 0.3)
        self.declare_parameter("exploration_rate", 2.0)

        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid, "/map", self.map_callback, 10
        )
        self.state_sub = self.create_subscription(
            TurtleBotState, "/state", self.state_callback, 10
        )
        self.nav_success_sub = self.create_subscription(
            Bool, "/nav_success", self.nav_success_callback, 10
        )

        # Publishers - FIXED: Changed to TurtleBotState
        self.cmd_nav_pub = self.create_publisher(TurtleBotState, "/cmd_nav", 10)
        self.exploration_complete_pub = self.create_publisher(Bool, "/exploration_complete", 10)
        
        # Timer for periodic exploration checks
        exploration_period = 1.0 / self.get_parameter("exploration_rate").value
        self.exploration_timer = self.create_timer(
            exploration_period,
            self.exploration_callback
        )
        
        self.get_logger().info("Frontier node initialized. Waiting for map and state...")

    def map_callback(self, msg: OccupancyGrid) -> None:
        """Callback triggered when the map is updated"""
        self.occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=9,
            probs=msg.data,
        )

    def state_callback(self, msg: TurtleBotState) -> None:
        """Callback triggered when the robot state is updated"""
        self.state = msg

    def nav_success_callback(self, msg: Bool) -> None:
        """Callback when navigation completes or fails"""
        self.is_navigating = False
        if msg.data:
            self.get_logger().info("Navigation successful. Searching for next frontier...")
        else:
            self.get_logger().warn("Navigation failed. Searching for alternative frontier...")

    def exploration_callback(self) -> None:
        """Periodic callback to check exploration status and command new frontiers"""
        # Wait for necessary data
        if self.state is None or self.occupancy is None:
            return
        
        # Check if exploration is complete
        if self.check_exploration_complete():
            if not self.exploration_complete:
                self.exploration_complete = True
                self.get_logger().info("=" * 60)
                self.get_logger().info("EXPLORATION COMPLETE!")
                self.get_logger().info("=" * 60)
                self.exploration_complete_pub.publish(Bool(data=True))
            return
        
        # If not currently navigating, find and command next frontier
        if not self.is_navigating:
            self.command_next_frontier()

    def explore(self, occupancy: StochOccupancyGrid2D) -> np.ndarray:
        """Returns potential states to explore
        
        Args:
            occupancy: Represents the known, unknown, occupied, and unoccupied states
            
        Returns:
            frontier_states: state-vectors in (x, y) coordinates, shape (N, 2)
        """
        window_size = self.get_parameter("window_size").value
        unknown_thresh = self.get_parameter("unknown_thresh").value
        unoccupied_thresh = self.get_parameter("unoccupied_thresh").value
        
        # Create masks for different cell types
        unknown_mask = (occupancy.probs == -1.0)
        occupied_mask = (occupancy.probs >= 0.5)
        unoccupied_mask = (occupancy.probs >= 0.0) & (occupancy.probs < 0.5)
        
        # Convolve to count neighbors
        kernel = np.ones((window_size, window_size))
        unknown_counts = convolve2d(unknown_mask, kernel, mode='same', boundary='fill', fillvalue=0)
        occupied_counts = convolve2d(occupied_mask, kernel, mode='same', boundary='fill', fillvalue=0)
        unoccupied_counts = convolve2d(unoccupied_mask, kernel, mode='same', boundary='fill', fillvalue=0)
        
        # Apply thresholds
        unknown_threshold = unknown_thresh * (window_size * window_size)
        unoccupied_threshold = unoccupied_thresh * (window_size * window_size)
        
        frontier_states = []
        for x in range(occupancy.probs.shape[0]):
            for y in range(occupancy.probs.shape[1]):
                if (unknown_counts[x, y] >= unknown_threshold and 
                    occupied_counts[x, y] == 0 and 
                    unoccupied_counts[x, y] >= unoccupied_threshold):
                    frontier_states.append(occupancy.grid2state(np.array([x, y])))

        return np.array(frontier_states) if frontier_states else np.array([]).reshape(0, 2)

    def select_best_frontier(self, frontier_states: np.ndarray) -> T.Optional[np.ndarray]:
        """Select the best frontier to explore based on distance and visit history"""
        if len(frontier_states) == 0:
            return None
        
        current_pos = np.array([self.state.x, self.state.y])
        min_frontier_dist = self.get_parameter("min_frontier_distance").value
        
        # Filter out frontiers too close to previously visited ones
        valid_frontiers = []
        for frontier in frontier_states:
            too_close = False
            for visited in self.visited_frontiers:
                if np.linalg.norm(frontier - visited) < min_frontier_dist:
                    too_close = True
                    break
            if not too_close:
                valid_frontiers.append(frontier)
        
        if len(valid_frontiers) == 0:
            # If all frontiers visited, clear history and try again
            self.get_logger().info("All frontiers visited. Clearing history...")
            self.visited_frontiers.clear()
            valid_frontiers = frontier_states
        else:
            valid_frontiers = np.array(valid_frontiers)
        
        # Select closest valid frontier
        distances = np.linalg.norm(valid_frontiers - current_pos, axis=1)
        best_idx = np.argmin(distances)
        best_frontier = valid_frontiers[best_idx]
        
        self.get_logger().info(
            f"Selected frontier at ({best_frontier[0]:.2f}, {best_frontier[1]:.2f}), "
            f"distance: {distances[best_idx]:.2f}m"
        )
        
        return best_frontier

    def command_next_frontier(self) -> None:
        """Find and command the robot to navigate to the next frontier"""
        # Find all frontier points
        frontier_states = self.explore(self.occupancy)
        
        if len(frontier_states) == 0:
            self.get_logger().info("No frontiers found.")
            return
        
        self.get_logger().info(f"Found {len(frontier_states)} frontier cells")
        
        # Select best frontier
        target_frontier = self.select_best_frontier(frontier_states)
        
        if target_frontier is None:
            self.get_logger().warn("No valid frontier selected.")
            return
        
        # FIXED: Create TurtleBotState instead of TurtleBotControl
        goal_state = TurtleBotState()
        goal_state.x = float(target_frontier[0])
        goal_state.y = float(target_frontier[1])
        goal_state.theta = 0.0  # Navigator will handle orientation
        
        # Publish navigation command
        self.cmd_nav_pub.publish(goal_state)
        self.is_navigating = True
        
        # Add to visited list
        self.visited_frontiers.append(target_frontier)
        
        # Limit history size to prevent memory issues
        if len(self.visited_frontiers) > 50:
            self.visited_frontiers.pop(0)
        
        self.get_logger().info(
            f"Commanding navigation to frontier: ({goal_state.x:.2f}, {goal_state.y:.2f})"
        )

    def check_exploration_complete(self) -> bool:
        """Check if exploration is complete by measuring percentage of unknown cells"""
        unknown_mask = (self.occupancy.probs == -1.0)
        total_cells = unknown_mask.size
        unknown_cells = np.sum(unknown_mask)
        unknown_percentage = unknown_cells / total_cells
        
        exploration_thresh = self.get_parameter("exploration_complete_thresh").value
        
        if unknown_percentage <= exploration_thresh:
            self.get_logger().info(
                f"Exploration complete: {unknown_percentage*100:.1f}% unknown "
                f"(threshold: {exploration_thresh*100:.1f}%)"
            )
            return True
        
        return False


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