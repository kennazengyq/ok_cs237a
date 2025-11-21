#!/usr/bin/env python3
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_msgs.msg import TurtleBotState, TurtleBotControl
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.tf_utils import quaternion_to_yaw
from asl_tb3_lib.grids import StochOccupancyGrid2D, snap_to_grid
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool
import rclpy
from rclpy.node import Node
import numpy as np
from scipy.interpolate import splev, splrep
from scipy.signal import convolve2d
import typing as T


class Navigator(BaseNavigator):
    def __init__(self, kpx: float = 1, kpy: float= 1, kdx: float= 1, kdy: float= 1,
                 node_name: str ='navigator',) -> None:
        super().__init__(node_name) 
        self.kp = 10.0
        self.kpx = kpx
        self.kpy = kpy
        self.kdx = kdx
        self.kdy = kdy
        self.V_prev = 0
        self.om_prev = 0
        self.t_prev = 0
        self.V_PREV_THRES = 0.0001
        self.coeffs = np.zeros(8)
        
        self.declare_parameter('exploration_window_size', 13)
        self.window_size = self.get_parameter('exploration_window_size').value
        
        self.current_map = None
        self.occupancy_grid = None
        self.is_exploring = False
        self.current_goal = None
        
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        
        self.nav_success_sub = self.create_subscription(Bool, '/nav_success', self.nav_success_callback, 10)
        
        self.get_logger().info("Navigator initialized with exploration enabled.")
    
    def map_callback(self, msg: OccupancyGrid) -> None:
        """Update map and create StochOccupancyGrid2D"""
        self.current_map = msg
        try:
            # Extract map data
            map_data = np.array(msg.data, dtype=np.int8)
            map_size = np.array([msg.info.width, msg.info.height])
            origin_xy = np.array([msg.info.origin.position.x, msg.info.origin.position.y])
            resolution = msg.info.resolution
            
            # Create StochOccupancyGrid2D with proper format
            self.occupancy_grid = StochOccupancyGrid2D(
                resolution=resolution,
                size_xy=map_size,
                origin_xy=origin_xy,
                window_size=self.window_size,
                probs=map_data,
                thresh=0.5
            )
        except Exception as e:
            self.get_logger().warn(f"Failed to create occupancy grid: {e}")
    
    def nav_success_callback(self, msg: Bool) -> None:
        """Handle navigation completion"""
        if msg.data:
            self.get_logger().info("Navigation succeeded")
            self.nav_failure_count = 0
            if self.is_exploring:
                self.explore_step()
        else:
            self.get_logger().warn("Navigation failed")
            self.nav_failure_count += 1
            if self.nav_failure_count >= self.max_nav_failures:
                self.get_logger().warn("Max navigation failures reached, stopping exploration")
                self.stop_exploration()
    
    def explore(self) -> T.Optional[T.Tuple[np.ndarray, float]]:
        """
        Find frontier cells using convolution-based heuristics.
        Heuristics:
            1. ≥20% unknown cells in window
            2. 0 occupied cells in window
            3. ≥30% free cells in window
        
        Returns: (closest_frontier_state, distance) tuple or None if no frontier found
        """
        if self.occupancy_grid is None or self.state is None:
            return None
        
        occupancy = self.occupancy_grid
        current_state = np.array([self.state.x, self.state.y])
        
        unknown_mask = (occupancy.probs == -1)
        occupied_mask = (occupancy.probs >= 0.5)
        free_mask = ((occupancy.probs >= 0) & (occupancy.probs < 0.5))
        
        kernel = np.ones((self.window_size, self.window_size))
        n_unknown = convolve2d(unknown_mask.astype(float), kernel, mode='same', boundary='fill')
        n_occupied = convolve2d(occupied_mask.astype(float), kernel, mode='same', boundary='fill')
        n_free = convolve2d(free_mask.astype(float), kernel, mode='same', boundary='fill')
        
        n_total = self.window_size ** 2
        frontier_mask = (
            (n_unknown / n_total >= 0.2) & 
            (n_occupied == 0) & 
            (n_free / n_total >= 0.3)
        )
        frontier_mask &= free_mask
        
        frontier_grid_indices = np.argwhere(frontier_mask)
        
        if len(frontier_grid_indices) == 0:
            self.get_logger().info("No frontier cells found - exploration complete")
            return None
        
        frontier_states = occupancy.grid2state(frontier_grid_indices)
        
        frontier_distances = np.linalg.norm(frontier_states - current_state, axis=1)
        closest_idx = np.argmin(frontier_distances)
        closest_frontier_state = frontier_states[closest_idx]
        closest_distance = frontier_distances[closest_idx]
        
        self.get_logger().info(
            f"Found {len(frontier_states)} frontier cells. "
            f"Closest at ({closest_frontier_state[0]:.2f}, {closest_frontier_state[1]:.2f}), "
            f"distance: {closest_distance:.2f}m"
        )
        
        return closest_frontier_state, closest_distance
    
    def start_exploration(self) -> None:
        """Start frontier exploration"""
        self.is_exploring = True
        self.nav_failure_count = 0
        self.get_logger().info("Starting frontier exploration")
        self.explore_step()
    
    def stop_exploration(self) -> None:
        """Stop autonomous frontier exploration"""
        self.is_exploring = False
        self.get_logger().info("Stopping frontier exploration")
    
    def explore_step(self) -> None:
        """Execute one step of frontier exploration"""
        if not self.is_exploring:
            return
        
        result = self.explore()
        
        if result is None:
            self.stop_exploration()
            return
        
        frontier_state, distance = result
        
        # Create goal state
        goal = TurtleBotState()
        goal.x = frontier_state[0]
        goal.y = frontier_state[1]
        goal.theta = self.state.theta
        
        self.current_goal = goal
        self.navigate_to(goal)
    
    def compute_heading_control(self, currState: TurtleBotState, goalState: TurtleBotState) -> TurtleBotControl:
        error = goalState.theta - currState.theta
        wrappedDifference = wrap_angle(error)
        w = self.kp * wrappedDifference
        controlMsg = TurtleBotControl()
        controlMsg.omega = w
        return controlMsg

    def compute_trajectory_tracking_control(self,
        state: TurtleBotState,
        plan: TrajectoryPlan,
        t: float,
    ) -> TurtleBotControl:
        """ Compute control target using a trajectory tracking controller
        Args:
            state (TurtleBotState): current robot state
            plan (TrajectoryPlan): planned trajectory
            t (float): current timestep
        Returns:
            TurtleBotControl: control command
        """
        x = state.x
        y = state.y
        th = state.theta
        tck_x = plan.path_x_spline
        tck_y = plan.path_y_spline
        x_d = splev(t, tck_x, der=0)    
        xd_d = splev(t, tck_x, der=1)   
        xdd_d = splev(t, tck_x, der=2)  
        
        y_d = splev(t, tck_y, der=0)    
        yd_d = splev(t, tck_y, der=1)   
        ydd_d = splev(t, tck_y, der=2)  
        dt = t - self.t_prev
        ########## Code starts here ##########
        # avoid singularity
        if abs(self.V_prev) < self.V_PREV_THRES:
            self.V_prev = self.V_PREV_THRES
        xd = self.V_prev*np.cos(th)
        yd = self.V_prev*np.sin(th)
        # compute virtual controls
        u = np.array([xdd_d + self.kpx*(x_d-x) + self.kdx*(xd_d-xd),
                      ydd_d + self.kpy*(y_d-y) + self.kdy*(yd_d-yd)])
        # compute real controls
        J = np.array([[np.cos(th), -self.V_prev*np.sin(th)],
                          [np.sin(th), self.V_prev*np.cos(th)]])
        a, om = np.linalg.solve(J, u)
        V = self.V_prev + a*dt
        ########## Code ends here ##########
        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om
        control = TurtleBotControl()
        control.omega = om
        control.v = V
        return control
    
    def reset(self) -> None:
        self.V_prev = 0.
        self.om_prev = 0.
        self.t_prev = 0.

    def compute_trajectory_plan(self,
        state: TurtleBotState,
        goal: TurtleBotState,
        occupancy: StochOccupancyGrid2D,
        resolution: float,
        horizon: float,
    ) -> T.Optional[TrajectoryPlan]:
        """ Compute a trajectory plan using A* and cubic spline fitting
        Args:
            state (TurtleBotState): state
            goal (TurtleBotState): goal
            occupancy (StochOccupancyGrid2D): occupancy
            resolution (float): resolution
            horizon (float): horizon
        Returns:
            T.Optional[TrajectoryPlan]:
        """
        astar = AStar((-horizon+state.x, -horizon+state.y), (horizon+state.x, horizon+state.y), (state.x, state.y), (goal.x, goal.y), occupancy, resolution)
        solution = astar.solve()
        if not solution:
            return None
        path = np.asarray(astar.path)
        if len(path) < 4:
            return None
        
        self.reset()
        ts = None
        path_x_spline = None
        path_y_spline = None
        x = path[:,0]
        y = path[:,1]
        v_desired = 0.15
        spline_alpha = 0.05
        distance_arr = np.diff(path, axis=0)
        distance_arr = np.sqrt(np.sum(distance_arr**2, axis=1))
        segment_time = distance_arr/v_desired
        ts = np.concatenate([[0], np.cumsum(segment_time)])
        path_x_spline = splrep(ts, x, k=3, s=spline_alpha)
        path_y_spline = splrep(ts, y, k=3, s=spline_alpha)
        
        return TrajectoryPlan(
            path=path,
            path_x_spline=path_x_spline,
            path_y_spline=path_y_spline,
            duration=ts[-1],
        )


class AStar(object):
    """Represents a motion planning problem to be solved using A*"""
    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = statespace_lo
        self.statespace_hi = statespace_hi
        self.occupancy = occupancy
        self.resolution = resolution
        self.x_offset = x_init
        self.x_init = self.snap_to_grid(x_init)
        self.x_goal = self.snap_to_grid(x_goal)
        self.closed_set = set()
        self.open_set = set()
        self.est_cost_through = {}
        self.cost_to_arrive = {}
        self.came_from = {}
        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)
        self.path = None

    def is_free(self, x):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        """
        return self.occupancy.is_free(np.asarray(x))

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance
        """
        return np.linalg.norm(np.array(x1) - np.array(x2))

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES
        """
        neighbors = []
        directions = [(0,1), (1,1), (1,0), (1,-1), (0, -1), (-1,-1), (-1,0), (-1,1)]
        for d in directions:
            neighbor = tuple(self.resolution*np.array(d) + np.array(x))
            neighbor = self.snap_to_grid(neighbor)
            if self.is_free(neighbor):
                neighbors.append(neighbor)
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def solve(self):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found
        """
        while len(self.open_set) > 0:
            x_curr = self.find_best_est_cost_through()
            if x_curr == self.x_goal:
                self.path = self.reconstruct_path()
                return True
            self.open_set.remove(x_curr)
            self.closed_set.add(x_curr)
            for x_neigh in self.get_neighbors(x_curr):
                if x_neigh in self.closed_set:
                    continue
                tentative_cost_to_arrive = self.cost_to_arrive[x_curr] + self.distance(x_curr, x_neigh)
                if x_neigh not in self.open_set:
                    self.open_set.add(x_neigh)
                elif tentative_cost_to_arrive > self.cost_to_arrive[x_neigh]:
                    continue
                self.came_from[x_neigh] = x_curr
                self.cost_to_arrive[x_neigh] = tentative_cost_to_arrive
                self.est_cost_through[x_neigh] = tentative_cost_to_arrive + self.distance(x_neigh, self.x_goal)
        return False


if __name__ == "__main__":
    rclpy.init()
    node = Navigator()
    node.start_exploration()
    rclpy.spin(node)
    rclpy.shutdown()