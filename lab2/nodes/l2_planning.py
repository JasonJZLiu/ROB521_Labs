#!/usr/bin/env python3
#Standard Libraries
import numpy as np
import yaml
import pygame
import time
import pygame_utils
import matplotlib.image as mpimg
from skimage.draw import disk
from scipy.linalg import block_diag
from scipy.spatial import cKDTree
from scipy.linalg import inv



def load_map(filename):
    im = mpimg.imread("../maps/" + filename)
    if len(im.shape) > 2:
        im = im[:,:,0]
    im_np = np.array(im)  #Whitespace is true, black is false
    #im_np = np.logical_not(im_np)    
    return im_np


def load_map_yaml(filename):
    with open("../maps/" + filename, "r") as stream:
            map_settings_dict = yaml.safe_load(stream)
    return map_settings_dict

#Node for building a graph
class Node:
    def __init__(self, pose, parent_id, cost):
        self.pose = pose # [x, y, theta] of shape (3,)
        self.parent_id = parent_id # The parent node id that leads to this node (There should only ever be one parent in RRT)
        self.cost = cost # The cost to come to this node
        self.children_ids = [] # The children node ids of this node
        self.pose_traj_to_children = dict() # Maps node_id of children to a a trajectory array of shape (N, 3)
        return

#Path Planner 
class PathPlanner:
    #A path planner capable of perfomring RRT and RRT*
    def __init__(self, map_filename, map_setings_filename, goal_point, stopping_dist):
        #Get map information
        self.occupancy_map = load_map(map_filename)
        self.map_shape = self.occupancy_map.shape
        self.map_settings_dict = load_map_yaml(map_setings_filename)

        #Get the metric bounds of the map
        # [x1 x2]
        # [y1 y2]
        self.bounds = np.zeros([2,2]) #m
        self.bounds[0, 0] = self.map_settings_dict["origin"][0]
        self.bounds[1, 0] = self.map_settings_dict["origin"][1]
        self.bounds[0, 1] = self.map_settings_dict["origin"][0] + self.map_shape[1] * self.map_settings_dict["resolution"]
        self.bounds[1, 1] = self.map_settings_dict["origin"][1] + self.map_shape[0] * self.map_settings_dict["resolution"]
        # [x1 y1]
        # [x2 y2]
        self.bounds = self.bounds.T

        #Robot information
        self.robot_radius = 0.22 #m

        # self.vel_max = 0.26 #m/s (Feel free to change!)
        # self.rot_vel_max = 1.82 #0.2 #rad/s (Feel free to change!)

        self.vel_max = 1 #m/s (Feel free to change!)
        self.rot_vel_max = 1.82 #0.2 #rad/s (Feel free to change!)

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m
        self.best_goal_node_id = -1
        self.goal_node_ids = list()
        self.best_goal_pose_traj = np.empty((0, 3))
        self.best_goal_pose_path = list()

        #Trajectory Simulation Parameters
        self.timestep = 1.0 #s
        self.num_substeps = 20 #10

        #Trajectory Rollout Options
        vel_grid_res = 5 #4
        rot_vel_res = 9 #7
        num_vel = vel_grid_res*rot_vel_res

        vel_range = np.linspace(-self.vel_max, self.vel_max, vel_grid_res)
        rot_vel_range = np.linspace(-self.rot_vel_max, self.rot_vel_max, rot_vel_res)
        vel_grid, rot_vel_grid = np.meshgrid(vel_range, rot_vel_range, indexing='ij')
        # (num_vel, 1)
        vel_candidates = vel_grid.flatten().reshape(-1, 1)
        rot_vel_candidates = rot_vel_grid.flatten().reshape(-1, 1)

        self.vel_options = np.hstack([vel_candidates, rot_vel_candidates])

        # if there is a [0, 0] option, remove it
        all_zeros_index = (np.abs(self.vel_options) < [0.001, 0.001]).all(axis=1).nonzero()[0]
        if all_zeros_index.size > 0:
            self.vel_options = np.delete(self.vel_options, all_zeros_index, axis=0)

        #Planning storage
        self.nodes = [Node(np.zeros((3,)), -1, 0)]

        #Sampling Parameters
        self.sampling_space_params = dict()
        self.sampling_space_params["goal_range_multiplier"] = 4

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5
        
        #Pygame window for visualization
        # self.window = pygame_utils.PygameWindow(
        #     "Path Planner", (2500, 2500), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        self.window = pygame_utils.PygameWindow(
            "Path Planner", (3000, 3000), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        return



    #Functions required for RRT
    def sample_map_space(self, visualize=0):
        #Return an [x,y] coordinate to drive the robot towards

        sample_goal_range_prob = np.random.rand()
        if sample_goal_range_prob < 0.06:
            # samples the goal position
            sample_goal_prob = np.random.rand()
            if sample_goal_prob < 0.5:
                sample = self.goal_point
            else:
                k = self.sampling_space_params["goal_range_multiplier"]
                delta = k*self.stopping_dist * np.random.randn(2,)
                sample = self.goal_point + delta
        else:
            if self.best_goal_node_id == -1:
                # a path has not been found yet
                sample = (self.bounds[1] - self.bounds[0])*np.random.rand(2) + self.bounds[0]
                sample = sample.reshape(2,)
            else:
                # perform informed RRT* sampling
                a = self.sampling_space_params["a"]
                center = self.sampling_space_params["center"]

                angle = np.random.uniform(0, 2*np.pi)
                r = a * np.sqrt(np.random.uniform(0, 1))
                x = center[0] + r * np.cos(angle)
                y = center[1] + r * np.sin(angle)
                sample = np.array([x, y])
            
        if visualize != 0:
            self.window.add_point(sample, radius=5, color=(255, 0, 0), update=True)
        
        return sample
          
    
    def check_if_duplicate(self, pose):
        #Check if point is a duplicate of an already existing node
        # pose: (3,)

        # constructing a KDTree with the full pose (x, y, theta)
        nodes_KDTree = cKDTree([node.pose for node in self.nodes])
        distance, node_idx = nodes_KDTree.query(pose)
        if np.isclose(distance, 0.0):
            return True
        else:
            return False

        # # constructing a KDTree with the full pose (x, y, theta)
        # nodes_KDTree = cKDTree([node.pose[0:2] for node in self.nodes])
        # distance, node_idx = nodes_KDTree.query(pose[0:2])
        # if np.isclose(distance, 0.0):
        #     return True
        # else:
        #     return False

    
    def closest_nodes(self, point, k=1):
        #Returns the index of the closest node
        # point: (2,)

        # re-construct the KDTree:
        nodes_KDTree = cKDTree([node.pose[0:2] for node in self.nodes])
        distance, node_idx = nodes_KDTree.query(point, k)
        return node_idx


    def collision_check(self, points):
        # expects point to be of shape (num_pts, 2)
        # returns a boolean array of shape (num_pts, ), where True means collision
        
        # (num_pts, num_pts_per_circle, 2)
        occupied_coords = self.points_to_robot_circle(points)
        # occupancy_map = 0 means it is in collision (num_pts, num_pts_per_circle)
        collision_mask_per_point = self.occupancy_map[occupied_coords[:, :, 1], occupied_coords[:, :, 0]] == 0
        # (num_pts, )
        collision_mask = np.any(collision_mask_per_point, axis=1)
        return collision_mask


    def simulate_trajectory(self, node_i, point_s, visualize=0):
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does has many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y]
        
        # Return: (N, 3), where N can change depending on the length of the trajectory
        vel, rot_vel, robot_traj = self.robot_controller(node_i, point_s, visualize)
        return robot_traj
    

    def robot_controller(self, node_i, point_s, visualize=0):
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced

        # Idea: perform trajectory rollout on various velocities, select the point closest to point_s
        point_s = point_s.reshape(1, 2)

        vel_candidates = self.vel_options[:, 0]
        rot_vel_candidates = self.vel_options[:, 1]

        # rollout the trajectories given the velocities
        pose_traj, last_valid_substep, last_valid_poses = self.trajectory_rollout(
            vel_candidates, rot_vel_candidates, node_i.pose, self.timestep, self.num_substeps
        )

        # measure each trajectory's final position error
        last_position_error = np.linalg.norm(last_valid_poses[:, 0:2] - point_s, axis=1)
        best_vel_idx = np.argmin(last_position_error)
        best_vel = vel_candidates[best_vel_idx]
        best_rot_vel = rot_vel_candidates[best_vel_idx]
        best_traj = pose_traj[best_vel_idx, 0:last_valid_substep[best_vel_idx]+1, :]

        if visualize != 0:
            if visualize == 2:
                # draw all paths
                for i in range(pose_traj.shape[0]):
                    for j in range(pose_traj.shape[1]):
                        if not np.all(np.isnan(pose_traj[i, j]) == np.isnan(np.array([np.nan, np.nan, np.nan]))):
                            self.window.add_point(pose_traj[i, j, 0:2], radius=2, color=(255, 0, 0), update=False)
                # draw all end points
                for i in range(last_valid_poses.shape[0]):
                    self.window.add_point(last_valid_poses[i, 0:2], radius=2, color=(0, 255, 0), update=False)
            # draw best path
            for i in range(best_traj.shape[0]):
                self.window.add_point(best_traj[i, 0:2], radius=2, color=(0, 0, 255), update=False)
            self.window.update()

        return best_vel, best_rot_vel, best_traj


    def trajectory_rollout(self, vel, rot_vel, pose, timestep, num_substeps):
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions

        # Args:
        # vel: (N, 1)
        # rot_vel: (N, 1)
        # pose: (3, )

        # Returns:
        # an array of shape (num_vel, num_substeps, 3), where points post collision are filled with np.nan
        # an array of shape (num_vel,) that indicates the last collision-free substep idx 
        # an array of shape (num_vel, 3) that represents the last collision-free pose

        # kinematics closed-form solution:
        # x(t) = x_0 + (v/w)*[sin(wt+theta_0) - sin(theta_0)]
        # y(t) = y_0 - (v/w)*[cos(wt+theta_0) - cos(theta_0)]
        # theta(t) = theta_0 + w*t

        num_vel = vel.shape[0]
        x_0, y_0, theta_0 = pose.reshape(3,)

        # (num_vel, 1)
        vel = vel.reshape(-1, 1)
        rot_vel = rot_vel.reshape(-1, 1)

        # (num_substeps, )
        t = np.linspace(0, timestep, num_substeps)

        # simulate trajectories for all timesteps (num_vel, num_substeps)
        x_traj = np.where(
            np.isclose(rot_vel, 0), 
            x_0 + vel*t*np.cos(theta_0), 
            x_0 + (vel/rot_vel)*(np.sin(rot_vel*t + theta_0) - np.sin(theta_0)),
        )
        y_traj = np.where(
            np.isclose(rot_vel, 0), 
            y_0 + vel*t*np.sin(theta_0), 
            y_0 - (vel/rot_vel)*(np.cos(rot_vel*t + theta_0) - np.cos(theta_0)),
        )
        theta_traj = theta_0 + rot_vel*t

        # normalize thetas to be between -pi and pi
        theta_traj = (theta_traj + np.pi) % (2 * np.pi) - np.pi

        # (num_vel, num_substeps, 3)
        pose_traj = np.dstack((x_traj, y_traj, theta_traj))

        # (num_vel, num_substeps)
        collision_mask = self.collision_check(pose_traj.reshape(-1, 3)[:, 0:2]).reshape(-1, num_substeps)
        # set to True for all substeps after the first True (num_vel, num_substeps)
        collision_mask_corrected = np.cumsum(collision_mask, axis=1).astype(bool)
        # (num_vel, num_substeps, 3)
        pose_traj[collision_mask_corrected] = np.array([np.nan, np.nan, np.nan])

        # first substep that encounters a collision
        first_collision_indices = np.argmax(collision_mask_corrected, axis=1)
        # rows that do not have collisions
        no_collision_indices = np.all(~collision_mask_corrected, axis=1)
        # for no collision rows, return the last substep idx
        # for rows with collisions, return the last valid idx
        last_valid_substep = np.where(
            no_collision_indices, 
            num_substeps - 1, 
            first_collision_indices - 1
        )
        
        last_valid_poses = pose_traj[np.arange(num_vel), last_valid_substep, :]
        return pose_traj, last_valid_substep, last_valid_poses

    
    def point_to_cell(self, point):
        #Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
        #point is a N by 2 matrix of points of interest
        point = point.copy()
        # origin vector is from map to world coord
        origin_x, origin_y, theta = self.map_settings_dict["origin"]
        # we want world to map coord, so we subtract origin vector first
        point += -np.array([[origin_x, origin_y]])
        # to convert from map coord to map pixel coord, we divide by resolution
        point = point / self.map_settings_dict["resolution"]
        # so far, y = 0 is at the bottom, but in an image y = 0 is at the top
        point[:, 1] = self.map_shape[0] - point[:, 1] 
        # round to integer for pixel coordinate (N, 2)
        point = point.astype(int)
        return point


    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function
        # points (N, 2)

        # (num_points, 2)
        pixel_coords = self.point_to_cell(points)
        r_px = self.robot_radius / self.map_settings_dict["resolution"]

        rr, cc = disk((0, 0), r_px)
        # A list of pixel coords that draw out a cirlce at the origin (num_px_per_circle, 2)
        circle_px_coords = np.array([rr, cc]).T
        # (num_points, num_px_per_circle, 2)
        circle_points = pixel_coords[:, np.newaxis, :] + circle_px_coords
        # clip any pixel coords outside of the range to within the range
        circle_points = np.clip(circle_points, a_min=[0, 0], a_max=[self.map_shape[0]-1, self.map_shape[1]-1])
        return circle_points


    #RRT* specific functions
    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)
    

    def connect_node_to_point(self, node_i, point_f, visualize=0):
        #Given two nodes find the non-holonomic path that connects them
        #Settings
        #node is a 3 by 1 node
        #point is a 2 by 1 point
        
        pose_i = node_i.pose.reshape(3,)
        x_w_0 = pose_i[0]
        y_w_0 = pose_i[1]
        theta_0 = pose_i[2]

        point_f = point_f.reshape(2,)
        x_w_f = point_f[0]
        y_w_f = point_f[1]

        # find point_f in robot frame
        w_T_r = np.array([
            [np.cos(pose_i[2]), -np.sin(pose_i[2]), pose_i[0]],
            [np.sin(pose_i[2]),  np.cos(pose_i[2]), pose_i[1]],
            [0                ,  0                ,  1.0     ],
        ])
        r_T_w = inv(w_T_r)
        point_f_robot_frame = r_T_w @ np.array([point_f[0], point_f[1], 1.0]).T
        x_r_f = point_f_robot_frame[0]
        y_r_f = point_f_robot_frame[1]

        if np.isclose(y_r_f, 0.0):
            # drive straight since no lateral deviation in robot's frame
            # substeps = np.ceil(d / self.robot_radius).astype(np.int64) + 1
            pose_s = np.array([x_w_f, y_w_f, theta_0])
            pose_traj = np.linspace(pose_i, pose_s, self.num_substeps)[None, ...]
        else:
            # in robot's frame, define a circle that passes through
            # the robot's current position (0, 0) to the point_f in
            # robot's frame. The circle must also have a slope of 0
            # at (0, 0) to match the robot's current heading, causing
            # the circle's center to lie on the y-axis of robot's frame
            traj_radius = (x_r_f**2 + y_r_f**2) / (2*y_r_f)
            traj_center_robot_frame = np.array([0.0, traj_radius, 1]).T
            traj_center = (w_T_r @ traj_center_robot_frame).reshape(3,)[0:2]
            # self.window.add_point(traj_center, radius=5)

            # calculate the linear and angular velocities to achieve traj_radius
            rot_vel = self.rot_vel_max
            vel = traj_radius * rot_vel

            if vel > self.vel_max:
                vel = self.vel_max
                rot_vel = vel / traj_radius

            # calculate time step assuming maximum angular velocity
            theta_f = np.arctan2(
                (x_w_f - x_w_0) / traj_radius + np.sin(theta_0),
                -(y_w_f - y_w_0) / traj_radius + np.cos(theta_0),
            )
            angle_distance = (theta_f - theta_0 + np.pi) % (2*np.pi) - np.pi
            timestep = (angle_distance) / rot_vel

            if timestep < 0:
                timestep *= -1
                rot_vel *= -1
                vel *= -1

            # (num_substeps, )
            t = np.linspace(0, timestep, self.num_substeps)

            # simulate trajectories for all timesteps (num_vel, num_substeps)
            x_traj = np.where(
                np.isclose(rot_vel, 0), 
                x_w_0 + vel*t*np.cos(theta_0), 
                x_w_0 + (vel/rot_vel)*(np.sin(rot_vel*t + theta_0) - np.sin(theta_0)),
            )
            y_traj = np.where(
                np.isclose(rot_vel, 0), 
                y_w_0 + vel*t*np.sin(theta_0), 
                y_w_0 - (vel/rot_vel)*(np.cos(rot_vel*t + theta_0) - np.cos(theta_0)),
            )
            theta_traj = theta_0 + rot_vel*t

            # normalize thetas to be between -pi and pi
            theta_traj = (theta_traj + np.pi) % (2 * np.pi) - np.pi

            # (1, num_substeps, 3)
            pose_traj = np.dstack((x_traj, y_traj, theta_traj))


        # (1, num_substeps)
        collision_mask = self.collision_check(pose_traj.reshape(-1, 3)[:, 0:2]).reshape(-1, self.num_substeps)
        # set to True for all substeps after the first True (1, num_substeps)
        collision_mask_corrected = np.cumsum(collision_mask, axis=1).astype(bool)
        # (1, num_substeps, 3)
        pose_traj[collision_mask_corrected] = np.array([np.nan, np.nan, np.nan])

        # first substep that encounters a collision
        first_collision_indices = np.argmax(collision_mask_corrected, axis=1)
        # rows that do not have collisions
        no_collision_indices = np.all(~collision_mask_corrected, axis=1)
        # for no collision rows, return the last substep idx
        # for rows with collisions, return the last valid idx
        last_valid_substep = np.where(
            no_collision_indices, 
            self.num_substeps - 1, 
            first_collision_indices - 1
        )
        
        last_valid_poses = pose_traj[np.arange(1), last_valid_substep, :]

        if visualize != 0:
            for i in range(pose_traj[0].shape[0]):
                self.window.add_se2_pose(pose_traj[0, i], color=(0, 0, 255), length=10)

        return pose_traj[0], last_valid_substep[0], last_valid_poses[0]

    
    def cost_to_come(self, trajectory_o):
        #The cost to get to a node from lavalle 
        # trajectory_o: (N, 3) 
        cost_to_come = np.linalg.norm(trajectory_o[1:, 0:2] - trajectory_o[0:-1, 0:2], axis=1).sum()
        return cost_to_come

    
    def update_children(self, node_id):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        parent_node = self.nodes[node_id]
        for child_id in parent_node.children_ids:
            # (N, 3)
            pose_traj_to_children = parent_node.pose_traj_to_children[child_id]
            edge_cost = self.cost_to_come(pose_traj_to_children)
            self.nodes[child_id].cost = parent_node.cost + edge_cost
            self.update_children(child_id)


    #Planner Functions
    def rrt_planning(self):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        while True:
            # Sample map space
            point = self.sample_map_space(visualize=0)

            # Get the closest point
            closest_node_id = self.closest_nodes(point, k=1)

            # Simulate driving the robot towards the closest point
            pose_traj = self.simulate_trajectory(self.nodes[closest_node_id], point, visualize=1)

            # Check if the new node pose is a duplicate and skip if it is
            new_node_pose = pose_traj[-1, :]
            if self.check_if_duplicate(new_node_pose):
                continue

            # Add the last pose in the trajectory as a Node and update closest node's children            
            new_node = Node(
                pose=new_node_pose,
                parent_id=closest_node_id,
                cost=0,
            )
            self.nodes.append(new_node)
            new_node_id = len(self.nodes) - 1
            self.nodes[closest_node_id].children_ids.append(new_node_id)
            self.nodes[closest_node_id].pose_traj_to_children[new_node_id] = pose_traj
            
            # Check if goal has been reached
            if np.linalg.norm(new_node_pose[0:2] - self.goal_point) <= self.stopping_dist:
                self.best_goal_node_id = new_node_id
                break
        return self.nodes
    
    def rrt_star_planning(self, visualize=1, max_iteration=40000, save_path=True):
        #This function performs RRT* for the given map and robot
        iteration = 0
        while True:
            iteration += 1
            # print(iteration)

            # Sample map space
            point = self.sample_map_space(visualize=0)

            # Get the closest point
            closest_node_id = self.closest_nodes(point, k=1)

            # Simulate driving the robot from the closest point to the sampled point
            pose_traj_from_closest_node = self.simulate_trajectory(self.nodes[closest_node_id], point, visualize=visualize)

            # Check if the new node pose is a duplicate and skip if it is
            new_node_pose = pose_traj_from_closest_node[-1, :]
            if self.check_if_duplicate(new_node_pose):
                continue

            # Compute the cost-to-come of the new node
            new_node_cost_to_come = self.nodes[closest_node_id].cost + self.cost_to_come(pose_traj_from_closest_node)

            # Add the last pose in the trajectory as a Node and update closest node's children            
            new_node = Node(
                pose=new_node_pose,
                parent_id=closest_node_id,
                cost=new_node_cost_to_come, 
            )
            self.nodes.append(new_node)
            new_node_id = len(self.nodes) - 1
            self.nodes[closest_node_id].children_ids.append(new_node_id)
            self.nodes[closest_node_id].pose_traj_to_children[new_node_id] = pose_traj_from_closest_node


            # Query the KDTree for all node ids within the ball_radius
            nodes_KDTree = cKDTree([node.pose[0:2] for node in self.nodes])
            close_node_ids = nodes_KDTree.query_ball_point(new_node_pose[0:2], self.ball_radius())
            # Remove the new_node id in close_node_indices
            close_node_ids.remove(new_node_id)

            # Find the best node within the ball_radius to connect to new_node
            best_node_id = None
            best_cost_to_come = new_node.cost
            best_pose_traj = None
            for close_node_id in close_node_ids:
                # compute the potential connection between the close_node to the new_node
                potential_pose_traj, last_valid_substep, last_valid_pose = self.connect_node_to_point(
                    self.nodes[close_node_id], new_node.pose[0:2], visualize=0
                )

                # reject path with collisions
                if last_valid_substep != (potential_pose_traj.shape[0]-1):
                    # cannot connect to new_node due to collision
                    continue

                potential_cost_to_come = self.nodes[close_node_id].cost + self.cost_to_come(potential_pose_traj)
                if potential_cost_to_come < best_cost_to_come:
                    best_cost_to_come = potential_cost_to_come
                    best_node_id = close_node_id
                    best_pose_traj = potential_pose_traj
            
            # Re-wire connection to new_node from nodes within the ball_radius
            if best_node_id is not None:
                # there exists a node that yields a lower cost-to-come than the currently wired node
                # hence needs re-wiring

                # remove existing connection from closest_node_id
                if visualize != 0:
                    existing_pose_traj = self.nodes[closest_node_id].pose_traj_to_children[new_node_id]
                    for i in range(1, existing_pose_traj.shape[0]):
                        self.window.remove_point(existing_pose_traj[i, 0:2], radius=2, update=False)
                        # self.window.add_point(existing_pose_traj[i, 0:2], radius=2, color=(255, 0, 0), update=False)
                    self.window.update()
                self.nodes[closest_node_id].children_ids.remove(new_node_id)
                del self.nodes[closest_node_id].pose_traj_to_children[new_node_id]
                
                # re-wire to connect from best_node
                new_node.pose = best_pose_traj[-1]
                new_node.parent_id = best_node_id
                new_node.cost = best_cost_to_come
                self.nodes[best_node_id].children_ids.append(new_node_id)
                self.nodes[best_node_id].pose_traj_to_children[new_node_id] = best_pose_traj

                if visualize != 0:
                    for i in range(best_pose_traj.shape[0]):
                        self.window.add_point(best_pose_traj[i, 0:2], radius=2, color=(0, 0, 255), update=False)
                    self.window.update()
                

            # For all nodes within the ball_radius, try connecting to them from new_node
            # and see if the path including new_node yields a lower cost-to-come
            for close_node_id in close_node_ids:
                # compute the potential connection between new_node to a close_node
                potential_pose_traj, last_valid_substep, last_valid_pose = self.connect_node_to_point(
                    new_node, self.nodes[close_node_id].pose[0:2], visualize=0
                )

                # reject path with collisions
                if last_valid_substep != (potential_pose_traj.shape[0]-1):
                    # cannot connect to new_node due to collision
                    continue
                
                potential_cost_to_come = new_node.cost + self.cost_to_come(potential_pose_traj)
                if potential_cost_to_come < self.nodes[close_node_id].cost:
                    # remove close_node's parent node's information
                    close_node_parent_id = self.nodes[close_node_id].parent_id
                    if visualize != 0:
                        existing_pose_traj = self.nodes[close_node_parent_id].pose_traj_to_children[close_node_id]
                        for i in range(1, existing_pose_traj.shape[0]):
                            self.window.remove_point(existing_pose_traj[i, 0:2], radius=2, update=False)
                        self.window.update()
                    self.nodes[close_node_parent_id].children_ids.remove(close_node_id)
                    del self.nodes[close_node_parent_id].pose_traj_to_children[close_node_id]

                    # re-wire from new_node to close_node
                    self.nodes[close_node_id].parent_id = new_node_id
                    self.nodes[close_node_id].cost = potential_cost_to_come
                    new_node.children_ids.append(close_node_id)
                    new_node.pose_traj_to_children[close_node_id] = potential_pose_traj
                    self.update_children(close_node_id)

                    if visualize != 0:
                        for i in range(potential_pose_traj.shape[0]):
                            self.window.add_point(potential_pose_traj[i, 0:2], radius=2, color=(0, 0, 255), update=False)
                        self.window.update()
            

            # Check if goal has been reached
            if np.linalg.norm(new_node_pose[0:2] - self.goal_point) <= self.stopping_dist:
                self.goal_node_ids.append(new_node_id)
                if (self.best_goal_node_id == -1):
                    # setting self.best_goal_node_id for the first time
                    print("SOLUTION FOUND")
                    self.best_goal_node_id = new_node_id
                    self.best_goal_pose_path, self.best_goal_pose_traj = self.recover_path(
                        self.best_goal_node_id, visualize=visualize
                    )
                    self.update_sampling_space()
                    if save_path:
                        np.save("rrt_star_path.npy", np.array(self.best_goal_pose_path))

            for goal_node_id in self.goal_node_ids:
                if self.nodes[goal_node_id].cost < self.nodes[self.best_goal_node_id].cost:
                    print("BETTER SOLUTION FOUND")
                    self.best_goal_node_id = goal_node_id
                    if visualize:
                        for i in range(self.best_goal_pose_traj.shape[0]):
                            self.window.remove_point(self.best_goal_pose_traj[i, 0:2], radius=5, update=False)
                        self.window.update()
                    self.best_goal_pose_path, self.best_goal_pose_traj = self.recover_path(
                        self.best_goal_node_id, visualize=visualize
                    )
                    self.update_sampling_space()
                    if save_path:
                        np.save("rrt_star_path.npy", np.array(self.best_goal_pose_path))  

        return self.nodes
    

    def update_sampling_space(self):
        # take self.best_goal_pose_traj and pass it through cost to come
        trajectory_length = self.cost_to_come(self.best_goal_pose_traj)
        a = trajectory_length / 2

        center = (self.best_goal_pose_traj[0, 0:2] + self.best_goal_pose_traj[-1, 0:2]) / 2
        self.sampling_space_params["a"] = a
        self.sampling_space_params["center"] = center
        self.sampling_space_params["goal_range_multiplier"] = 1


    def recover_path(self, node_id=-1, visualize=0):
        path = [self.nodes[node_id].pose]
        node_id_path = [node_id]

        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].pose)
            node_id_path.append(current_node_id)
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()
        node_id_path.reverse()

        trajectory = np.empty((0, 3))
        for i in range(len(node_id_path)-1):
            curr_node_id = node_id_path[i]
            next_node_id = node_id_path[i+1]
            trajectory = np.vstack((trajectory, self.nodes[curr_node_id].pose_traj_to_children[next_node_id]))
        
        if visualize:
            for i in range(trajectory.shape[0]):
                self.window.add_point(trajectory[i, 0:2], radius=5, color=(0, 255, 0), update=False)
            self.window.update()
        return path, trajectory


def main():
    #Set map information
    map_filename = "willowgarageworld_05res.png"
    map_setings_filename = "willowgarageworld_05res.yaml"

    #robot information
    goal_point = np.array([10.0, 10.0]) #m
    # goal_point = np.array([-10.0, 0.0]) #m

    stopping_dist = 0.5 #m

    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)
    # nodes = path_planner.rrt_planning()
    nodes = path_planner.rrt_star_planning()
    
    input("")


if __name__ == '__main__':
    main()
