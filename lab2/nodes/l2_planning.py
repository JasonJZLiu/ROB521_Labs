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
        self.goal_node_id = -1

        #Trajectory Simulation Parameters
        self.timestep = 1.0 #s
        self.num_substeps = 10 #10

        #Planning storage
        self.nodes = [Node(np.zeros((3,)), -1, 0)]

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5
        
        #Pygame window for visualization
        # (1000, 1000)
        # self.window = pygame_utils.PygameWindow(
        #     "Path Planner", (2500, 2500), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        self.window = pygame_utils.PygameWindow(
            "Path Planner", (3000, 3000), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        return



    #Functions required for RRT
    def sample_map_space(self, visualize=0):
        #Return an [x,y] coordinate to drive the robot towards (1, 2)

        sample_goal_prob = np.random.rand()
        if sample_goal_prob < 0.06:
            # samples the goal position
            sample = self.goal_point
        else:
            sample = (self.bounds[1] - self.bounds[0])*np.random.rand(2) + self.bounds[0]
            sample = sample.reshape(2,)
        
        if visualize != 0:
            self.window.add_point(sample, radius=5, color=(255, 0, 0), update=True)
        
        return sample
          
    
    def check_if_duplicate(self, point):
        #Check if point is a duplicate of an already existing node
        # point: (3,)

        # constructing a KDTree with the full pose (x, y, theta)
        nodes_KDTree = cKDTree([node.pose for node in self.nodes])
        distance, node_idx = nodes_KDTree.query(point)
        if np.isclose(distance, 0.0):
            return True
        else:
            return False

    
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

        vel_grid_res = 4
        rot_vel_res = 9 #7
        num_vel = vel_grid_res*rot_vel_res

        vel_range = np.linspace(-self.vel_max, self.vel_max, vel_grid_res)
        rot_vel_range = np.linspace(-self.rot_vel_max, self.rot_vel_max, rot_vel_res)
        vel_grid, rot_vel_grid = np.meshgrid(vel_range, rot_vel_range, indexing='ij')
        # (num_vel, 1)
        vel_candidates = vel_grid.flatten().reshape(-1, 1)
        rot_vel_candidates = rot_vel_grid.flatten().reshape(-1, 1)
        # print(np.hstack([vel_candidates, rot_vel_candidates]))

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
    #Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function


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
        print("TO DO: Implement a cost to come metric")
        return 0
    
    def update_children(self, node_id):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        print("TO DO: Update the costs of connected nodes after rewiring.")
        return

    #Planner Functions
    def rrt_planning(self):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        while True: #Most likely need more iterations than this to complete the map!
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
            
            # Check if goal has been reached
            if np.linalg.norm(new_node_pose[0:2] - self.goal_point) <= self.stopping_dist:
                # self.goal_nodes[len(self.nodes) - 1] = self.nodes[-1]
                # self.best_goal_node_id = len(self.nodes) - 1
                self.goal_node_id = new_node_id
                break
        return self.nodes
    
    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot        
        for i in range(1): #Most likely need more iterations than this to complete the map!
            #Sample
            point = self.sample_map_space()

            #Closest Node
            closest_node_id = self.closest_node(point)

            #Simulate trajectory
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].pose, point)

            #Check for Collision
            print("TO DO: Check for collision.")

            #Last node rewire
            print("TO DO: Last node rewiring")

            #Close node rewire
            print("TO DO: Near point rewiring")

            #Check for early end
            print("TO DO: Check for early end")
        return self.nodes
    
    def recover_path(self, node_id=-1, visualize=0):
        path = [self.nodes[node_id].pose]
        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].pose)
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()

        if visualize:
            for pose in path:
                self.window.add_point(pose[0:2], radius=5, color=(0, 255, 0), update=True)
                self.window.add_se2_pose(pose, length=5, color=(0, 255, 0), update=True)
        return path


def main():
    #Set map information
    map_filename = "willowgarageworld_05res.png"
    map_setings_filename = "willowgarageworld_05res.yaml"

    #robot information
    goal_point = np.array([10.0, 10.0]) #m
    # goal_point = np.array([20.0, -30.0]) #m

    stopping_dist = 0.5 #m

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)

    # # nodes = path_planner.rrt_star_planning()
    # nodes = path_planner.rrt_planning()
    # node_path_metric = np.array(path_planner.recover_path(visualize=1))

    # #Leftover test functions
    # np.save("rrt_path.npy", node_path_metric)
    # input("")


    # TEST
    # from PIL import Image
    # map_image = Image.open("../maps/"+map_filename).convert('RGB')
    # map_pixels = map_image.load()

    # points = np.array([[0.0, 0.0], [300, -10.0]])

    # # (num_pts, num_pts_per_circle, 2)
    # point_circle_coords = path_planner.points_to_robot_circle(points)

    # # testing points_to_robot_circle
    # for i in range(point_circle_coords.shape[0]):
    #     for j in range(point_circle_coords.shape[1]):
    #         px_coor = point_circle_coords[i, j]
    #         px_coor = (px_coor[0], px_coor[1])
    #         map_pixels[px_coor] = (255, 0, 0)

    
    # # testing trajectory_rollout
    # vel = np.array([1.0, 1, 1]).reshape(-1, 1)
    # rot_vel = np.array([1.5, 0.0, -1.5]).reshape(-1, 1)
    # x_y_theta = np.array([0.0, 0.0, 0.])#.reshape(3, 1)
    # res, ind, last_points = path_planner.trajectory_rollout(
    #     vel, rot_vel, x_y_theta, 1*2, 10*2
    # )
    # for i in range(res.shape[0]):
    #     res_1 = res[i, :, 0:2]
    #     point_circle_coords = path_planner.point_to_cell(res_1)
    #     for j in range(point_circle_coords.shape[0]):
    #         px_coor = point_circle_coords[j]
    #         px_coor = (px_coor[0], px_coor[1])
    #         map_pixels[px_coor] = (255, 0, 0)

    # last_points_px = path_planner.point_to_cell(last_points)
    # for i in range(last_points.shape[0]):
    #     px_coor = last_points_px[i]
    #     point_circle_coords = path_planner.point_to_cell(res_1)
    #     px_coor = (px_coor[0], px_coor[1])
    #     map_pixels[px_coor] = (0, 255, 0)
    # map_image.show()

    # # testing robot_controller
    # sampled_point = np.array([2, 0.6])
    # path_planner.window.add_point(sampled_point, radius=3)
    # node_0 = Node(pose=np.array([0, 0, 0.5]), parent_id=-1, cost=0)
    # path_planner.window.add_se2_pose(node_0.pose, length=10)
    # path_planner.robot_controller(node_0, sampled_point, visualize=2)

    # # test rrt_plan
    # path_planner.rrt_planning()

    # path = path_planner.recover_path(path_planner.goal_node_id, visualize=1)
    # input("")
    
    # # test connect_node_to_point
    # node_i = Node(
    #     pose=np.array([10, 10, 0.1]),
    #     parent_id=-1,
    #     cost=0,
    # )

    # # point_f = np.array([11, 12.])
    # # point_f = np.array([9, 12.])
    # # point_f = np.array([11, 8])
    # # point_f = np.array([9, 8])

    # point_f = np.array([10, 5.])
    # path_planner.window.add_point(point_f, radius=5, color=(0, 255, 0))
    # path_planner.connect_node_to_point(node_i, point_f, visualize=1)
    # input("")

    



if __name__ == '__main__':
    main()
