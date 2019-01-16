"""
env_models.py

Contains environmental models for testing with drones from drones.py in 
MinimizingUncertainty project:
    
    Model_StaticFire: static fire model on flat terrain

"""

import time

import math
import logging
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, LinearRing, LineString, Point, MultiPoint
from shapely.ops import unary_union
from matplotlib.path import Path
import csv

import matplotlib.cm as colors
import matplotlib.pyplot as plt


####################
"""
Helper functions
"""
def dist_l1(x, y):
    ind_end = 3
    if (len(x) < 3 or len(y) < 3):
        ind_end = 2
    return np.sum(np.abs(x[0:ind_end] - y[0:ind_end]))


def dist_l2(x, y):
    ind_end = 3
    if (len(x) < 3 or len(y) < 3):
        ind_end = 2
    return np.sqrt(np.sum(np.power(x[0:ind_end] - y[0:ind_end], 2)))


####################
class Model_Base():
    """
    Base class for environmental models that holds all basic functionality
    """
    
    def __init__(self, env_size=[500,500], step_size=1, step_time=1,  
                 b_verbose=True, b_logging=True):
        """
        Initializing the base for the models requires at a minimum:
            step_size: physical sizing scale
            step_time: time that the environmental model will be pictured at
        """
        self.b_verbose = b_verbose
        self.b_logging = b_logging
        self.step_size = step_size
        self.step_time = step_time
        
        self.logger = logging.getLogger('root')
        
        # Create a default circle
        self.init_gamma()
        self.init_poi()
        
        self.map_terrain = None
    
    
    def init_gamma(self):
        """
        Initialize the path
        """
        # TO BE UPDATED FOR SPECIFIC MODELS
        theta = np.linspace(0, 2*math.pi, 17)
        r = 50
        x = r*np.sin(theta).reshape((-1,1))
        y = -r*np.cos(theta).reshape((-1,1))
        self.gamma = np.append(x, y, axis=1)
        
        
    def init_poi(self):
        """
        Initialize the points of interest x and associated covariance of growth
        """
        # TO BE UPDATED FOR SPECIFIC MODELS
        self.list_x = self.decimate_line(self.gamma, dec_factor=20)
        theta = np.linspace(0, 2*math.pi, len(self.x) + 1)
        self.covar_env = np.diag(5 + 5*np.cos(theta[:-1]))
    
    
    def decimate_line(self, line_in, dec_factor=10):
        """
        Simplify a line by selecting points at a set distance provided by 
        the dec_factor using linear interpolation along exterior edge
        """
        list_coords = np.array(line_in)
        list_x_new = list_coords[0].reshape((1,-1))
        x_prev = None
        dist = 0
        
        for x_curr in list_coords:
            if (x_prev is None):
                x_prev = x_curr
                continue
            
            dist_seg = dist_l2(x_curr, x_prev)

            while ((dist + dist_seg) > dec_factor):
                dist_frac = (dec_factor - dist)/dist_seg
                x_new = ((1 - dist_frac)*x_prev + dist_frac*x_curr).reshape((1,-1))
                list_x_new = np.append(list_x_new, x_new, axis=0)
                dist -= dec_factor
            dist += dist_seg
            x_prev = x_curr
                
        return list_x_new
    
    
    def set_terrain_map(self, map_in):
        """
        Provides an object reference to a terrain map for use in environmental
        models
        """
        self.map_terrain = map_in
        
    
    def set_steps(self, step_size, step_time):
        """
        Sets the step size for space and time
        """
        self.step_size = step_size
        self.step_time = step_time
        
        
    def get_gamma(self):
        """
        Returns path through environment
        """
        return self.gamma
    
    
    def get_poi(self):
        """
        Returns underlying state positions
        """
        return self.list_x
    
    
    def get_covar_env(self):
        """
        Returns covariance matrix of Wiener process
        """
        return self.covar_env
    
    
    def update(self):
        """
        Update model for changes over 1 time step
        """
        # REPLACE WITH REQUIRED CODE IF NONSTATIC MODEL
        pass
    
    
    def visualize(self):
        """
        Visualizes the path (gamma) and the points of interest (list_x)
        """
        ax = plt.gca()
        
        if not(self.gamma is None):
            # gamma_x, gamma_y = self.gamma.xy # for self.gamma as LinearRing
            gamma_x = self.gamma[:,0]
            gamma_y = self.gamma[:,1]
            ax.plot(gamma_x, gamma_y, color='red', zorder=2)
            
        # cmap_fire = colors.get_cmap('hot')
        # cmap_fire.set_under('k', alpha=0)
        # 
        # plt.pcolor(self.list_x[:, 0], self.list_x[:, 1], 
        #            np.diag(self.covar_env), vmin=0.01, vmax=1.5, 
        #            cmap=cmap_fire, zorder=1)
        # plt.colorbar()
        
        ax.scatter(self.list_x[:, 0], self.list_x[:, 1], color='black', marker='x')


####################
class Model_StaticFire(Model_Base):
    """
    Model of a static fire border with growing uncertainty related to position.
    The position was calculated using the elliptical growth model of the 
    FARSITE model and the uncertainty was calculated as 0.35*(change in 
    position over one second in normal direction to the ellipse)
    """
    
    def __init__(self, env_size=[500,500], step_size=1, step_time=1, 
                 param_ros=10, b_verbose=True, b_logging=True):
        """
        Initializing the static fire model
        
        Additional parameters to base model:
            param_ros: scale factor for env covariance matrix
        """
        self.b_logging = b_logging
        self.b_verbose = b_verbose
        self.param_ros = param_ros
        
        super(Model_StaticFire, self).__init__(env_size=env_size, 
             step_size=step_size, step_time=step_time, b_verbose=b_verbose, 
             b_logging=b_logging)
    
    
    def init_gamma(self):
        """
        Initialize the path from a csv located in directory
        """
        temp = []
        # with open('fire_data_120_121.csv') as fid:
        with open('fire_data_240_241.csv') as fid:
            csvreader = csv.reader(fid, delimiter=',')
            for ind,row in enumerate(csvreader):
                temp.append(row)
        
        self.fire_poly_0 = LinearRing(np.array([temp[0], temp[1]]).T)
        self.fire_poly_1 = LinearRing(np.array([temp[2], temp[3]]).T)
        
        self.gamma = np.array(self.fire_poly_0)
        
        
    def init_poi(self):
        """
        Initialize the points of interest x and associated covariance of growth
        """
        self.list_x = self.decimate_line(self.gamma, dec_factor=50)
        self.list_xnorm = self.calc_xnorm()
        
        # Using normal vectors, determine growth of fire over 1 second and use
        # as a scale factor for Wiener process model of fire
        scale_noise_env = 10
        self.covar_env = np.array(scale_noise_env*np.eye(len(self.list_x)))
    
        # Predicted growth determined by examining intersection of normal 
        # vectors with ring growth over 1 second
        ros_var_factor = 0.35 * self.param_ros # 35% uncertainty proposed by 
                                               # Cruz and Alexander (2013)
        for x, xnorm, xconf, ind in zip(self.list_x, self.list_xnorm, 
                                        np.diag(self.covar_env), range(len(self.list_x))):

            line_xnorm = LineString(np.append(x, x + xconf*xnorm).reshape((-1, 2)))
            
            point_x = line_xnorm.intersection(self.fire_poly_1)
            ros = -1
            if (isinstance(point_x, Point)):
                # If one intersection, determine distance
                x_new = np.array(point_x.coords)
                ros = dist_l2(x_new, x)
                
            elif (isinstance(point_x, MultiPoint)):
                # If multiple points, save distance closest to expected
                ind_min = 0
                ros = dist_l2(np.array(point_x[0].coords), x)
                for ind_x, point_x_ in enumerate(point_x):
                    temp_ros = dist_l2(np.array(point_x_.coords), x)
                    if (temp_ros < ros):
                        ind_min = ind_x
                        ros = temp_ros
                x_new = np.array(point_x[ind_min].coords)
            
            if (ros >= 0):
                self.covar_env[ind, ind] = ros*ros_var_factor
    
    
    def calc_xnorm(self, list_x=None):
        """
        Update normal vectors for each target where normal is outward from a
        CCW polygon 
        """        
        if (list_x is None):
            list_x = self.list_x
            
        list_xnorm = np.zeros(list_x.shape)
        if (len(list_x) > 3):
            mat_norm = np.matrix([[0,1],[-1,0]])
            
            # First
            vec_temp = list_x[-1] - list_x[1]
            vec_temp = vec_temp / np.sqrt(np.sum(np.power(vec_temp, 2)))
            list_xnorm[0] =  vec_temp*mat_norm
            
            # Middle
            for ind in range(1, len(list_x)-1):
                vec_temp = list_x[ind-1] - list_x[ind+1]
                vec_temp = vec_temp / np.sqrt(np.sum(np.power(vec_temp, 2)))
                list_xnorm[ind] =  vec_temp*mat_norm
                
            # Last
            vec_temp = list_x[-2] - list_x[0]
            vec_temp = vec_temp / np.sqrt(np.sum(np.power(vec_temp, 2)))
            list_xnorm[-1] =  vec_temp*mat_norm
            
        elif (len(list_x) > 0):
            for ind in range(len(list_x)):
                list_xnorm[ind] = np.matrix([[1,0]]) * (-1)**ind
                
        return list_xnorm
    

####################
class Model_StaticCircle_set(Model_Base):
    """
    Base model that uses a static circle and provided POIs
    """
    def __init__(self, list_theta, list_p, env_size=[500,500], step_size=1, 
                 step_time=1, param_ros=1, path_length=500,
                 b_verbose=True, b_logging=True):
        """
        Initializing a model with a static circular path with random radius,
        random POI positions, and random growth rates
        
        Additional parameters to base model:
            param_ros: scale factor for env covariance matrix
            path_length: approximate length of path around the circle
            env_size: total size of the environment
        """
        self.b_logging = b_logging
        self.b_verbose = b_verbose
        self.L = path_length
        self.list_theta = np.array(list_theta)
        self.list_p = np.array(list_p)
        
        self.N_q = len(list_theta)
        self.offset = np.array(env_size)/2
        
        self.gamma_map = None
        
        super(Model_StaticCircle_set, self).__init__(env_size=env_size, 
             step_size=step_size, step_time=step_time, b_verbose=b_verbose, 
             b_logging=b_logging)
    
    
    def init_gamma(self):
        """
        Initialize the path as a circle in the middle of the environment
        """
        rad = self.L / (2*math.pi)
        self.gamma = np.array(Point(np.array(self.offset)).buffer(rad).exterior)
        
        
    def init_poi(self):
        """
        Initialize the points of interest x and associated covariance of growth
        using random values
        """  
        self.list_x = np.zeros((self.N_q, 2))
        for ind, theta in enumerate(self.list_theta):
            self.list_x[ind, :] = self.calc_pos(theta)
            
        self.covar_env = np.diag(self.list_p)
        
    
    def calc_pos(self, theta):
        """
        Calculates position along closed path
        """
        # If look up table hasn't been generated yet, then generate
        if (self.gamma_map is None):
            temp_gamma = np.array(self.gamma)
            gamma_dist = np.sqrt(np.sum(
                    np.power(temp_gamma - np.append([temp_gamma[-1,:]], 
                                                   temp_gamma[:-1,:], axis=0), 2), 
                                 axis=1))
            self.gamma_map = np.append(np.cumsum(gamma_dist).reshape((-1,1)), 
                                       temp_gamma, axis=1)
            
        # Using look up table, find points that 
        ind_1 = self.gamma_map.shape[0] - 1 - np.argmax(self.gamma_map[-1::-1,0] <= theta)
        ind_2 = (ind_1 + 1) % self.gamma_map.shape[0]
        g_1 = self.gamma_map[ind_1, 1:]
        g_2 = self.gamma_map[ind_2, 1:]
        while all(g_1 == g_2):
            ind_2 = (ind_2 + 1) % self.gamma_map.shape[0]
            g_2 = self.gamma_map[ind_2, 1:]
        theta_frac = (theta - self.gamma_map[ind_1, 0]) / dist_l2(g_1, g_2)
        
        pos = (1 - theta_frac)*g_1 + (theta_frac)*g_2
        
        return pos
        
    
    def get_config(self):
        """
        Returns configuration of environment in a list
        """
        list_config = [self.L, self.offset.flatten(), self.list_theta.flatten(), np.diag(self.covar_env)]
        
        return list_config
    
    
####################
class Model_StaticCircle_Random(Model_Base):
    """
    
    """
    def __init__(self, env_size=[500,500], step_size=1, step_time=1, 
                 param_ros=1, path_length=500, N_q=5, dtheta_min=30,
                 b_verbose=True, b_logging=True, b_overlap=False):
        """
        Initializing a model with a static circular path with random radius,
        random POI positions, and random growth rates
        
        Additional parameters to base model:
            param_ros: scale factor for env covariance matrix
            path_length: approximate length of path around the circle
            env_size: total size of the environment
        """
        self.b_logging = b_logging
        self.b_verbose = b_verbose
        self.L = path_length
        self.N_q = int(N_q)
        self.offset = np.array(env_size)/2
        self.param_ros = param_ros
        self.dtheta_min = dtheta_min
        self.b_overlap = b_overlap
        
        self.gamma_map = None
        
        super(Model_StaticCircle_Random, self).__init__(env_size=env_size, 
             step_size=step_size, step_time=step_time, b_verbose=b_verbose, 
             b_logging=b_logging)
    
    
    def init_gamma(self):
        """
        Initialize the path as a circle in the middle of the environment
        """
        rad = self.L / (2*math.pi)
        self.gamma = np.array(Point(np.array(self.offset)).buffer(rad).exterior)
        
        
    def init_poi(self):
        """
        Initialize the points of interest x and associated covariance of growth
        using random values
        """
        if not(self.b_overlap):
            # If no overlap, ensure sufficient spacing (2*dtheta_min) between
            # points and that number of points is valid
            if (self.N_q >= self.L/(2*self.dtheta_min)):
                self.N_q = np.floor(self.L/(2*self.dtheta_min)).astype(int)
            list_frac = np.random.random([self.N_q, 1])
            list_frac = (self.L - self.N_q*2*self.dtheta_min) * list_frac / np.sum(list_frac)
            self.list_theta = np.cumsum(2*self.dtheta_min*np.ones((self.N_q, 1))) + np.cumsum(list_frac) - list_frac[0]/2
        else:
            # Else enforce at least one overlap, defined as two points within
            # dtheta_min
            list_frac = np.random.random([self.N_q, 1])
            list_frac = (self.L - 2*self.dtheta_min) * list_frac / np.sum(list_frac)
            while not(any(list_frac < self.dtheta_min)):
                list_frac = np.random.random([self.N_q, 1])
                list_frac = (self.L - 2*self.dtheta_min) * list_frac / np.sum(list_frac)
            self.list_theta = np.cumsum(list_frac) + self.dtheta_min - list_frac[0]/2
        
        self.list_x = np.zeros((self.N_q, 2))
        for ind, theta in enumerate(self.list_theta):
            self.list_x[ind, :] = self.calc_pos(theta)
            
        self.covar_env = np.diag(self.param_ros * np.random.random(self.N_q))
        
    
    def calc_pos(self, theta):
        """
        Calculates position along closed path
        """
        # If look up table hasn't been generated yet, then generate
        if (self.gamma_map is None):
            temp_gamma = np.array(self.gamma)
            gamma_dist = np.sqrt(np.sum(
                    np.power(temp_gamma - np.append([temp_gamma[-1,:]], 
                                                   temp_gamma[:-1,:], axis=0), 2), 
                                 axis=1))
            self.gamma_map = np.append(np.cumsum(gamma_dist).reshape((-1,1)), 
                                       temp_gamma, axis=1)
            
        # Using look up table, find points that 
        ind_1 = self.gamma_map.shape[0] - 1 - np.argmax(self.gamma_map[-1::-1,0] <= theta)
        ind_2 = (ind_1 + 1) % self.gamma_map.shape[0]
        g_1 = self.gamma_map[ind_1, 1:]
        g_2 = self.gamma_map[ind_2, 1:]
        while all(g_1 == g_2):
            ind_2 = (ind_2 + 1) % self.gamma_map.shape[0]
            g_2 = self.gamma_map[ind_2, 1:]
        theta_frac = (theta - self.gamma_map[ind_1, 0]) / dist_l2(g_1, g_2)
        
        pos = (1 - theta_frac)*g_1 + (theta_frac)*g_2
        
        return pos
        
    
    def get_config(self):
        """
        Returns configuration of environment in a list
        """
        list_config = [self.L, self.offset.flatten(), self.list_theta.flatten(), np.diag(self.covar_env)]
        
        return list_config