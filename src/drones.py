"""
drones.py
Michael Ostertag

Classes for drones under test
    1. Drone_Constant
    2. Drone_Smith
    3. Drone_Ostertag
"""

import time

import math
import logging
import numpy as np
from numpy import linalg
from shapely.geometry import Polygon, LineString, LinearRing, MultiPoint, Point, MultiLineString
from shapely.ops import unary_union, split, cascaded_union, snap, linemerge
from shapely.affinity import translate
import shapely
from matplotlib.path import Path
import csv

# SymPy used for generating functions to solve for Ostertag Greedy KF 
# implementation
import sympy

# PuLP is a linear programming optimization library
import pulp

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


def generate_func_kfsteadystate(depth=10):
    """
    Creates the functions to solve the steady state Kalman Filter equations 
    around a loop with multiple consecutive observations of a point
    """
    V = sympy.symbols('V')
    Twc = sympy.symbols('Twc')
    W = sympy.symbols('W')
    list_lambda = sympy.symbols(['lambda_T{0}'.format(i) for i in range(depth)])
    list_func = []
    
    for num_obs in range(1, depth+1):
        # time_start = time.time()
        # print('Depth {0}'.format(num_obs))
        if (num_obs > 1):
            expr = sympy.Eq(list_lambda[0], list_lambda[num_obs-1] + Twc*W - \
                            list_lambda[num_obs-1]**2 / (list_lambda[num_obs-1] + V))
            for k in range(num_obs-1, 0, -1):
                expr = expr.subs(list_lambda[k], 
                                 list_lambda[k-1] + W - list_lambda[k-1]**2 / \
                                 (list_lambda[k-1] + V))
        else:
           expr = sympy.Eq(list_lambda[0], list_lambda[0] + Twc*W - \
                           list_lambda[0]**2 / (list_lambda[0] + V))

        list_func.append(expr)
        # print('{0:0.3f} s'.format(time.time() - time_start))
    
    return list_func


####################
class Drone_Base():
    """
    Base class for drones that holds all basic functionality
    """
    
    def __init__(self, loop_path, poi, covar_e, obs_window, covar_obs, 
                 drone_id=0, theta0=0, fs=1, covar_s_0_scale=100, 
                 v_max=20, v_min=0, b_verbose=True, b_logging=True):
        """
        Initializing the drone requires at minimum:
            loop_path: a path to follow 
            poi: a list of points of interest 
            covar_e: covariance matrix for environment noise
            obs_window: an observation window
            covar_obs: covariance matrix for observation noise
        
        and optionally:
            drone_id: unique number representing a drone
            theta0: initial position around loop_path
            fs: sampling frequency
            covar_s_0_scale: initial covariance of each poi
            v_max: maximum velocity
            v_min: minimum velocity
        """
        self.b_verbose = b_verbose
        self.b_logging = b_logging
        
        self.drone_id = drone_id
        self.fs = fs                    # Sampling frequency
        
        # Path to follow can take multiple forms (Polygon, LinearRing, or 
        # numpy array). Convert it to a LinearRing
        if isinstance(loop_path, Polygon):
            self.gamma = loop_path.exterior    
        elif isinstance(loop_path, LinearRing):
            self.gamma = loop_path
        elif isinstance(loop_path, np.ndarray):
            self.gamma = LinearRing(loop_path)
        self.L = self.calc_path_length(np.array(self.gamma))
        self.gamma_map = None           # Look up table to speed up position determination
        self.theta = theta0 % self.L    # Current drone position
        self.dtheta = 0                 # Movement of drone
        self.pos = self.calc_pos(self.theta)
        
        self.list_q = poi               # Points of interest
        self.N_q = self.list_q.shape[0] # Number of POI
        self.list_s = np.zeros((self.N_q, 1)) # Value of points of interest, s estimates x
        self.covar_e = covar_e          # Covariance matrix of Wiener process noise
        self.covar_s = np.zeros((self.N_q, self.N_q))   # Covariance of estimate s
        self.init_covar(covar_s_0_scale)# Initialize self.covar_s
        
        # Observation window can take multiple forms (Polygon, LinearRing, 
        # numpy array). Convert it to a Polygon.
        if isinstance(obs_window, Polygon):
            self.B = obs_window  
        elif isinstance(obs_window, LinearRing):
            self.B = Polygon(obs_window)
        elif isinstance(obs_window, np.ndarray):
            self.B = Polygon(obs_window)            
        self.covar_obs = covar_obs      # Variance of observation noise TODO change to function call to env_model
        self.list_y = np.zeros((0,1))   # Observations y = Hx + noise
        self.M_obs = 0                  # Number of observations
        self.H = np.zeros((0, self.N_q))# Observation matrix 
    
        self.v_max = v_max
        self.v_min = v_min
        
        
        self.logger = logging.getLogger('root')
        
        if (b_verbose):
            print('Drone {0} as '.format(self.drone_id, self.__class__.__name__))
            print('  Initialized')
            print('  theta: {0:0.2f} / {0:0.2f}'.format(self.theta, self.L))
            print('  pos: ({0:0.2f}, {1:0.2f})'.format(self.pos[0], self.pos[1]))
        
        if (self.b_logging):
            self.logger.debug('Drone {0} as {1}'.format(self.drone_id, self.__class__.__name__))
            self.logger.debug('gamma = [' + '; '.join([np.array2string(np.array(q).flatten(), separator=',', formatter={'float_kind':lambda x: '{0:5.1f}'.format(x)}) for q in np.array(self.gamma)]) + ']')
            self.logger.debug('L = {0:0.2f}'.format(self.L))
            self.log_iter()

        
    def init_covar(self, covar_scale=100):
        """
        Initialize covariance matrix for s, the estimate of x
        """
        self.covar_s = self.covar_e * covar_scale
    
    
    def log_iter(self):
        """
        Log relevant information for current iteration:
            dtheta
            theta
            pos
            covar_s
        """
        if (self.b_logging):
            self.logger.debug('Drone {0}'.format(self.drone_id))
            self.logger.debug('dtheta = {0:0.2f}'.format(self.dtheta))
            self.logger.debug('theta = {0:0.2f}'.format(self.theta))
            self.logger.debug('pos = ({0:0.2f}, {1:0.2f})'.format(self.pos[0], self.pos[1]))
            # Create long single line of diagonal elements of covariance matrix
            self.logger.debug('covar_s  = ' + np.array2string(np.diag(self.covar_s), separator=', ', formatter={'float_kind':lambda x: '{0:6.3f}'.format(x)}, max_line_width=2148))
            if (self.H.shape[0] > 0):
                self.logger.debug('H = ' + np.array2string(self.H.flatten(), separator=', ', formatter={'int_kind':lambda x: '{0:1d}'.format(x)}, max_line_width=2148))
            else:
                self.logger.debug('H = ' + np.array2string(np.zeros((self.N_q,1)).flatten(), separator=', ', formatter={'int_kind':lambda x: '{0:1d}'.format(x)}, max_line_width=2148))
        
        if (self.b_verbose):
            print('Drone {0}'.format(self.drone_id))
            print('  dtheta = {0:0.2f}'.format(self.dtheta))
            print('  theta = {0:0.2f}'.format(self.theta))
            print('  pos = ({0:0.2f}, {1:0.2f})'.format(self.pos[0], self.pos[1]))


    def set_steps(self, step_size=None, step_time=None):
        """
        Set sampling frequency (step_time) and spatial scaling (step_size)
        """
        if (step_size):
            self.step_size = step_size
            
        if (step_time):
            self.fs = step_time
    
    def set_theta(self, theta=0):
        """
        Set theta and position of drone
        """
        self.theta = theta % self.L
        self.pos = self.calc_pos(theta)
    
    
    def calc_path_length(self, loop_path):
        """
        Calculate path length assuming linear interpolation between points and
        each row is a new point
        """
        L = np.sum(np.sqrt(np.sum(np.power(loop_path[1:,:] - loop_path[:-1,:], 2), axis=1)))
        return L
    
    
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
    
    
    def calc_movement(self):
        """
        Calculates optimal velocity to move along path
        """
        # TO BE REPLACED WITH MOVEMENT ALGORITHM
        v = max(self.v_min, min(self.v_max, 0))
        self.dtheta = v/self.fs
    
        
    def update(self):
        """
        Moves the drone based on the velocity calculated by calc_movement(). 
        Then, captures an observation and updates the Kalman filter.
        """
        # Update current position
        self.theta = (self.theta + self.dtheta) % self.L
        self.pos = self.calc_pos(self.theta)
        
        # Update current prediction of estimated states
        self.update_predict()
        
        # Capture an observation and update Kalman filter
        self.capture_data()
        self.update_kalman()
        
        # Log information
        self.log_iter()
    
    
    def capture_data(self):
        """
        Captures data for any points within observation window B(theta) where 
        B(theta) = self.pos + self.B
        
        The observation matrix H is saved where y = Hx + noise
        """
        list_ind_detected = []
        B_theta = translate(self.B, xoff=self.pos[0], yoff=self.pos[1])
        
        for ind_q, point in enumerate(MultiPoint(self.list_q)):
            if (B_theta.contains(point)):
                list_ind_detected.append(ind_q)
        
        self.M_obs = len(list_ind_detected)
        
        # Reset observation matrix to correct size
        self.H = np.zeros((self.M_obs, self.N_q))
        
        for ind_row, ind_detected in enumerate(list_ind_detected):
            self.H[ind_row, ind_detected] = 1
        
        # Create observations
        # To be updated with appropriate noise models
        self.y = np.zeros((self.M_obs, 1))
        self.covar_V = self.covar_obs*np.identity(self.M_obs)
    
    
    def update_predict(self):
        """
        Updates prediction of estimate using knowledge of environmental model
        """
        self.list_s = self.list_s + np.zeros(self.list_s.shape)
        self.covar_s = self.covar_s + self.covar_e
        

    def update_kalman(self):
        """
        Updates covariance of s using noise levels
        """
        # Calculate Kalman gain
        K = linalg.solve(np.matmul(np.matmul(self.H,self.covar_s), self.H.T) + self.covar_V, 
                         np.matmul(self.H, self.covar_s)).T
        
        # Update estimates and covariances
        self.list_s = self.list_s + np.matmul(K, self.y - np.matmul(self.H, self.list_s))
        self.covar_s = np.matmul(np.identity(self.covar_s.shape[0]) - np.matmul(K, self.H), self.covar_s)
        
        
    def get_covar_s(self):
        """
        Returns the current covariance matrix for the estimate of s immediately
        preceeding the next point
        """
        return self.covar_s + self.covar_e/self.fs
    
            
    def visualize(self, plot_color):
        """
        Plots the drone position and its viewing window on previously selected
        figure
        """
        # Plot position
        plt.scatter(self.pos[0], self.pos[1], color=plot_color, marker='x')
        
        # Plot sensing region
        temp_B = np.array(self.B.exterior) + self.pos
        B_x = temp_B[:,0]
        B_y = temp_B[:,1]
        plt.plot(B_x, B_y, color=plot_color)
    
    
    def get_optimalbound(self):
        """ 
        Returns optimal bound for the chosen velocity controller or -1 if no
        bound exists. To be updated and implemented by any controller with
        an optimal bound
        """
        return -1


####################
class Drone_Constant(Drone_Base):
    """
    Drone that flies at a constant speed.
    """
    
    def __init__(self, loop_path, poi, covar_e, obs_window, covar_obs, 
                 drone_id=0, theta0=0, fs=1, v_max=20, v_min=0, v_const=10, 
                 covar_s_0_scale=100, b_verbose=True, b_logging=True):
        """
        Initialze a drone that flies at a constant velocity around the path
        
        New variables:
            v_const: constant velocity that the drone flies at
        """
        super(Drone_Constant, self).__init__(loop_path, poi, covar_e, 
             obs_window, covar_obs, drone_id=drone_id, theta0=theta0, fs=fs, 
             v_max=v_max, v_min=v_min, covar_s_0_scale=covar_s_0_scale,
             b_logging=b_logging, b_verbose=b_verbose)
        
        self.v_const = v_const
        
    
    def calc_movement(self):
        """
        Velocity is given as a constant value
        """
        v = max(self.v_min, min(self.v_max, self.v_const))
        self.dtheta = v/self.fs
    

####################
class Drone_Smith(Drone_Base):
    """
    Drone that flies with a velocity controller as proposed in Smith (2012)
        Smith SL, Schwager M, Rus D. Persistent robotic tasks: Monitoring and 
        sweeping in changing environments. IEEE Transactions on Robotics. 2012 
        Apr;28(2):410-26.
    
    New variables:
        J: number of rectangular segments to use as basis functions
    """
    
    def __init__(self, loop_path, poi, covar_e, obs_window, covar_obs, 
                 drone_id=0, theta0=0, fs=1, v_max=20, v_min=0, 
                 covar_s_0_scale=100, J=100, b_verbose=True, b_logging=True):
        """
        Initialze a drone that flies at a constant velocity around the path
        """
        super(Drone_Smith, self).__init__(loop_path, poi, covar_e, 
             obs_window, covar_obs, drone_id=drone_id, theta0=theta0, fs=fs, 
             v_max=v_max, v_min=v_min, covar_s_0_scale=covar_s_0_scale,
             b_logging=b_logging, b_verbose=b_verbose)
        
        self.J = J
        self.create_v_controller() # Creates an optimal velocity controller
        
        
    def create_v_controller(self):
        """
        Creates an optimal position-dependent velocity controller with J steps
        based on the methodology outlined in Smith (2012)
        """
        # Growth in uncertainty is due to environmental noise
        p = np.diag(self.covar_e)
        
        # Decrease in uncertainty is from Kalman filter, which can only measure
        # when q is within sensing range. The decrease at steady state can be
        # approximated as the product of amount of time for 1 cycle of the loop
        # and the growth of the point
        
        # Total amount of time that each point is observed
        dtheta_obs = np.zeros((self.N_q, 1))
        list_gamma_obs = []     # list of observation segments
        list_theta_obs = []     # list of theta for beginning and ending of observation segments
        
        for ind_q, q in enumerate(self.list_q):            
            # Create region around q in which q can be sensed
            B_q = translate(self.B, xoff=q[0], yoff=q[1])
            # Split gamma and determine which section can sense point q
            gamma_temp = LineString(self.gamma)
            gamma_split = split(gamma_temp, B_q)
            
            list_dtheta = np.array([gamma_seg.length for gamma_seg in gamma_split])
            list_theta = np.cumsum(list_dtheta)
            # list_valid method fails around start/stop of loop 
            # list_valid = np.array([B_q.contains(Point(np.array(gamma_seg)[1,:])) for gamma_seg in gamma_split])
            # Valid/invalid segments will alternate. Determine if second
            # segment is valid or not. If too large, likely invalid.
            list_valid = np.zeros(list_theta.shape).astype(bool).flatten()
            if (list_dtheta[1] > self.L/2):
                list_valid[0::2] = True
            else:
                list_valid[1::2] = True
                
            N_segs = list_valid.shape[0]
            dtheta_obs[ind_q] = np.sum(list_dtheta[list_valid])
            theta_obs = []
            gamma_obs = []
            for ind_seg, valid, theta, gamma_seg in zip(range(N_segs), list_valid, list_theta, gamma_split):
                if (valid):
                    gamma_obs.append(gamma_seg)
                    if (ind_seg == 0):
                        theta_obs.append([0, theta])
                    elif (ind_seg == (N_segs-1)):
                        theta_obs.append([list_theta[ind_seg-1], self.L])
                    else:
                        theta_obs.append([list_theta[ind_seg-1], theta])
            
            list_theta_obs.append(theta_obs)
            list_gamma_obs.append(gamma_obs)
        
        list_gamma_obs_flat = [gamma_seg for sublist in list_gamma_obs for gamma_seg in sublist]
        gamma_cumobs = MultiLineString(list_gamma_obs_flat)
        gamma_cumobs = linemerge(gamma_cumobs)
        
        # Observed path should move at either max speed or minimum required 
        # time to take one sample
        T_obs = np.max(np.append(dtheta_obs/self.v_max, np.ones((self.N_q, 1))/self.fs, axis=1), axis=1)
        
        # Unobserved path should move at top speed
        if (isinstance(gamma_cumobs, shapely.geometry.LineString)):
            L_notobs = self.L - gamma_cumobs.length
        elif (isinstance(gamma_cumobs, shapely.geometry.MultiLineString)):
            L_notobs = self.L
            for gamma_cumobs_ in gamma_cumobs:
                    L_notobs -= gamma_cumobs_.length
                    
        T_notobs = L_notobs / self.v_max
        
        # Predicted time for single loop with 1 obs per location
        T0 = np.sum(T_obs) + T_notobs
        # First-order approximation of Kalman filter decrease
        c = T0 * p
        
        # Create basis functions
        list_beta = np.linspace(0, self.L, num=(self.J+1))
        list_beta = list_beta[0:-1].reshape((-1,1))

        list_dtheta_beta = self.calc_int_beta(list_theta_obs, list_beta)
        N_segs_total = sum([len(dtheta_beta_seg)**2 for dtheta_beta_seg in list_dtheta_beta[:,0]])
        
        # Calculate K as defined in Smith (2012) Eq. (8)
        K = np.zeros((self.N_q, self.J))
        for ind_i in range(self.N_q):
            for ind_j in range(self.J):
                K[ind_i, ind_j] = np.sum(list_dtheta_beta[ind_i,ind_j]) - p[ind_i]/c[ind_i]*self.L/self.J
        
        # Calculate X as defined in Smith (2012) Eq. (19)
        list_dtheta_beta_edge2edge = self.calc_int_beta_edge2edge(list_theta_obs, list_beta)
        X = np.zeros((N_segs_total, self.J))
        ind_seg = 0
        for ind_i in range(self.N_q):
            N_segs = len(list_dtheta_beta[ind_i, ind_j])
            for k in range(N_segs):
                for b in range(N_segs):
                    for ind_j in range(self.J):
                        X_growth = p[ind_i]*list_dtheta_beta_edge2edge[ind_seg, ind_j]
                        ind_decay = (k - np.arange(b + 1)) % N_segs
                        X_decay = c[ind_i]*sum([list_dtheta_beta[ind_i, ind_j][ind] for ind in ind_decay])
                        X[ind_seg, ind_j] = X_growth - X_decay
                    ind_seg += 1
        
        # Set up optimization problem
        prob_statement = pulp.LpProblem('Smith 2012', pulp.LpMinimize)
        
        # Create variables
        list_alphavar = [pulp.LpVariable('a{0:03d}'.format(i), lowBound=1/self.v_max,
                         cat=pulp.LpContinuous) for i in range(self.J)]
        marginvar = pulp.LpVariable('B', lowBound=0, cat=pulp.LpContinuous)
        
        # Add objective statement
        prob_statement += (marginvar)
        
        # Add constraints with X
        for ind_X, X_row in enumerate(X):
            prob_statement += pulp.LpConstraint((pulp.lpDot(X_row, list_alphavar) - marginvar), 
                                        sense=pulp.constants.LpConstraintLE, 
                                        name='X const{0}'.format(ind_X), rhs=0)
        # Add constraints with K
        for ind_K, K_row in enumerate(K):
            prob_statement += pulp.LpConstraint((pulp.lpDot(K_row, list_alphavar)), 
                                        sense=pulp.constants.LpConstraintGE, 
                                        name='K const{0}'.format(ind_K), rhs=0)
        
        # prob_statement.writeLP('SmithModel.lp')
        prob_statement.solve()
        
        list_alpha = np.array([v.varValue for v in prob_statement.variables()])[1:].reshape((-1,1))
        
        self.v_controller = np.append(list_beta, list_alpha, axis=1)
        print('Smith v_controller solved ({0})'.format(pulp.LpStatus[prob_statement.status]))
        if (self.b_logging):
            self.logger.debug('v_controller solved ({0})'.format(pulp.LpStatus[prob_statement.status]))
            self.logger.debug('alpha  = ' + np.array2string(list_alpha.flatten(), separator=', ', formatter={'float_kind':lambda x: '{0:2.5f}'.format(x)}, max_line_width=2148))

    
    def calc_int_beta(self, list_theta_obs, list_beta):
        """
        Calculates the length of path along each gamma segment, denoted by 
        theta_start and theta_stop, where q_i is observable that is in 
        list_beta[j:j+1]
        """
        list_dtheta_beta = []
        
        # Iterate through all points of interest
        for ind_i, theta_obs in enumerate(list_theta_obs):
            dtheta_beta = []
            # Iterate through all betas
            beta_start = list_beta.item(0)
            for ind_j in range(self.J):                
                if (ind_j == (len(list_beta) - 1)):
                    beta_stop = self.L
                else:
                    beta_stop = list_beta.item(ind_j+1)

                # Iterate through all different observable segments of gamma
                dtheta_beta_seg = []
                for theta_obs_seg in theta_obs:
                    temp_dtheta = 0
                    
                    # theta_obs_ is a range where q_i is observable from gamma
                    theta_start = theta_obs_seg[0]
                    theta_stop = theta_obs_seg[1]
                    
                    # Covers all cases
                    if (theta_stop < theta_start):
                        if (theta_start < beta_stop):
                            temp_dtheta += min([(beta_stop - beta_start),
                                                (beta_stop - theta_start)])
                        elif (theta_stop > beta_start):
                            temp_dtheta += min([(beta_stop - beta_start),
                                                (theta_stop - beta_start)])
                    else:
                        if ((theta_start < beta_stop) and (theta_stop > beta_start)):
                            temp_dtheta += min([(theta_stop - beta_start),
                                                (beta_stop - beta_start), 
                                                (theta_stop - theta_start),
                                                (beta_stop - theta_start)])
                    dtheta_beta_seg.append(temp_dtheta)
                dtheta_beta.append(dtheta_beta_seg)
                beta_start = beta_stop
            list_dtheta_beta.append(dtheta_beta)
                
        return np.array(list_dtheta_beta)
    
    
    def calc_int_beta_edge2edge(self, list_theta_obs, list_beta):
        """
        Calculates the length of path overlapping with the basis functions 
        from the end of observable segment to the end of the next segment.
        theta_start and theta_stop represent the end of the first observable
        segment and end of the target observable segment, respectively, where 
        q_i is observable that is in list_beta[j:j+1]
        """
        list_dtheta_beta = []

        # Iterate through all points of interest and all combinations of segments
        for ind_i, theta_obs in enumerate(list_theta_obs):
            N_segs = len(theta_obs)
            # Iterate through all combinations
            dtheta_beta = []
            for k in range(N_segs):
                for b in range(N_segs):
                    ind_start = (k - b - 1) % N_segs
                    ind_stop = k
                    # theta_obs_ is a range where q_i is observable from gamma
                    theta_start = theta_obs[ind_start][1]
                    theta_stop = theta_obs[ind_stop][1]
                    
                    beta_start = list_beta.item(0)
                    # Iterate through all betas
                    dtheta_beta_seg = []
                    for ind_j in range(self.J):                
                        if (ind_j == (len(list_beta) - 1)):
                            beta_stop = self.L
                        else:
                            beta_stop = list_beta.item(ind_j+1)
                        
                        temp_dtheta = 0
                        
                        # Covers all cases
                        if (theta_stop < theta_start):
                            if (theta_start < beta_stop):
                                temp_dtheta += min([(beta_stop - beta_start),
                                                    (beta_stop - theta_start)])
                            elif (theta_stop > beta_start):
                                temp_dtheta += min([(beta_stop - beta_start),
                                                    (theta_stop - beta_start)])
                        elif (theta_stop > theta_start):
                            if ((theta_start < beta_stop) and (theta_stop > beta_start)):
                                temp_dtheta += min([(theta_stop - beta_start),
                                                    (beta_stop - beta_start), 
                                                    (theta_stop - theta_start),
                                                    (beta_stop - theta_start)])
                        else:
                            # Condition means checking beta over entire loop
                            temp_dtheta += (beta_stop - beta_start)
                            
                        dtheta_beta_seg.append(temp_dtheta)
                        beta_start = beta_stop
                    dtheta_beta.append(dtheta_beta_seg)
            list_dtheta_beta.extend(dtheta_beta)
                
        return np.array(list_dtheta_beta)
    
    
    def calc_movement(self):
        """
        Velocity depends on current position and movement that can be taken 
        until next step
        """
        self.dtheta = 0
        time_plan = 1/self.fs
        ind_1 = np.argmin(self.v_controller[:,0] < self.theta)
        ind_1 = (ind_1 - 1) % self.J
        while (time_plan > 0):
            ind_2 = (ind_1 + 1) % self.J
            v_seg = 1/self.v_controller[ind_1, 1]
            dist_seg = (self.v_controller[ind_2, 0] - (self.theta + self.dtheta)) % self.L
            t_seg = (dist_seg / v_seg) # time to reach the next beta segment
            if (t_seg >= time_plan):
                self.dtheta += time_plan*v_seg
                time_plan = 0
            else:
                self.dtheta += dist_seg
                time_plan -= t_seg
                ind_1 = ind_2


####################       
class Drone_Ostertag(Drone_Base):
    """
    Drone that flies with a velocity controller as proposed in Ostertag (2018)

    Utilizes a greedy algorithm to find the optimal velocity controller that
    meets a minimium bound of the maximum steady state Kalman filter 
    uncertainty
    
    New variables:
        J: number of rectangular segments to use as basis functions
    """
    
    def __init__(self, loop_path, poi, covar_e, obs_window, covar_obs, 
                 drone_id=0, theta0=0, fs=1, v_max=20, v_min=0, 
                 covar_s_0_scale=100, J=100, b_logging=True, b_verbose=True):
        """
        Initialze a drone that 
        """
        super(Drone_Ostertag, self).__init__(loop_path, poi, covar_e, 
             obs_window, covar_obs, drone_id=drone_id, theta0=theta0, fs=fs, 
             v_max=v_max, v_min=v_min, covar_s_0_scale=covar_s_0_scale,
             b_logging=b_logging, b_verbose=b_verbose)
        
        self.J = J
        self.create_v_controller() # Creates an optimal velocity controller
        
        
    def create_v_controller(self):
        """
        Creates an optimal position-dependent velocity controller with J steps
        based on the methodology outlined in Ostertag (2018)
        """
        # Total amount of time that each point is observed
        dtheta_obs = np.zeros((self.N_q, 1))
        list_gamma_obs = []     # list of observation segments
        list_theta_obs = []     # list of theta for beginning and ending of observation segments
        
        for ind_q, q in enumerate(self.list_q):            
            # Create region around q in which q can be sensed
            B_q = translate(self.B, xoff=q[0], yoff=q[1])
            # Split gamma and determine which section can sense point q
            gamma_temp = LineString(self.gamma)
            gamma_split = split(gamma_temp, B_q)
            
            list_dtheta = np.array([gamma_seg.length for gamma_seg in gamma_split])
            list_theta = np.cumsum(list_dtheta)
            # Valid/invalid segments will alternate. Determine if second
            # segment is valid or not. If too large, likely invalid.
            list_valid = np.zeros(list_theta.shape).astype(bool).flatten()
            if (list_dtheta[1] > self.L/2):
                list_valid[0::2] = True
            else:
                list_valid[1::2] = True
            
            N_segs = list_valid.shape[0]
            dtheta_obs[ind_q] = np.sum(list_dtheta[list_valid])
            theta_obs = []
            gamma_obs = []
            for ind_seg, valid, theta, gamma_seg in zip(range(N_segs), list_valid, list_theta, gamma_split):
                if (valid):
                    gamma_obs.append(gamma_seg)
                    if (ind_seg == 0):
                        theta_obs.append([0, theta])
                    elif (ind_seg == (N_segs-1)):
                        theta_obs.append([list_theta[ind_seg-1], self.L])
                    else:
                        theta_obs.append([list_theta[ind_seg-1], theta])
            
            list_theta_obs.append(theta_obs)
            list_gamma_obs.append(gamma_obs)
        
        list_gamma_obs_flat = [gamma_seg for sublist in list_gamma_obs for gamma_seg in sublist]
        gamma_cumobs = MultiLineString(list_gamma_obs_flat)
        gamma_cumobs = linemerge(gamma_cumobs)
        
        # Unobserved path should move at top speed
        if (isinstance(gamma_cumobs, shapely.geometry.LineString)):
            L_notobs = self.L - gamma_cumobs.length
        elif (isinstance(gamma_cumobs, shapely.geometry.MultiLineString)):
            L_notobs = self.L
            for gamma_cumobs_ in gamma_cumobs:
                    L_notobs -= gamma_cumobs_.length
        
        T_notobs = L_notobs / self.v_max
        
        # Generate Kalman filter steady state equations that need to be solved
        # for each iteration of the loop
        kf_depth = 6
        self.list_kf_eqs = generate_func_kfsteadystate(depth=kf_depth)

        V = sympy.symbols('V')
        Twc = sympy.symbols('Twc')
        W = sympy.symbols('W')
        lambda_T0 = sympy.symbols('lambda_T0')
        
        N_loops = 100
        N_kf = np.zeros((N_loops, self.N_q)).astype(int); # Number of observations at each point of interest
        N_kf[0,:] = 1
        sig_max = np.zeros((N_loops-1, self.N_q))
        for ind_loop in range(N_loops-1):
            # Observed path should move at either max speed or minimum required 
            # time to take one sample
            T_obs = np.max(np.append(dtheta_obs/self.v_max, N_kf[ind_loop,:].reshape(-1,1)/self.fs, axis=1), axis=1)
            
            # Predicted worst case time for single loop with d_i observations
            # per location i
            T_loop = np.ceil((np.sum(T_obs) + T_notobs)/self.fs)*self.fs
            
            # Calculate steady state value for each point
            for ind_i in range(self.N_q):
                eq_temp = self.list_kf_eqs[N_kf[ind_loop,ind_i] - 1].subs({V:self.covar_obs, W:self.covar_e[ind_i, ind_i], Twc:T_loop})
                if (ind_loop == 0):
                    sig_max[ind_loop, ind_i] = sympy.nsolve(eq_temp, lambda_T0, self.covar_s[ind_i, ind_i])
                else:
                    sig_max[ind_loop, ind_i] = sympy.nsolve(eq_temp, lambda_T0, sig_max[ind_loop-1, ind_i])
            
            N_kf[ind_loop+1, :] = N_kf[ind_loop, :]
            # Add observation the poi with the highest uncertainty
            ind_max = np.argmax(sig_max[ind_loop, :])
            N_kf[ind_loop+1, ind_max] += 1
            
            if (N_kf[ind_loop+1, ind_max] >= kf_depth):
                break
        
        sig_max_temp = sig_max.max(axis=1)
        ind_row_max = np.argmin(sig_max_temp[(sig_max_temp > 0).flatten()])
        
        if (self.b_verbose):
            print(N_kf[ind_row_max,:])
            print(sig_max[ind_row_max,:])
        if (self.b_logging):
            self.logger.debug(N_kf[ind_row_max,:].flatten())
            self.logger.debug(sig_max[ind_row_max,:].flatten())
        
        # Create basis functions
        list_beta = np.linspace(0, self.L, num=(self.J+1))
        list_beta = list_beta[0:-1].reshape((-1,1))
        
        # Calculate the betas and portion of betas from which POI can be
        # observed
        list_dtheta_beta = self.calc_int_beta(list_theta_obs, list_beta)
        
        # Create list of alpha coefficients for optimization
        list_coeff = np.zeros((self.N_q, self.J))
        for ind_i in range(self.N_q):
            for ind_j in range(self.J):
                list_coeff[ind_i, ind_j] = np.sum(list_dtheta_beta[ind_i, ind_j])

        # Set up optimization problem
        prob_statement = pulp.LpProblem('Ostertag 2018', pulp.LpMinimize)
        
        # Create variables
        list_alphavar = [pulp.LpVariable('a{0:03d}'.format(i), lowBound=1/self.v_max,
                         cat=pulp.LpContinuous) for i in range(self.J)]
        
        # Add objective statement
        prob_statement += pulp.lpSum(list_alphavar)
        
        # Add constraints from greedy Kalman Filter alg
        for ind_coeff, coeff_row in enumerate(list_coeff):
            prob_statement += pulp.LpConstraint((pulp.lpDot(coeff_row, list_alphavar)), 
                                        sense=pulp.constants.LpConstraintGE, 
                                        name='KF const{0}'.format(ind_coeff), rhs=N_kf[ind_row_max, ind_coeff]/self.fs)
        
        # prob_statement.writeLP('OstertagModel.lp')
        prob_statement.solve()
        
        list_alpha = np.array([v.varValue for v in prob_statement.variables()]).reshape((-1,1))
        
        try:
            self.v_controller = np.append(list_beta, list_alpha, axis=1)
        except:
            print('list_beta: %d'.format(len(list_beta)))
            print('list_alpha %d'.format(len(list_alpha)))
        
        print('Ostertag v_controller solved ({0})'.format(pulp.LpStatus[prob_statement.status]))
        self.covar_bound = sig_max_temp[ind_row_max]
        # print(list_alpha)
        if (self.b_logging):
            self.logger.debug('v_controller solved ({0})'.format(pulp.LpStatus[prob_statement.status]))
            self.logger.debug('alpha  = ' + np.array2string(list_alpha.flatten(), separator=', ', formatter={'float_kind':lambda x: '{0:2.5f}'.format(x)}, max_line_width=2148))


    def calc_int_beta(self, list_theta_obs, list_beta):
        """
        Calculates the length of path along each gamma segment, denoted by 
        theta_start and theta_stop, where q_i is observable that is in 
        list_beta[j:j+1]
        """
        list_dtheta_beta = []
        
        # Iterate through all points of interest
        for ind_i, theta_obs in enumerate(list_theta_obs):
            dtheta_beta = []
            # Iterate through all betas
            beta_start = list_beta.item(0)
            for ind_j in range(self.J):                
                if (ind_j == (len(list_beta) - 1)):
                    beta_stop = self.L
                else:
                    beta_stop = list_beta.item(ind_j+1)

                # Iterate through all different observable segments of gamma
                dtheta_beta_seg = []
                for theta_obs_seg in theta_obs:
                    temp_dtheta = 0
                    
                    # theta_obs_ is a range where q_i is observable from gamma
                    theta_start = theta_obs_seg[0]
                    theta_stop = theta_obs_seg[1]
                    
                    # Covers all cases
                    if (theta_stop < theta_start):
                        if (theta_start < beta_stop):
                            temp_dtheta += min([(beta_stop - beta_start),
                                                (beta_stop - theta_start)])
                        elif (theta_stop > beta_start):
                            temp_dtheta += min([(beta_stop - beta_start),
                                                (theta_stop - beta_start)])
                    else:
                        if ((theta_start < beta_stop) and (theta_stop > beta_start)):
                            temp_dtheta += min([(theta_stop - beta_start),
                                                (beta_stop - beta_start), 
                                                (theta_stop - theta_start),
                                                (beta_stop - theta_start)])
                    dtheta_beta_seg.append(temp_dtheta)
                dtheta_beta.append(dtheta_beta_seg)
                beta_start = beta_stop
            list_dtheta_beta.append(dtheta_beta)
                
        return np.array(list_dtheta_beta)
    

    def calc_movement(self):
        """
        Velocity depends on current position and movement that can be taken 
        until next step
        """
        self.dtheta = 0
        time_plan = 1/self.fs
        ind_1 = np.argmin(self.v_controller[:,0] < self.theta)
        ind_1 = (ind_1 - 1) % self.J
        while (time_plan > 0):
            ind_2 = (ind_1 + 1) % self.J
            v_seg = 1/self.v_controller[ind_1, 1]
            dist_seg = (self.v_controller[ind_2, 0] - (self.theta + self.dtheta)) % self.L
            t_seg = (dist_seg / v_seg) # time to reach the next beta segment
            if (t_seg >= time_plan):
                self.dtheta += time_plan*v_seg
                time_plan = 0
            else:
                self.dtheta += dist_seg
                time_plan -= t_seg
                ind_1 = ind_2
 
  
    def get_optimalbound(self):
        """
        Return the theoretical optimal bound for the algorithm
        """
        return self.covar_bound








