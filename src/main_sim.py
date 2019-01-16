#!/usr/bin/env python
"""
main_sim.py
Michael Ostertag

A simulation environment to test fire tracking control algorithms on drones.
The simulation keeps track of drone positions and states, the current state of
the fire, the overall terrain, and (TODO) communication links between the 
drones.
"""

import time
import argparse

import math
import logging
import numpy as np
import csv
from env_models import Model_StaticFire, Model_StaticCircle_Random, Model_StaticCircle_set
from drones import Drone_Constant, Drone_Smith, Drone_Ostertag
from shapely.geometry import Point, Polygon, LineString
from shapely.geometry.polygon import orient

import matplotlib.cm as colors
import matplotlib.pyplot as plt

from noise import pnoise2

####################
class Sim_Environment():
    """
    description
    """
    
    def __init__(self, size=[100,100], swarm_controller=None, env_model=None, 
                 step_size=1, step_time=1, b_perlin=True, b_verbose=True, 
                 b_logging=True):
        self.b_verbose = b_verbose
        self.b_logging = b_logging
        self.size = size
        self.step_size = step_size
        self.step_time = step_time
        self.map_terrain = np.zeros((size))
        self.generate_terrain(b_perlin)
        
        self.swarm_controller = swarm_controller
        if (swarm_controller == None):
            self.b_swarm_controller = False
        else:
            self.b_swarm_controller = True
            
        self.env_model = env_model
        if (env_model == None):
            self.b_env_model = False
        else:
            self.b_env_model = True
            self.env_model.set_map(self.map_terrain)
        
        self.filename_results = time.strftime('Result_%Y%m%d_%H%M%S', time.localtime())
        
        plt.ion()   # Enable interactive plotting
    
    
    def set_swarm_controller(self, swarm_controller):
        """
        Links a swarm controller to the simulation environment after initialization
        """
        self.swarm_controller = swarm_controller
        self.b_swarm_controller = True
        self.swarm_controller.set_terrain_map(self.map_terrain)
        self.swarm_controller.set_steps(self.step_size, self.step_time)
       
        
    def set_env_model(self, env_model):
        """
        Links a fire tracker to the simulation environment after initialization
        """
        self.env_model = env_model
        self.b_env_model = True
        self.env_model.set_terrain_map(self.map_terrain)
        self.env_model.set_steps(self.step_size, self.step_time)
        
        
    def get_map_terrain(self):
        """
        Gets the object reference to the terrain map
        """
        return self.map_terrain
    
    
    def generate_terrain(self, b_perlin=True):
        """
        Generates terrain using Perlin noise from Python package noise 1.2.2 
        (https://pypi.python.org/pypi/noise/)
         
         pnoise2 = noise2(x, y, octaves=1, persistence=0.5, lacunarity=2.0, 
                          repeatx=1024, repeaty=1024, base=0.0)
         
         return perlin "improved" noise value for specified coordinate
        
         octaves -- specifies the number of passes for generating fBm noise,
             defaults to 1 (simple noise).
        
         persistence -- specifies the amplitude of each successive octave relative
             to the one below it. Defaults to 0.5 (each higher octave's amplitude
             is halved). Note the amplitude of the first pass is always 1.0.
        
         lacunarity -- specifies the frequency of each successive octave relative
             to the one below it, similar to persistence. Defaults to 2.0.
        
         repeatx, repeaty, repeatz -- specifies the interval along each axis when 
             the noise values repeat. This can be used as the tile size for creating 
             tileable textures
        
         base -- specifies a fixed offset for the input coordinates. Useful for
             generating different noise textures with the same repeat interval
        """
        
        scale_x = 0.01*self.step_size
        offset_x = 10
        scale_y = 0.01*self.step_size
        offset_y = 10
        scale_alt = 0 # 0.1
        offset_alt = 1000
        if (b_perlin):
            for val_x in range(self.size[0]):
                for val_y in range(self.size[1]):
                    x = scale_x*val_x + offset_x
                    y = scale_y*val_y + offset_y
                    self.map_terrain[val_x, val_y] = pnoise2(x, y, octaves=4, persistence=1.5, lacunarity=0.5)*scale_alt + offset_alt
        else:
            self.map_terrain = np.zeros((self.size)) + offset_alt
        
        if (self.b_verbose):
            print('Min: {0:0.2f} m   Max: {1:0.2f} m'.format(self.map_terrain.min(), 
                  self.map_terrain.max()))
    
    
    def update(self):
        """
        description
        """
        
        if (self.env_model):
            self.env_model.update()
        
        if (self.swarm_controller):
            self.swarm_controller.update()
    
    
    def save_results(self,param_ros=0, 
                          param_covar_0_scale=0,
                          param_v_max=0,
                          param_n_obs=0,
                          param_N_q=0,
                          theta0=0):
        """
        Save the configuration results from the environmental model and the 
        covariance results from the individual drones to a csv file
        """
        list_config = self.env_model.get_config()
        list_results = self.swarm_controller.get_results()
        
        np.set_printoptions(linewidth=1024, suppress=True)
        
        filename_temp = '{0}_vmax{2:d}_nobs{3:d}_Nq{4:d}_ros{1:d}.csv'.format(
                self.filename_results, math.trunc(param_ros), 
                math.trunc(param_v_max),math.trunc(param_n_obs), 
                math.trunc(param_N_q))
        with open('Results/' + filename_temp, 'a', newline='') as fid:
            writer = csv.writer(fid)            
            for result in list_results:
                temp_row = list_config + [theta0] + list(result.flatten())
                writer.writerow(temp_row)
            
        
    def visualize(self):
        # Make something pretty like http://www.harrisgeospatial.com/docs/EM1_SurfacePlot.html
        # https://matplotlib.org/basemap/users/examples.html
        tempx, tempy = np.meshgrid(range(self.size[0]), range(self.size[1]))
        
        plt.clf()
        plt.figure(1)
        plt.subplot(111)
        plt.pcolor(tempx, tempy, self.map_terrain, cmap='Greens_r') # Can use 'terrain' colormap once elevation is normalized
        plt.colorbar()
        
        if (self.b_env_model):
            self.env_model.visualize()
        
        if (self.b_swarm_controller):
            self.swarm_controller.visualize()
        
        plt.draw()
        plt.pause(0.01)


####################
class SwarmController():
    """
    SwarmController controls the communication between all drones and manages 
    their movement calculation, updates, and visualizations
    """
    
    def __init__(self, gamma, poi, covar_e, map_terrain=np.matrix([]), map_data=np.matrix([]), 
                 N_drones=1, pos_start=0, list_pos_start=[], step_size=1, step_time=1, 
                 param_covar_0_scale=100, param_v_max=10, param_n_obs=5, param_ros=1,
                 b_verbose=True, b_logging=True):
        """
        Initializes the swarm controller
        """
        self.b_verbose = b_verbose
        self.b_logging = b_logging
        self.map_terrain = map_terrain
        self.map_data = map_data
        
        self.step_size = 1
        self.step_time = 1
        
        self.list_drones = []
        temp_covar_max = []
        for drone_id in range(N_drones):
            if (list_pos_start and len(list_pos_start) > drone_id):
                pos = list_pos_start[drone_id]
            elif (list_pos_start and len(list_pos_start) <= drone_id):
                pos = list_pos_start[drone_id] + np.random.randn(1)*2*self.step_size
            elif (not(list_pos_start) and (drone_id > 0)):
                pos = pos_start + np.random.randn(1)*2*self.step_size
            else:
                pos = pos_start
            
            # Generate a rectangular observation window for a camera with 60 
            # deg FOV in x and 45 deg FOV in y
            z = 15
            ang_fov = np.array([45/180, 30/180])*math.pi
            dx, dy = np.tan(ang_fov)*z
            obs_window = Polygon([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
            
            self.list_drones.append(Drone_Constant(loop_path=gamma, poi=poi, 
                                        covar_e=covar_e, obs_window=obs_window,
                                        covar_obs=param_n_obs, drone_id=(100+drone_id), 
                                        theta0=pos, fs=self.step_time, 
                                        covar_s_0_scale=param_covar_0_scale,
                                        v_const=param_v_max, v_max=param_v_max,
                                        b_verbose=self.b_verbose, b_logging=self.b_logging))
            temp_covar_max.append([100+drone_id, self.list_drones[-1].get_optimalbound()])
            self.list_drones.append(Drone_Smith(loop_path=gamma, poi=poi, 
                                        covar_e=covar_e, obs_window=obs_window,
                                        covar_obs=param_n_obs, drone_id=(200+drone_id), 
                                        theta0=pos, fs=self.step_time, 
                                        covar_s_0_scale=param_covar_0_scale,
                                        v_max=param_v_max, J=200,
                                        b_verbose=self.b_verbose, b_logging=self.b_logging))
            temp_covar_max.append([200+drone_id, self.list_drones[-1].get_optimalbound()])
            self.list_drones.append(Drone_Ostertag(loop_path=gamma, poi=poi, 
                                        covar_e=covar_e, obs_window=obs_window,
                                        covar_obs=param_n_obs, drone_id=(300+drone_id), 
                                        theta0=pos, fs=self.step_time, 
                                        covar_s_0_scale=param_covar_0_scale,
                                        v_max=param_v_max, J=200,
                                        b_verbose=self.b_verbose, b_logging=self.b_logging))
            temp_covar_max.append([300+drone_id, self.list_drones[-1].get_optimalbound()])
            
            if (self.b_logging):
                logger.debug('param_v_max = {0}'.format(param_v_max))
                logger.debug('param_n_obs = {0}'.format(param_n_obs))
                logger.debug('param_ros = {0}'.format(param_ros))
        
        self.list_covar_max = np.array(temp_covar_max).reshape((-1,2))
        self.list_covar_max_0 = np.array(temp_covar_max).reshape((-1,2))
        
        # for ind, drone in enumerate(self.list_drones):
        #     list_temp = self.list_drones[:]
        #     del list_temp[ind]
        #     drone.set_neighbors(list_temp)
            
    
    def update(self):
        """
        description
        """
        # Calculate potentials for drones for movement
        for drone in self.list_drones:
            drone.calc_movement()
            
        # Move all drones simultaneously and then capture data
        temp_covar_max = np.zeros((len(self.list_drones), 1))
        for ind, drone in enumerate(self.list_drones):
            drone.update()
            temp_covar_max[ind,0] = np.max(np.diag(drone.get_covar_s()))
        self.list_covar_max = np.append(self.list_covar_max, temp_covar_max, axis=1)
    
    
    def reset_covar_max(self):
        """
        Reset self.list_covar_max to initial value
        """
        self.list_covar_max= self.list_covar_max_0[:]
    
    def set_terrain_map(self, map_terrain):
        """
        Sets the map of the terrain and updates all drones
        """
        self.map_terrain = map_terrain
        # for drone in self.list_drones:
        #     drone.set_terrain_map(self.map_terrain) # Drones not currently using map
        
        
    def set_data_map(self, map_data):
        """
        Sets the map of the data values and updates all drones
        """
        self.map_data = map_data
        for drone in self.list_drones:
            drone.set_data_map(map_data)
    
    
    def set_data_poly(self, poly_in):
        for drone in self.list_drones:
            drone.set_data_poly(poly_in)
    
    
    def set_steps(self, step_size=None, step_time=None):
        if (step_size):
            self.step_size = step_size
        if (step_time):
            self.step_time = step_time
        for drone in self.list_drones:
            drone.set_steps(step_size=self.step_size, step_time=self.step_time)
        
        if (not(step_size) and not(step_time)):
            return False
        else:
            return True


    def get_results(self):
        """
        Return results of the drone IDs and their covariance values
        """
        return self.list_covar_max
    
    
    def reset_drone(self, theta0, covar_0_scale=100):
        """
        Resets the drone to a provided theta0 with some initial 
        covariance scaled by covar_0_scale while maintaining any generated
        models
        """
        self.reset_covar_max()
        for ind, drone in enumerate(self.list_drones):
            drone.init_covar(covar_0_scale)
            drone.set_theta(theta0)
    
        
    def visualize(self):
        for ind, drone in enumerate(self.list_drones):
            drone.visualize(colors.get_cmap('tab10')(ind % 10))


####################
"""
Main control loop
"""
if __name__ == '__main__':
    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--vmax', type=float, default=30.0, 
                        help='maximum velocity of drone')
    parser.add_argument('--nobs', type=float, default=20.0,
                        help='noise in observation model')
    parser.add_argument('--Nq', type=int, default=6,
                        help='number of points of interest (q)')
    parser.add_argument('--Nsteps', type=int, default=600,
                        help='number of simulation steps')
    parser.add_argument('--Ntests', type=int, default=100,
                        help='number of independent tests')
    parser.add_argument('--Nrobust', type=int, default=1,
                        help='number of repetitions at different initial positions')
    parser.add_argument('--verbose', action='store_true',
                        help='enables verbose output')
    parser.add_argument('--logging', action='store_true',
                        help='enables debug logging')
    parser.add_argument('--overlap', action='store_true',
                        help='forces two points of interest to have overlapping sensing region')
    args = parser.parse_args()
    
    N_steps = args.Nsteps
    N_drones = 1
    env_size = [500, 500]
    step_size = 1 # 1 meter
    step_time = 1 # 1 second
    
    N_tests = args.Ntests
    N_robust = args.Nrobust
    param_ros = 1
    param_covar_0_scale = 100
    param_v_max = args.vmax
    param_n_obs = args.nobs
    param_N_q = args.Nq
    
    b_verbose = args.verbose
    b_logging = args.logging
    b_overlap = args.overlap
    
    if (b_logging):
        logger_filename = time.strftime('Log_%Y%m%d_%H%M%S.log', time.localtime())
        # TODO If Logs doesn't exist, create folder.
        logger_format = "[%(asctime)s:%(filename)s:%(lineno)s - %(funcName)s] %(message)s"
        logging.basicConfig(filename='Logs/' + logger_filename, level=logging.DEBUG,
                            format=logger_format, datefmt='%Y%m%d %H%M%S')
        logger = logging.getLogger('root')
    
    # Configure connections and initialize
    sim_env = Sim_Environment(size=env_size, step_size=step_size, 
                              step_time=step_time, b_perlin=False, b_verbose=b_verbose)  
    
    for ind_test in range(N_tests):
        print('Test {0:4d}/{1}'.format(ind_test+1, N_tests))
        list_drone_start = [0]    # value represents distance around the path
        # TODO Update list_drone_start to be a random variable
    
        env_model = Model_StaticCircle_Random(env_size=env_size, step_size=step_size, 
                                     step_time=step_time, param_ros=param_ros, 
                                     path_length=500, N_q=param_N_q, dtheta_min=30,
                                     b_verbose=b_verbose, b_logging=b_logging,
                                     b_overlap=b_overlap)
        # list_theta = [ 63.31402806, 164.60632657, 239.76381444, 327.7605043,
        #               400.60929276, 496.68597194]
        # list_p = [0.80423176, 0.85280985, 0.77557588, 0.13116952, 0.84202459,
        #           0.97468646]
        # env_model = Model_StaticCircle_set(list_theta, list_p, 
        #                              env_size=env_size, step_size=step_size, 
        #                              step_time=step_time, param_ros=param_ros, 
        #                              path_length=500,
        #                              b_verbose=b_verbose, b_logging=b_logging)
        sim_env.set_env_model(env_model)
        
        swarm_controller = SwarmController(gamma=env_model.get_gamma(), 
                            poi=env_model.get_poi(), covar_e=env_model.get_covar_env(),
                            N_drones=1, 
                            list_pos_start=list_drone_start,
                            param_covar_0_scale=param_covar_0_scale, 
                            param_v_max=param_v_max, param_n_obs=param_n_obs,
                            param_ros=param_ros,
                            b_verbose=b_verbose, b_logging=b_logging)
        sim_env.set_swarm_controller(swarm_controller)
        
        # Simulate for N_robust different initial positions
        list_theta0 = np.linspace(0, param_v_max, N_robust+1)
        list_theta0 = list_theta0[:-1]
        
        for theta0 in list_theta0:
            swarm_controller.reset_drone(theta0, param_covar_0_scale)
            
            # Iterate through simulation
            for ind_loop in range(N_steps):
                if (b_verbose):
                    print(' ')
                    print('Time: {0} s ({1:3.1f}%)'.format(step_time*ind_loop, 100*(ind_loop+1)/N_steps))
                if (b_logging):
                    logger.debug('-------------------------------')
                    logger.debug('Time = {0} s ({1:3.1f}%)'.format(step_time*ind_loop, 100*(ind_loop+1)/N_steps))
                sim_env.update()
                
                # Update visualization
                # if ((ind_loop >= 2) and ((ind_loop % 5) == 0)):
                #     sim_env.visualize()
                # sim_env.visualize()
                time.sleep(0.01)
        
            sim_env.save_results(param_ros=param_ros, 
                                 param_covar_0_scale=param_covar_0_scale,
                                 param_v_max=param_v_max,
                                 param_n_obs=param_n_obs,
                                 param_N_q=param_N_q,
                                 theta0=theta0)
    logging.shutdown()