# minimizing-uncertainty
Code and simulation for minimizing the maximum eigenvalue of the Kalman filter covariance matrix.

## Getting Started
Project requires Python3.6.x. It may work on other versions but not been tested. 

Once Python3.6 has been installed, navigate to the desired workspace and clone the project using git.
```
git clone https://github.com/TheMonstertag/minimizing-uncertainty.git
```

### Prerequisites
With the project cloned to your local machine, enter the directory and download all required packages with pip.
```
cd minimizing-uncertainty
pip3 install -r requirements.txt
```

## Running the Simulator
The simulator files can be found under /src/, which contains:
  * main_sim.py: main script that should be executed to run the simulation
  * env_models.py: class containing different environmental models
  * drones.py: class containing different drone controller and sensing models

To run the simulation, simply call:
```
python3 main_sim.py
```

The simulation will run each drone control and sensing system from the same start point on a static path with random locations and uncertainty growth rates of N points of interest. After each drone has completed Nsteps of the simulation. A new set of random locations and growth rates for the points of interest will be generated and each drone will perform another Nsteps of simulation. By default, the simulation runs with maximum velocity VMAX = 30.0, observation noise V = 20.0, number of points of interest N = 6, and with nonoverlapping sensing regions, meaning that a drone can only sense one point of interest at a time. 

Additional arguments can be provided:
```
  -h, --help         show this help message and exit
  --vmax VMAX        maximum velocity of drone
  --v V              noise in observation model
  --n N              number of points of interest (q)
  --Nsteps NSTEPS    number of simulation steps
  --Ntests NTESTS    number of independent tests
  --Nrobust NROBUST  number of repetitions at different initial positions
  --verbose          enables verbose output
  --logging          enables debug logging
  --overlap          forces at least two points of interest to have 
                     overlapping sensing regions

```

Results are saved into csv files under minimizing-uncertainty/results/ with each row representing a different test instance. Each row has the following format:

length of path, [xoffset, yoffset of circle center], [position of points of interest], [uncertainty growth rates of points of interest], max eigenvalue of uncertainty at each timestep





