# import numpy as np;
# import pandas as pd;

# import simulation;
# import utils;

# ###########
# ## Input ##
# ###########

# # Size of measurable area (match image size, units is in microns in the example)
# measure_rect_side = 702 // 2;
# measure_rect = [-measure_rect_side, -measure_rect_side, measure_rect_side, measure_rect_side];

# # Simulation arena size [left, bottom, right, top]
# # Make e.g 30% larger than the measure rect
# arena_side = measure_rect_side * 1.3;
# arena_rect = [-arena_side, -arena_side, arena_side, arena_side];

# # Contains all the static parameters use by the simulation and ABC-method
# simulation_parameters = {
# 	"num_particles": 100,				# Number of initial particles in the simulation
# 	"max_particles": 2000,				# Max number of allowed particles before quitting simulation
# 	"simulation_length": 3600 * 24 * 4,	# Length of simulation, units in seconds in the example
# 	"simulation_step": 50,				# Max time in between timesteps
# 	"dr": 2.6,							# Resoultion of pair correlation function (should match what is used when measuing real data)
# 	"rMax": 65,						# Max range for the pair correlation function (should match what is used when measuing real data)
# 	"arena_rect": arena_rect,			
# 	"measure_rect": measure_rect, 
# 	"edge_force": 0.01,					# Collision force between cells and arena edge
# 	"repulsion_force": 0.1,				# Collision force between cells
# 	"beta_proportion": 1, 				
# 	"ceta": 1,	
# };

# # Contains uniform parameter priors ([min, max]) for the three non-static parameters. 
# # parameter_priors = {
# # 	"log10_alpha": [-6, -4.8], 	# Division rate (log 10 divisions / second)
# # 	"log10_d":     [-2, 1] , 	# Diffusion constant (log 10 microns^2 / 2)
# # 	"r": 		   [2, 20] , 	# Cell radius
# # };

# #######################
# ## Simulation set up ##
# #######################

# from simulation import *
# import utils

# '''
# Runs a simulation and returns summary statistics.
# param: seed Random seed used to draw the priors and in the simulation
# returns params: Static parameters used
# returns summary statistics: Summary statistics formatted as one numpy array. First indice is cell count, rest is pari correlation.
# return drawn parameters: Parameters drawn from the prior for this simulation
# returns ecode: Exit code from the simulation. 0: Ok simulation, 1: No cells left, 2: Max number of cells reached
# '''


# '''
# Run a simulation given parameter values and parameter priors. 
# param: seed Random seed used to draw the priors and in the simulation
# returns P: Particle positions as numpy array
# returns parameter: The variable parameter values as an array [log10_d, log10_alpha, r]
# returns ecode: Exit code from the simulation. 0: Ok simulation, 1: No cells left, 2: Max number of cells reached
# '''
# def run_simulation(seed):
#     params = simulation_parameters
#     rs = np.random.RandomState(seed=seed)
    
#     diffusion = 0.0
#     alpha = 10**-4.8
#     size = 7.5
#     cellSpeed = np.sqrt(2 * diffusion)
#     arena  = simulation.RectangularArena(params["arena_rect"], params["edge_force"] , size)
#     events = [simulation.BirthEvent(alpha, params["beta_proportion"], params["ceta"], size)]
#     collision_function = simulation.createRigidPotential(params["repulsion_force"], 2 * size)
#     sim = simulation.Simulation(
#                                 minTimeStep=params["simulation_step"], 
#                                 initialParticles=params["num_particles"], 
#                                 maxParticles=params["max_particles"],
#                                 particleSpeed=cellSpeed, arena=arena,
#                                 particleCollision=collision_function, 
#                                 particleCollisionMaxDistance=2 * size, 
#                                 events=events,
#                                 rs=rs
#                                 )
#     P, ecode = sim.simulate(params["simulation_length"])
#     simulation.plotFrame(sim, P=P, size = size)
    
	
    
# if __name__ == "__main__":
#     run_simulation(42)



import numpy as np
import pandas as pd

import simulation
import utils

###########
## Input ##
###########

# Size of measurable area (match image size, units is in microns in the example)
measure_rect_side = 702 // 2
measure_rect = [-measure_rect_side, -measure_rect_side, measure_rect_side, measure_rect_side]

# Simulation arena size [left, bottom, right, top]
# Make e.g 30% larger than the measure rect
arena_side = measure_rect_side * 1.3
arena_rect = [-arena_side, -arena_side, arena_side, arena_side]

# Contains all the static parameters use by the simulation and ABC-method
simulation_parameters = {
	"num_particles": 100,				# Number of initial particles in the simulation
	"max_particles": 2000,				# Max number of allowed particles before quitting simulation
	"simulation_length": 3600 * 24 * 4,	# Length of simulation, units in seconds in the example
	"simulation_step": 50,				# Max time in between timesteps
	"dr": 2.6,							# Resoultion of pair correlation function (should match what is used when measuing real data)
	"rMax": 65,							# Max range for the pair correlation function (should match what is used when measuing real data)
	"arena_rect": arena_rect,			
	"measure_rect": measure_rect, 
	"edge_force": 0.01,					# Collision force between cells and arena edge
	"repulsion_force": 0.1,				# Collision force between cells
	"ceta": 1,							# This now corresponds to gamma in the paper's model
}

# Contains uniform parameter priors ([min, max]) for the three non-static parameters. 
# parameter_priors = {
# 	"log10_alpha": [-6, -4.8], 	# Division rate (log 10 divisions / second)
# 	"log10_d":     [-2, 1] , 	# Diffusion constant (log 10 microns^2 / 2)
# 	"r": 		   [2, 20] , 	# Cell radius
# }

#######################
## Simulation set up ##
#######################

from simulation import *
import utils

'''
Runs a simulation and returns summary statistics.
param: seed Random seed used to draw the priors and in the simulation
returns params: Static parameters used
returns summary statistics: Summary statistics formatted as one numpy array. First indice is cell count, rest is pari correlation.
return drawn parameters: Parameters drawn from the prior for this simulation
returns ecode: Exit code from the simulation. 0: Ok simulation, 1: No cells left, 2: Max number of cells reached
'''


'''
Run a simulation given parameter values and parameter priors. 
param: seed Random seed used to draw the priors and in the simulation
returns P: Particle positions as numpy array
returns parameter: The variable parameter values as an array [log10_d, log10_alpha, r]
returns ecode: Exit code from the simulation. 0: Ok simulation, 1: No cells left, 2: Max number of cells reached
'''
def run_simulation(seed):
    params = simulation_parameters
    rs = np.random.RandomState(seed=seed)
    
    diffusion = 0.0
    alpha = 10**-4.8
    size = 3.5
    # The paper uses D for the diffusion coefficient, which is 'diffusion' here.
    # The relation cellSpeed = sqrt(2*D) is correct per the Langevin equation.
    cellSpeed = np.sqrt(2 * diffusion)
    arena  = simulation.RectangularArena(params["arena_rect"], params["edge_force"] , size)
    
    #  Instantiation of BirthEvent now passes the fixed beta value (0.8) from the paper
    beta = 0.8
    events = [simulation.BirthEvent(alpha, beta, params["ceta"], size)]
    
    collision_function = simulation.createRigidPotential(params["repulsion_force"], 2 * size)
    sim = simulation.Simulation(
                                minTimeStep=params["simulation_step"], 
                                initialParticles=params["num_particles"], 
                                maxParticles=params["max_particles"],
                                particleSpeed=cellSpeed, arena=arena,
                                particleCollision=collision_function, 
                                particleCollisionMaxDistance=2 * size, 
                                events=events,
                                rs=rs
                                )
    P, ecode = sim.simulate(params["simulation_length"])
    simulation.plotFrame(sim, P=P, size = size)
    
	
    
if __name__ == "__main__":
    run_simulation(42)