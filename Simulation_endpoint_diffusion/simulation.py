# import numpy as np;

# from scipy.spatial import cKDTree;
# from scipy.spatial.distance import cdist;

# import matplotlib.pyplot as plt;

# import time;

# ############
# ## Events ##
# ############
# class BirthEvent:
# 	def __init__(self, alpha, betaProportion, ceta, particleRadius):
# 		self.alpha 			= alpha;
# 		self.ceta  			= ceta;
# 		self.particleRadius = particleRadius;

# 		self.beta  = self.setBeta(betaProportion);

# 	def probability(self, P):
# 		inhibition = np.zeros(shape=P.shape[0]);

# 		if self.beta > 0:
# 			tree  = cKDTree(P);
# 			pairs = tree.query_pairs(r=6 * self.particleRadius, output_type='ndarray');

# 			if pairs.shape[0] > 0:
# 				AB = P[pairs[:, 1]] - P[pairs[:, 0]];
# 				D  = np.linalg.norm(AB, axis=1);
# 				D[D < self.particleRadius * 2] = self.particleRadius * 2;

# 				I = np.exp(-self.ceta / self.particleRadius * D);

# 				inhibition[pairs[:, 0]] += I;
# 				inhibition[pairs[:, 1]] += I;

# 		return self.alpha * (1 - self.beta * inhibition);

# 	def execute(self, P, eventIndex):
# 		index = np.argmax(np.isnan(P[:, 0]));

# 		P[index, :] = P[eventIndex, :];

# 		return P;

# 	def setBeta(self, proportion=1):
# 		return proportion / self.findMaxInhibition();

# 	def findMaxInhibition(self, layers=1000):
# 		s = 0;
# 		for i in range(1, layers):
# 			angles = [];
# 			for n in range(i):
# 				angles.append(0 + n * 60 / i);

# 			angles = np.array(angles) * np.pi / 180;

# 			distances = 2 * i * np.sqrt(3) / (np.sqrt(3) * np.cos(angles) + np.sin(angles));

# 			s += 6 * np.sum(np.exp(-distances * self.ceta));

# 		return s;

# class DeathEvent:
# 	def __init__(self, mu):
# 		self.mu = mu;

# 	def probability(self, P):
# 		return self.mu;

# 	def execute(self, P, eventIndex):
# 		P[eventIndex, :] = np.nan;

# 		return P;

# ###########
# ## Arena ##
# ###########
# class RectangularArena:
# 	def __init__(self, rect, repulsionForce, particleRadius):
# 		self.rect 	   		= rect;
# 		self.repulsionForce = repulsionForce;
# 		self.particleRadius = particleRadius;

# 	def getForce(self, P):
# 		F = np.zeros(shape=P.shape);

# 		leftEdgeMap   = P[:, 0] < self.rect[0] + self.particleRadius;
# 		rightEdgeMap  = P[:, 0] > self.rect[2] - self.particleRadius;
# 		bottomEdgeMap = P[:, 1] < self.rect[1] + self.particleRadius;
# 		topEdgeMap    = P[:, 1] > self.rect[3] - self.particleRadius;
		
# 		F[leftEdgeMap  , 0] +=  self.repulsionForce;
# 		F[rightEdgeMap , 0] += -self.repulsionForce;
# 		F[bottomEdgeMap, 1] +=  self.repulsionForce;
# 		F[topEdgeMap   , 1] += -self.repulsionForce;

# 		return F;

# 	def getBounds(self):
# 		return self.rect;

# 	def getRandomPositions(self, rs, N=1):
# 		return rs.rand(N, 2) * np.array([self.rect[2] - self.rect[0], self.rect[3] - self.rect[1]]) + np.array([self.rect[0], self.rect[1]]);

# class CircularArena:
# 	def __init__(self, centre, radius, repulsionForce, particleRadius):
# 		self.centre 	   	= centre;
# 		self.radius 	   	= radius;
# 		self.repulsionForce = repulsionForce;
# 		self.particleRadius = particleRadius;

# 	def getForce(self, P):
# 		F = np.zeros(shape=P.shape);

# 		AB = P - np.array(self.centre)[None, :];

# 		D = np.linalg.norm(AB, axis=1);
# 		D[D == 0] = 1;

# 		edgeMap = (D + self.particleRadius) > self.radius;

# 		if np.sum(edgeMap > 0):
# 			N = AB[edgeMap] / D[edgeMap, None];
# 			F[edgeMap, :] -= N * self.repulsionForce;

# 		return F;

# 	def getBounds(self):
# 		return [self.centre[0] - self.radius, self.centre[1] - self.radius, self.centre[0] + self.radius, self.centre[1] + self.radius];

# 	def getRandomPositions(self, rs, N=1):
# 		angle  = rs.rand(N, 1) * 2 * np.pi;
# 		radius = rs.rand(N, 1) * self.radius;

# 		return np.concatenate((np.cos(angle), np.sin(angle)), axis=1) * radius + np.array(self.centre);

# ##########################
# ## Particle interaction ##
# ##########################
# # De * a: Attraction strength
# # a: Width
# # re: Distance
# def createMorsePotential(De, a, re, Mp=np.inf):
# 	def morse(N, D):
# 		p = -a * (D - re);

# 		F = 2 * De * a * (np.exp(2 * p) - np.exp(p));
# 		F[F > Mp] = Mp;

# 		FA = np.matlib.repmat(F, 2, 1).T;

# 		return FA * N;

# 	return morse;

# def createRigidPotential(Fr, Lr):
# 	def step(N, D):
# 		N[D > Lr, :] = 0;

# 		return N * Fr;

# 	return step;


# ################
# ## Simulation ##
# ################
# class Simulation:
# 	def __init__(self, minTimeStep, initialParticles, maxParticles, arena, particleSpeed, particleCollision, particleCollisionMaxDistance, events, rs, callbacks=None):
# 		self.minTimeStep 	= minTimeStep;
# 		self.initialParticles = initialParticles;
# 		self.maxParticles	= maxParticles;
# 		self.particleSpeed 	= particleSpeed;

# 		self.particleCollision            = particleCollision;
# 		self.particleCollisionMaxDistance = particleCollisionMaxDistance;

# 		self.arena  = arena;
# 		self.events = events;

# 		self.rs = rs;

# 		self.callbacks = callbacks;

# 	def step(self, P, timeStep):
# 		numCells = P.shape[0];

# 		F = np.zeros(shape=P.shape);

# 		#################
# 		## Random walk ##
# 		#################
# 		angle  = self.rs.rand(numCells, 1) * 2 * np.pi;

# 		# TODO: Stochastic term should be multiplied by sqrt(delta t) not delta t!!!!! Solved ?? just multiplied instead of division?? 
# 		radius = self.rs.normal(size=(numCells, 1)) * self.particleSpeed * np.sqrt(timeStep);

# 		F += np.concatenate((np.cos(angle), np.sin(angle)), axis=1) * radius;

# 		##########################
# 		## Arena edge collision ##
# 		##########################
# 		F += self.arena.getForce(P);

# 		###########################
# 		## Cell cell interaction ##
# 		###########################
# 		tree  = cKDTree(P);
# 		pairs = tree.query_pairs(r=self.particleCollisionMaxDistance, output_type='ndarray');
		
# 		if pairs.shape[0] > 0:
# 			AB = P[pairs[:, 1]] - P[pairs[:, 0]];
# 			D  = np.linalg.norm(AB, axis=1);

# 			DN = np.copy(D);
# 			DN[DN == 0] = 1;
# 			N = AB / DN[:, None];

# 			PPF = self.particleCollision(N, D);

# 			F[pairs[:, 0]] -= PPF;
# 			F[pairs[:, 1]] += PPF;

# 			#for n in range(N.shape[0]):
# 			#	F[pairs[n, 0], :] -= PPF[n, :];
# 			#	F[pairs[n, 1], :] += PPF[n, :];

# 		# Integrate
# 		P += timeStep * F;

# 		return P;

# 	def simulate(self, simulationLength):
# 		################
# 		## Initialize ##
# 		################
# 		nextEventTime = -1;
# 		eventIndex    = -1;
# 		chosenEvent   = None;

# 		callbackTimings = {};
# 		if self.callbacks is not None:
# 			for key in self.callbacks:
# 				callbackTimings[key] = -1;

# 		# Initialize cells
# 		P = np.zeros(shape=(self.maxParticles, 2)) + np.nan;
# 		P[:self.initialParticles, :] = self.arena.getRandomPositions(self.rs, self.initialParticles);

# 		t = 0;
# 		nextPhysicsStep = t + self.minTimeStep;

# 		while t < simulationLength:
# 			###################
# 			## Execute event ##
# 			###################
# 			if t >= nextEventTime and nextEventTime > 0 and chosenEvent is not None:
# 				chosenEvent.execute(P, eventIndex);

# 			##################
# 			## Sanity Check ##
# 			##################
# 			numCells = np.sum(~np.isnan(P[:, 0]));
# 			if numCells == self.maxParticles:
# 				print("Error: Number of cells reached max cells allowed at time " + str(t));
					
# 				return P[cellMap, :], 2;

# 			if numCells == 0:
# 				print("Error: Number of cells reached 0 at time " + str(t));
				
# 				return P[cellMap, :], 1;

# 			#####################
# 			## Find next event ##
# 			#####################
# 			cellMap  = ~np.isnan(P[:, 0]);
# 			numCells = np.sum(cellMap);

# 			if t >= nextEventTime and numCells > 0:
# 				# Compute probability weight for each cell and event type
# 				W = np.zeros(shape=numCells * len(self.events));
# 				for i, event in enumerate(self.events):
# 					W[(i * numCells):((i + 1) * numCells)] = event.probability(P[cellMap, :]);

# 				W = np.cumsum(W);
# 				weightSum = W[-1];
				
# 				nextEventTime = t + (1 / weightSum) * np.log(1 / (0.00001 + self.rs.rand()));

# 				# Find event type
# 				R = np.random.rand() * weightSum;
# 				I = np.argmax(R <= W);

# 				eventIndex  = I % numCells;
# 				chosenEvent = self.events[int(I / numCells)];

# 				if nextEventTime < 0:
# 					print("Error: nextEventTime negative");
# 					break;

# 			#####################
# 			## Step simulation ##
# 			#####################
# 			timeStep = np.minimum(nextEventTime - t, self.minTimeStep);	# Advance time by either the time step or until next event

# 			#P[cellMap, :] = self.step(P[cellMap, :], timeStep);
			
# 			if t >= nextPhysicsStep:
# 				P[cellMap, :] = self.step(P[cellMap, :], self.minTimeStep);

# 				nextPhysicsStep += self.minTimeStep;
			

# 			t += timeStep;

# 			if self.callbacks is not None:
# 				for key in self.callbacks:
# 					if callbackTimings[key] < t:
# 						self.callbacks[key]["callback"](self, t, P);

# 						callbackTimings[key] += self.callbacks[key]["delay"];

# 		return P[cellMap, :], 0;

# ###########
# ## Debug ##
# ###########
# def plotFrame(simulation: Simulation, P, size):
# 	fig, ax = plt.subplots(figsize=(5, 5));

# 	for n in range(P.shape[0]):
# 		ax.add_artist(plt.Circle((P[n, 0], P[n, 1]), size, color='r'));		

# 	#plt.text();

# 	b = simulation.arena.getBounds();
# 	plt.axis([b[0], b[2], b[1], b[3]]);
# 	plt.show();


import numpy as np

from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt

import time

############
## Events ##
############
class BirthEvent:
	# MODIFIED: Removed betaProportion and added a fixed beta parameter
	def __init__(self, alpha, beta, ceta, particleRadius):
		self.alpha 			= alpha
		self.ceta  			= ceta
		self.particleRadius = particleRadius
		# MODIFIED: Beta is now a fixed value as per the paper (0.8)
		self.beta  			= beta

	def probability(self, P):
		inhibition = np.zeros(shape=P.shape[0])

		if self.beta > 0:
			tree  = cKDTree(P)
			pairs = tree.query_pairs(r=6 * self.particleRadius, output_type='ndarray')

			if pairs.shape[0] > 0:
				AB = P[pairs[:, 1]] - P[pairs[:, 0]]
				D  = np.linalg.norm(AB, axis=1)
				D[D < self.particleRadius * 2] = self.particleRadius * 2

				# MODIFIED: The inhibition kernel now matches e^(-gamma*r) with gamma=ceta
				# Removed division by self.particleRadius
				I = np.exp(-self.ceta * D)

				inhibition[pairs[:, 0]] += I
				inhibition[pairs[:, 1]] += I

		return self.alpha * (1 - self.beta * inhibition)

	# MODIFIED: Method signature changed to accept a random state object 'rs'
	def execute(self, P, eventIndex, rs):
		index = np.argmax(np.isnan(P[:, 0]))

		# MODIFIED: Daughter cell is now placed at distance R (radius) at a random angle
		parent_pos = P[eventIndex, :]
		angle = rs.rand() * 2 * np.pi
		offset = np.array([np.cos(angle), np.sin(angle)]) * self.particleRadius
		
		P[index, :] = parent_pos + offset

		return P

class DeathEvent:
	def __init__(self, mu):
		self.mu = mu

	def probability(self, P):
		return self.mu

	def execute(self, P, eventIndex):
		P[eventIndex, :] = np.nan

		return P

###########
## Arena ##
###########
class RectangularArena:
	def __init__(self, rect, repulsionForce, particleRadius):
		self.rect 	   		= rect
		self.repulsionForce = repulsionForce
		self.particleRadius = particleRadius

	def getForce(self, P):
		F = np.zeros(shape=P.shape)

		leftEdgeMap   = P[:, 0] < self.rect[0] + self.particleRadius
		rightEdgeMap  = P[:, 0] > self.rect[2] - self.particleRadius
		bottomEdgeMap = P[:, 1] < self.rect[1] + self.particleRadius
		topEdgeMap    = P[:, 1] > self.rect[3] - self.particleRadius
		
		F[leftEdgeMap  , 0] +=  self.repulsionForce
		F[rightEdgeMap , 0] += -self.repulsionForce
		F[bottomEdgeMap, 1] +=  self.repulsionForce
		F[topEdgeMap   , 1] += -self.repulsionForce

		return F

	def getBounds(self):
		return self.rect

	def getRandomPositions(self, rs, N=1):
		return rs.rand(N, 2) * np.array([self.rect[2] - self.rect[0], self.rect[3] - self.rect[1]]) + np.array([self.rect[0], self.rect[1]])

class CircularArena:
	def __init__(self, centre, radius, repulsionForce, particleRadius):
		self.centre 	   	= centre
		self.radius 	   	= radius
		self.repulsionForce = repulsionForce
		self.particleRadius = particleRadius

	def getForce(self, P):
		F = np.zeros(shape=P.shape)

		AB = P - np.array(self.centre)[None, :]

		D = np.linalg.norm(AB, axis=1)
		D[D == 0] = 1

		edgeMap = (D + self.particleRadius) > self.radius

		if np.sum(edgeMap > 0):
			N = AB[edgeMap] / D[edgeMap, None]
			F[edgeMap, :] -= N * self.repulsionForce

		return F

	def getBounds(self):
		return [self.centre[0] - self.radius, self.centre[1] - self.radius, self.centre[0] + self.radius, self.centre[1] + self.radius]

	def getRandomPositions(self, rs, N=1):
		angle  = rs.rand(N, 1) * 2 * np.pi
		radius = rs.rand(N, 1) * self.radius

		return np.concatenate((np.cos(angle), np.sin(angle)), axis=1) * radius + np.array(self.centre)

##########################
## Particle interaction ##
##########################
# De * a: Attraction strength
# a: Width
# re: Distance
def createMorsePotential(De, a, re, Mp=np.inf):
	def morse(N, D):
		p = -a * (D - re)

		F = 2 * De * a * (np.exp(2 * p) - np.exp(p))
		F[F > Mp] = Mp

		FA = np.matlib.repmat(F, 2, 1).T

		return FA * N

	return morse

def createRigidPotential(Fr, Lr):
	def step(N, D):
		N[D > Lr, :] = 0

		return N * Fr

	return step


################
## Simulation ##
################
class Simulation:
	def __init__(self, minTimeStep, initialParticles, maxParticles, arena, particleSpeed, particleCollision, particleCollisionMaxDistance, events, rs, callbacks=None):
		self.minTimeStep 	= minTimeStep
		self.initialParticles = initialParticles
		self.maxParticles	= maxParticles
		self.particleSpeed 	= particleSpeed

		self.particleCollision            = particleCollision
		self.particleCollisionMaxDistance = particleCollisionMaxDistance

		self.arena  = arena
		self.events = events

		self.rs = rs

		self.callbacks = callbacks

	def step(self, P, timeStep):
		numCells = P.shape[0]

		# RENAMED: This vector now only holds deterministic forces
		F_deterministic = np.zeros(shape=P.shape)

		##########################
		## Arena edge collision ##
		##########################
		F_deterministic += self.arena.getForce(P)

		###########################
		## Cell cell interaction ##
		###########################
		tree  = cKDTree(P)
		pairs = tree.query_pairs(r=self.particleCollisionMaxDistance, output_type='ndarray')
		
		if pairs.shape[0] > 0:
			AB = P[pairs[:, 1]] - P[pairs[:, 0]]
			D  = np.linalg.norm(AB, axis=1)

			DN = np.copy(D)
			DN[DN == 0] = 1
			N = AB / DN[:, None]

			PPF = self.particleCollision(N, D)

			F_deterministic[pairs[:, 0]] -= PPF
			F_deterministic[pairs[:, 1]] += PPF

		#################
		## Random walk ##
		#################
		# MODIFIED: The stochastic displacement is now calculated separately
		# This follows the form sqrt(2*D*dt)*Z
		angle  = self.rs.rand(numCells, 1) * 2 * np.pi
		# Note: self.particleSpeed = sqrt(2*D), so this term is correct
		radius = self.rs.normal(size=(numCells, 1)) * self.particleSpeed * np.sqrt(timeStep)
		displacement_stochastic = np.concatenate((np.cos(angle), np.sin(angle)), axis=1) * radius


		# MODIFIED: Integration now correctly applies the Euler-Maruyama method
		# Deterministic forces are scaled by timeStep, stochastic displacement is added directly.
		P += timeStep * F_deterministic + displacement_stochastic

		return P

	def simulate(self, simulationLength):
		################
		## Initialize ##
		################
		nextEventTime = -1
		eventIndex    = -1
		chosenEvent   = None

		callbackTimings = {}
		if self.callbacks is not None:
			for key in self.callbacks:
				callbackTimings[key] = -1

		# Initialize cells
		P = np.zeros(shape=(self.maxParticles, 2)) + np.nan
		P[:self.initialParticles, :] = self.arena.getRandomPositions(self.rs, self.initialParticles)

		t = 0

		while t < simulationLength:
			###################
			## Execute event ##
			###################
			if t >= nextEventTime and nextEventTime > 0 and chosenEvent is not None:
				# MODIFIED: Pass the random state object to the execute method
				chosenEvent.execute(P, eventIndex, self.rs)

			##################
			## Sanity Check ##
			##################
			numCells = np.sum(~np.isnan(P[:, 0]))
			if numCells == self.maxParticles:
				print("Error: Number of cells reached max cells allowed at time " + str(t))
					
				return P[~np.isnan(P[:, 0])], 2

			if numCells == 0:
				print("Error: Number of cells reached 0 at time " + str(t))
				
				return P[~np.isnan(P[:, 0])], 1

			#####################
			## Find next event ##
			#####################
			cellMap  = ~np.isnan(P[:, 0])
			numCells = np.sum(cellMap)

			if t >= nextEventTime and numCells > 0:
				# Compute probability weight for each cell and event type
				W = np.zeros(shape=numCells * len(self.events))
				for i, event in enumerate(self.events):
					W[(i * numCells):((i + 1) * numCells)] = event.probability(P[cellMap, :])

				W = np.cumsum(W)
				weightSum = W[-1]
				
				nextEventTime = t + (1 / weightSum) * np.log(1 / (0.00001 + self.rs.rand()))

				# Find event type
				R = np.random.rand() * weightSum
				I = np.argmax(R <= W)

				eventIndex  = I % numCells
				chosenEvent = self.events[int(I / numCells)]

				if nextEventTime < 0:
					print("Error: nextEventTime negative")
					break

			#####################
			## Step simulation ##
			#####################
			# MODIFIED: The simulation is now correctly coupled
			# The timestep is the smaller of the time to the next event or a max step size
			timeStep = np.minimum(nextEventTime - t, self.minTimeStep)

			# MODIFIED: Physics is now updated using the variable timeStep from the Gillespie algorithm
			P[cellMap, :] = self.step(P[cellMap, :], timeStep)

			t += timeStep

			if self.callbacks is not None:
				for key in self.callbacks:
					if callbackTimings[key] < t:
						self.callbacks[key]["callback"](self, t, P)

						callbackTimings[key] += self.callbacks[key]["delay"]

		return P[cellMap, :], 0

###########
## Debug ##
###########
def plotFrame(simulation: Simulation, P, size):
	fig, ax = plt.subplots(figsize=(5, 5))

	for n in range(P.shape[0]):
		ax.add_artist(plt.Circle((P[n, 0], P[n, 1]), size, color='r'))	

	b = simulation.arena.getBounds()
	plt.axis([b[0], b[2], b[1], b[3]])
	plt.show()