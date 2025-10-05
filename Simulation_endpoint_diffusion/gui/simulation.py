import numpy as np
from scipy.spatial import cKDTree

class BirthEvent:
    def __init__(self, alpha, beta, ceta, particleRadius):
        self.alpha = alpha
        self.ceta = ceta
        self.particleRadius = particleRadius
        self.beta = beta

    def probability(self, P):
        inhibition = np.zeros(shape=P.shape[0])
        if self.beta > 0:
            tree = cKDTree(P)
            pairs = tree.query_pairs(r=6 * self.particleRadius, output_type='ndarray')
            if pairs.shape[0] > 0:
                AB = P[pairs[:, 1]] - P[pairs[:, 0]]
                D = np.linalg.norm(AB, axis=1)
                D[D < self.particleRadius * 2] = self.particleRadius * 2
                I = np.exp(-self.ceta * D)
                np.add.at(inhibition, pairs[:, 0], I)
                np.add.at(inhibition, pairs[:, 1], I)
        
        prob = self.alpha * (1 - self.beta * inhibition)
        prob[prob < 0] = 0 # Ensure probability is not negative
        return prob

    def execute(self, P, eventIndex, rs):
        try:
            # Find the first available (nan) slot for the new particle
            nan_indices = np.where(np.isnan(P[:, 0]))[0]
            if len(nan_indices) == 0:
                return P # Max particles reached, can't add more
            index = nan_indices[0]

            parent_pos = P[eventIndex, :]
            angle = rs.rand() * 2 * np.pi
            offset = np.array([np.cos(angle), np.sin(angle)]) * self.particleRadius
            P[index, :] = parent_pos + offset
        except IndexError:
            # This can happen in a race condition if the array is full
            pass
        return P

class DeathEvent:
    def __init__(self, mu):
        self.mu = mu

    def probability(self, P):
        return self.mu

    def execute(self, P, eventIndex, rs):
        P[eventIndex, :] = np.nan
        return P

class RectangularArena:
    def __init__(self, rect, repulsionForce, particleRadius):
        self.rect = rect
        self.repulsionForce = repulsionForce
        self.particleRadius = particleRadius

    def getForce(self, P):
        F = np.zeros(shape=P.shape)
        leftEdgeMap = P[:, 0] < self.rect[0] + self.particleRadius
        rightEdgeMap = P[:, 0] > self.rect[2] - self.particleRadius
        bottomEdgeMap = P[:, 1] < self.rect[1] + self.particleRadius
        topEdgeMap = P[:, 1] > self.rect[3] - self.particleRadius
        F[leftEdgeMap, 0] += self.repulsionForce
        F[rightEdgeMap, 0] += -self.repulsionForce
        F[bottomEdgeMap, 1] += self.repulsionForce
        F[topEdgeMap, 1] += -self.repulsionForce
        return F

    def getRandomPositions(self, rs, N=1):
        return rs.rand(N, 2) * np.array([self.rect[2] - self.rect[0], self.rect[3] - self.rect[1]]) + np.array([self.rect[0], self.rect[1]])

def createRigidPotential(Fr, Lr):
    def step(N, D):
        force_array = np.zeros_like(N)
        repel_mask = D < Lr
        force_array[repel_mask] = N[repel_mask] * Fr
        return force_array
    return step

class Simulation:
    def __init__(self, minTimeStep, initialParticles, maxParticles, arena, particleSpeed, particleCollision, particleCollisionMaxDistance, events, rs, callbacks=None):
        self.minTimeStep = minTimeStep
        self.initialParticles = initialParticles
        self.maxParticles = maxParticles
        self.particleSpeed = particleSpeed
        self.particleCollision = particleCollision
        self.particleCollisionMaxDistance = particleCollisionMaxDistance
        self.arena = arena
        self.events = events
        self.rs = rs
        self.callbacks = callbacks

    def step(self, P, timeStep):
        if timeStep <= 0: return P
        numCells = P.shape[0]
        F_deterministic = np.zeros(shape=P.shape)
        F_deterministic += self.arena.getForce(P)

        tree = cKDTree(P)
        pairs = tree.query_pairs(r=self.particleCollisionMaxDistance, output_type='ndarray')
        if pairs.shape[0] > 0:
            AB = P[pairs[:, 1]] - P[pairs[:, 0]]
            D = np.linalg.norm(AB, axis=1)
            DN = np.copy(D)
            DN[DN == 0] = 1
            N = AB / DN[:, None]
            PPF = self.particleCollision(N, D)
            np.add.at(F_deterministic, pairs[:, 0], -PPF)
            np.add.at(F_deterministic, pairs[:, 1], PPF)
            
        angle = self.rs.rand(numCells, 1) * 2 * np.pi
        radius = self.rs.normal(size=(numCells, 1)) * self.particleSpeed * np.sqrt(timeStep)
        displacement_stochastic = np.concatenate((np.cos(angle), np.sin(angle)), axis=1) * radius
        
        P += timeStep * F_deterministic + displacement_stochastic
        return P

    def simulate(self, simulationLength):
        P = np.full((self.maxParticles, 2), np.nan)
        P[:self.initialParticles, :] = self.arena.getRandomPositions(self.rs, self.initialParticles)
        t = 0
        nextEventTime = 0
        
        yield t, P[~np.isnan(P[:,0])]

        while t < simulationLength:
            cellMap = ~np.isnan(P[:, 0])
            active_P = P[cellMap, :]
            numCells = active_P.shape[0]

            if numCells == 0:
                print("Error: Number of cells reached 0")
                yield t, np.array([])
                return
            if numCells >= self.maxParticles:
                 print("Error: Max particles reached")
                 yield t, active_P
                 return

            if t >= nextEventTime:
                W = np.zeros(shape=numCells * len(self.events))
                for i, event in enumerate(self.events):
                    W[(i * numCells):((i + 1) * numCells)] = event.probability(active_P)
                
                weightSum = np.sum(W)
                if weightSum <= 0:
                    nextEventTime = t + self.minTimeStep
                else:
                    nextEventTime = t + (1 / weightSum) * np.log(1 / (1e-9 + self.rs.rand()))

                    R = self.rs.rand() * weightSum
                    W_cumsum = np.cumsum(W)
                    I = np.searchsorted(W_cumsum, R)
                    
                    event_type_idx = I // numCells
                    particle_idx_in_active = I % numCells
                    
                    # Map back to original P array index
                    original_indices = np.where(cellMap)[0]
                    eventIndex = original_indices[particle_idx_in_active]

                    chosenEvent = self.events[event_type_idx]
                    chosenEvent.execute(P, eventIndex, self.rs)

            timeStep = min(nextEventTime - t, self.minTimeStep)
            
            # Re-evaluate active particles after a potential event
            cellMap = ~np.isnan(P[:, 0])
            active_P = P[cellMap, :]
            if active_P.shape[0] > 0:
                updated_active_P = self.step(active_P, timeStep)
                P[cellMap, :] = updated_active_P

            t += timeStep
            
            yield t, P[~np.isnan(P[:,0])]

        yield t, P[~np.isnan(P[:,0])]