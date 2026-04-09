import numpy as np

class Particle:
    def __init__(self, func, dim, vmin, vmax):
        self.position = np.random.uniform(vmin, vmax, dim)
        self.velocity = np.zeros(dim)

        self.fitness = func(self.position)

        self.best_part_pos = np.copy(self.position)
        self.best_part_fitness = self.fitness


def pso(func, max_iter, num_particles, dim, vmin, vmax, params):

    wmax = params["wmax"]
    wmin = params["wmin"]
    c1 = params["c1"]
    c2 = params["c2"]

    swarm = [Particle(func, dim, vmin, vmax) for _ in range(num_particles)]

    best_swarm_pos = np.zeros(dim)
    best_swarm_fitness = np.inf

    for particle in swarm:
        if particle.fitness < best_swarm_fitness:
            best_swarm_fitness = particle.fitness
            best_swarm_pos = np.copy(particle.position)

    for it in range(max_iter):

        w = wmax - ((wmax - wmin) / max_iter) * it

        for particle in swarm:

            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            particle.velocity = (
                w * particle.velocity
                + c1 * r1 * (particle.best_part_pos - particle.position)
                + c2 * r2 * (best_swarm_pos - particle.position)
            )

            particle.position += particle.velocity
            particle.position = np.clip(particle.position, vmin, vmax)

            particle.fitness = func(particle.position)

            if particle.fitness < particle.best_part_fitness:
                particle.best_part_fitness = particle.fitness
                particle.best_part_pos = np.copy(particle.position)

            if particle.fitness < best_swarm_fitness:
                best_swarm_fitness = particle.fitness
                best_swarm_pos = np.copy(particle.position)

    return {"position": best_swarm_pos, "cost": best_swarm_fitness}