import matplotlib.pyplot as plt
import numpy as np

import optimization_functions as opt
from visualization import VisualizeSearch


class FishSchoolSearch:
    def __init__(self, landscape, num_individuals, weight_scale=10, step_ind=0.5, step_vol=1):
        self.landscape = landscape
        self.num_individuals = num_individuals
        self.weight_scale = weight_scale
        self.step_ind = step_ind
        self.step_vol = step_vol
        self.population = self._initialize_population()
        self.weights = self._initialize_weights()
        self.fitness = self._initialize_fitness()
        self.best_fitness = None

    def _initialize_population(self):
        population = np.random.random((self.num_individuals, 2))
        population[:, 0] = (self.landscape.limits[1] - self.landscape.limits[0]) * population[:, 0] + self.landscape.limits[0]
        population[:, 1] = (self.landscape.limits[3] - self.landscape.limits[2]) * population[:, 1] + self.landscape.limits[2]
        return population

    def _initialize_weights(self):
        weights = self.weight_scale / 2 * np.ones((self.num_individuals, 1))
        return weights

    def _initialize_fitness(self):
        fitness = np.expand_dims(np.array([self.landscape.evaluate_fitness(self.population[row, :]) for row in range(self.num_individuals)]), axis=1)
        self.best_fitness = max(fitness)
        return fitness

    def _bound_positions(self, pos_new, pos_old):
        idx_out_x = np.bitwise_or(pos_new[:, 0] > self.landscape.limits[1], pos_new[:, 0] < self.landscape.limits[0])
        idx_out_y = np.bitwise_or(pos_new[:, 1] > self.landscape.limits[3], pos_new[:, 1] < self.landscape.limits[2])
        pos_new[idx_out_x, :] = pos_old[idx_out_x, :]
        pos_new[idx_out_y, :] = pos_old[idx_out_y, :]
        return pos_new

    def _compute_individual_movement(self):
        pos_ind = np.zeros(self.population.shape)
        pos_ind = self.population + self.step_ind * np.random.uniform(-1, 1, size=(self.num_individuals, 2))
        pos_ind = self._bound_positions(pos_ind, self.population)
        return pos_ind

    def _compute_feeding(self, pos_ind):
        next_fitness = np.expand_dims(np.array([self.landscape.evaluate_fitness(pos_ind[row, :]) for row in range(self.num_individuals)]), axis=1)
        self.best_fitness = max(next_fitness)
        delta_fitness = next_fitness - self.fitness
        pos_ind[(delta_fitness < 0).flatten()] = pos_ind[(delta_fitness < 0).flatten()]
        delta_fitness[delta_fitness < 0] = 0
        next_weights = self.weights + delta_fitness / np.max(np.fabs(delta_fitness)) if np.max(delta_fitness) != 0 else self.weights
        next_weights[next_weights > self.weight_scale] = self.weight_scale
        next_weights[next_weights < 1] = 1
        return delta_fitness, next_fitness, next_weights

    def _compute_instintive_movement(self, pos_ind, delta_fitness):
        pos_ins = np.zeros(self.population.shape)
        pos_ins = pos_ind + np.tile(np.sum((pos_ind - self.population) * np.tile(delta_fitness, 2), axis=0) / np.sum(delta_fitness), (self.num_individuals, 1)) \
            if np.sum(delta_fitness) != 0 else pos_ind
        pos_ins = self._bound_positions(pos_ins, pos_ind)
        pos_ins = self._bound_positions(pos_ins, pos_ind)
        return pos_ins

    def _compute_volitive_movement(self, pos_ins, next_weights):
        bar = np.tile(np.sum(pos_ins * np.tile(next_weights, 2), axis=0) / np.sum(next_weights, axis=0), (self.num_individuals, 1))
        if np.mean(next_weights) >= np.mean(self.weights):
            pos_next = pos_ins - np.multiply(self.step_vol * np.random.uniform(0, 1, size=(self.num_individuals, 2)), (pos_ins - bar))
        else:
            pos_next = pos_ins + np.multiply(self.step_vol * np.random.uniform(0, 1, size=(self.num_individuals, 1)), (pos_ins - bar))
        pos_next = self._bound_positions(pos_next, pos_ins)
        return pos_next

    def _breed_population(self, pos_next, next_weights):
        idx_new = np.argmin(next_weights)
        idx_parent1, idx_parent2 = next_weights.flatten().argsort()[::-1][:2]
        next_weights[idx_new] = (next_weights[idx_parent1] + next_weights[idx_parent2]) / 2
        pos_next[idx_new, :] = (pos_next[idx_parent1] + pos_next[idx_parent2]) / 2
        return pos_next, next_weights

    def update(self, num_iterations):
        # Individual movement
        pos_ind = self._compute_individual_movement()
        # Feeding
        delta_fitness, next_fitness, next_weights = self._compute_feeding(pos_ind)
        # Collective-instinctive movement
        pos_ins = self._compute_instintive_movement(pos_ind, delta_fitness)
        # Collective-volitive movement
        pos_next_parents = self._compute_volitive_movement(pos_ins, next_weights)
        # Breeding
        pos_next, next_weights = self._breed_population(pos_next_parents, next_weights)
        # Update values
        self.weights = next_weights
        self.population = pos_next
        self.fitness = next_fitness
        self.step_ind -= self.step_ind / num_iterations
        self.step_vol -= self.step_vol / num_iterations

    def plot(self):
        self.landscape.plot()
        for idx in range(self.num_individuals):
            plt.plot(self.population[idx, 0], self.population[idx, 1], 'r*', markersize=self.weights[idx])
        plt.title(f"Best fitness: {np.max(self.fitness):.2f}")


def main():
    np.random.seed(1234)
    resolution = 100
    limits = [-5, 5, -3, 3]  # x_min, x_max, y_min, y_max
    landscape = opt.SphereLandscape(limits, resolution)
    search = FishSchoolSearch(landscape, num_individuals=10)
    VisualizeSearch.show_all(search, num_iterations=20)


if __name__ == '__main__':
    main()