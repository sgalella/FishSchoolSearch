import matplotlib.pyplot as plt
import numpy as np


class FishSchoolSearch:
    """ Fish school search. """
    def __init__(self, landscape, num_individuals, weight_scale=10, step_ind=0.5, step_vol=1):
        """
        Initializes the search.

        Args:
            landscape (FitnessLandscape): Fitness plane where fish interact.
            num_individuals (int): Number of fish.
            weight_scale (int, optional): Maximum weight.. Defaults to 10.
            step_ind (float, optional): Step individual movement.. Defaults to 0.5.
            step_vol (int, optional): Step volitive movement.. Defaults to 1.
        """
        self.landscape = landscape
        self.num_individuals = num_individuals
        self.weight_scale = weight_scale
        self.step_ind = step_ind
        self.step_vol = step_vol
        self.best_fitness = None
        self.population = self._initialize_population()
        self.weights = self._initialize_weights()
        self.fitness = self._initialize_fitness()

    def _initialize_population(self):
        """ Initializes position of fish randomly in landscape. """
        population = np.random.random((self.num_individuals, 2))
        population[:, 0] = (self.landscape.limits[1] - self.landscape.limits[0]) * population[:, 0] + self.landscape.limits[0]
        population[:, 1] = (self.landscape.limits[3] - self.landscape.limits[2]) * population[:, 1] + self.landscape.limits[2]
        return population

    def _initialize_weights(self):
        """ Computes the weights for the initial fish. """
        weights = self.weight_scale / 2 * np.ones((self.num_individuals, 1))
        return weights

    def _initialize_fitness(self):
        """ Computes the fitness for the initial fish. """
        fitness = np.expand_dims(np.array([self.landscape.evaluate_fitness(self.population[row, :]) for row in range(self.num_individuals)]), axis=1)
        self.best_fitness = max(fitness)[0]
        return fitness

    def _bound_positions(self, pos_new, pos_old):
        """
        Prevents fish from escaping the landscape.

        Args:
            pos_new (np.array): New position.
            pos_old (np.array): Position before movement.

        Returns:
            np.array: New position.
        """
        idx_out_x = np.bitwise_or(pos_new[:, 0] > self.landscape.limits[1], pos_new[:, 0] < self.landscape.limits[0])
        idx_out_y = np.bitwise_or(pos_new[:, 1] > self.landscape.limits[3], pos_new[:, 1] < self.landscape.limits[2])
        pos_new[idx_out_x, :] = pos_old[idx_out_x, :]
        pos_new[idx_out_y, :] = pos_old[idx_out_y, :]
        return pos_new

    def _compute_individual_movement(self):
        """ Moves individual in landscape. """
        pos_ind = np.zeros(self.population.shape)
        pos_ind = self.population + self.step_ind * np.random.uniform(-1, 1, size=(self.num_individuals, 2))
        pos_ind = self._bound_positions(pos_ind, self.population)
        return pos_ind

    def _compute_feeding(self, pos_ind):
        """
        Computes fitness and weights for the next iteration.

        Args:
            pos_ind (np.array): Individual position.

        Returns:
            tuple: Increase of fitness, fitness and weights for the next iteration.
        """
        next_fitness = np.expand_dims(np.array([self.landscape.evaluate_fitness(pos_ind[row, :]) for row in range(self.num_individuals)]), axis=1)
        self.best_fitness = max(self.best_fitness, max(next_fitness)[0])
        delta_fitness = next_fitness - self.fitness
        pos_ind[(delta_fitness < 0).flatten()] = pos_ind[(delta_fitness < 0).flatten()]
        delta_fitness[delta_fitness < 0] = 0
        next_weights = self.weights + delta_fitness / np.max(np.fabs(delta_fitness)) if np.max(delta_fitness) != 0 else self.weights
        next_weights[next_weights > self.weight_scale] = self.weight_scale
        next_weights[next_weights < 1] = 1
        return delta_fitness, next_fitness, next_weights

    def _compute_instintive_movement(self, pos_ind, delta_fitness):
        """
        Calculates movement following the individuals with maximum fitness.

        Args:
            pos_ind (np.array): Individual position.
            delta_fitness (np.array): Increase of fitness.

        Returns:
            np.array: Updated position after instintive movement.
        """
        pos_ins = np.zeros(self.population.shape)
        pos_ins = pos_ind + np.tile(np.sum((pos_ind - self.population) * np.tile(delta_fitness, 2), axis=0) / np.sum(delta_fitness), (self.num_individuals, 1)) \
            if np.sum(delta_fitness) != 0 else pos_ind
        pos_ins = self._bound_positions(pos_ins, pos_ind)
        pos_ins = self._bound_positions(pos_ins, pos_ind)
        return pos_ins

    def _compute_volitive_movement(self, pos_ins, next_weights):
        """
        Groups fishes together if weights increase. Otherwise, exploration is promoted. 

        Args:
            pos_ins (np.array): Position after instintive movement.
            next_weights (np.array): Weights of fish for the next iteration.

        Returns:
            np.array: Final position for the next iteration.
        """
        bar = np.tile(np.sum(pos_ins * np.tile(next_weights, 2), axis=0) / np.sum(next_weights, axis=0), (self.num_individuals, 1))
        if np.mean(next_weights) >= np.mean(self.weights):
            pos_next = pos_ins - np.multiply(self.step_vol * np.random.uniform(0, 1, size=(self.num_individuals, 2)), (pos_ins - bar))
        else:
            pos_next = pos_ins + np.multiply(self.step_vol * np.random.uniform(0, 1, size=(self.num_individuals, 1)), (pos_ins - bar))
        pos_next = self._bound_positions(pos_next, pos_ins)
        return pos_next

    def _breed_population(self, pos_next, next_weights):
        """
        Generates new individuals for the population.

        Args:
            pos_next (np.array): Final position for the next iteration.
            next_weights (np.array): Weights of fish for the next iteration.

        Returns:
            tuple: Final position and weights containing new individuals.
        """
        idx_new = np.argmin(next_weights)
        idx_parent1, idx_parent2 = next_weights.flatten().argsort()[::-1][:2]
        next_weights[idx_new] = (next_weights[idx_parent1] + next_weights[idx_parent2]) / 2
        pos_next[idx_new, :] = (pos_next[idx_parent1] + pos_next[idx_parent2]) / 2
        return pos_next, next_weights

    def update(self, num_iterations):
        """
        Computes one iteration of the fish school search.

        Args:
            num_iterations (int): Total number of iterations (update steps).
        """
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
        """ Plots fish in the landscape. """
        self.landscape.plot()
        for idx in range(self.num_individuals):
            plt.plot(self.population[idx, 0], self.population[idx, 1], 'r*', markersize=self.weights[idx])
        plt.title(f"Best fitness: {np.max(self.fitness):.2f}")
