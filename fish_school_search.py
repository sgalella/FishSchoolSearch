import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1234)
RES = 100

MIN_X = -5
MAX_X = 5
MIN_Y = -3
MAX_Y = 3

N = 10
W_SCALE = 10

MAX_ITER = 50


def get_lanscape_cost(x, y):
    """
    Generates a random landscape.

    Args:
        x (np.array): Meshgrid for x coordinate.
        y (np.array): Meshgrid for y coordinate.

    Returns:
        np.array: Array with costs for each coordinate.
    """
    return x ** 2 + y ** 2  # Sphere
    # return 1 + (x ** 2 / 4000) + (y ** 2 / 4000) - np.cos(x / np.sqrt(2)) - np.cos(y / np.sqrt(2))  # Gricwank
    # return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2  # Himmelblau
    # return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) - np.exp(0.5 * np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)) + np.exp(1) + 20   # Ackley
    # return 20 + x ** 2 - 10 * np.cos(2 * np.pi * x) - 10 * np.cos(2 * np.pi * y)  # Rastrigin


def generate_landscape():
    """
    Generates the landscape of the cost fitness.

    Returns:
        tuple: Coordinates of the landscape with the fitness of each position
    """
    x = np.linspace(MIN_X, MAX_X, RES)
    y = np.linspace(MIN_Y, MAX_Y, RES)
    X, Y = np.meshgrid(x, y)
    Z = get_lanscape_cost(X, Y)
    return X, Y, Z


def plot_landscape(X, Y, Z):
    """
    Plots the contour of the landscape.

    Args:
        X (np.array): x-coordinate.
        Y (np.array): y-coordinate.
        Z (np.array): Fitness at each location.
    """
    cs = plt.contour(X, Y, Z)
    plt.clabel(cs, inline=1, fontsize=6)
    plt.imshow(Z, extent=[MIN_X, MAX_X, MIN_Y, MAX_Y], origin="lower", alpha=0.3)


def compute_fitness(X_i):
    """
    Maps current position to the closest point in the landscape.

    Args:
        X_i (np.array): x and y coordinates of the fish position.

    Returns:
        float: Fitness of fish at position X_i.
    """
    pos_x, pos_y = X_i
    _, j = np.unravel_index((np.abs(X - pos_x)).argmin(), Z.shape)
    i, _ = np.unravel_index((np.abs(Y - pos_y)).argmin(), Z.shape)
    return np.fabs(Z[i, j] - np.max(Z))


class FishSchoolSearch:
    def __init__(self, num_individuals, step_ind=0.5, step_vol=1):
        self.step_ind = step_ind
        self.step_vol = step_vol
        self.num_individuals = num_individuals
        self.population = self._initialize_population()
        self.weights = self._initialize_weights()
        self.fitness = self._initialize_fitness()

    def _initialize_population(self):
        population = np.random.random((self.num_individuals, 2))
        population[:, 0] = (MAX_X - MIN_X) * population[:, 0] + MIN_X
        population[:, 1] = (MAX_Y - MIN_Y) * population[:, 1] + MIN_Y
        return population

    def _initialize_weights(self):
        weights = W_SCALE / 2 * np.ones((self.num_individuals, 1))
        return weights

    def _initialize_fitness(self):
        fitness = np.expand_dims(np.array([compute_fitness(self.population[row, :]) for row in range(self.num_individuals)]), axis=1) 
        return fitness

    def _bound_positions(self, pos_new, pos_old):
        idx_out_x = np.bitwise_or(pos_new[:, 0] > MAX_X, pos_new[:, 0] < MIN_X)
        idx_out_y = np.bitwise_or(pos_new[:, 1] > MAX_Y, pos_new[:, 1] < MIN_Y)
        pos_new[idx_out_x, :] = pos_old[idx_out_x, :]
        pos_new[idx_out_y, :] = pos_old[idx_out_y, :]
        return pos_new

    def _compute_individual_movement(self):
        pos_ind = np.zeros(self.population.shape)
        pos_ind = self.population + self.step_ind * np.random.uniform(-1, 1, size=(self.num_individuals, 2))
        pos_ind = self._bound_positions(pos_ind, self.population)
        return pos_ind

    def _compute_feeding(self, pos_ind):
        next_fitness = np.expand_dims(np.array([compute_fitness(pos_ind[row, :]) for row in range(self.num_individuals)]), axis=1)
        delta_fitness = next_fitness - self.fitness
        pos_ind[(delta_fitness < 0).flatten()] = pos_ind[(delta_fitness < 0).flatten()]
        delta_fitness[delta_fitness < 0] = 0
        next_weights = self.weights + delta_fitness / np.max(np.fabs(delta_fitness)) if np.max(delta_fitness) != 0 else self.weights
        next_weights[next_weights > W_SCALE] = W_SCALE
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

    def update(self):
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
        self.step_ind -= self.step_ind / MAX_ITER
        self.step_vol -= self.step_vol / MAX_ITER

    def plot_school(self):
        for idx in range(self.num_individuals):
            plt.plot(self.population[idx, 0], self.population[idx, 1], 'r*', markersize=self.weights[idx])
        plt.title(f"Best fitness: {np.max(self.fitness):.2f}")


def main():
    """ Runs the fish school optimization algorithm."""
    plt.figure(figsize=(8, 5))
    search = FishSchoolSearch(N)
    plot_landscape(X, Y, Z)
    search.plot_school()
    plt.colorbar(shrink=0.75)
    plt.ion()
    for _ in range(MAX_ITER):
        search.update()
        plt.cla()
        plot_landscape(X, Y, Z)
        search.plot_school()
        plt.draw()
        plt.pause(0.01)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    X, Y, Z = generate_landscape()
    main()
