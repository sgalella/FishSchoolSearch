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


def random_contour(x, y):
    # return x ** 2 + y ** 2  # Sphere
    # return 1 + (x ** 2 / 4000) + (y ** 2 / 4000) - np.cos(x / np.sqrt(2)) - np.cos(y / np.sqrt(2))  # Gricwank
    # return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2  # Himmelblau
    # return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) - np.exp(0.5 * np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)) + np.exp(1) + 20   # Ackley
    return 20 + x ** 2 - 10 * np.cos(2 * np.pi * x) - 10 * np.cos(2 * np.pi * y)  # Rastrigin


def random_meshgrid():
    x = np.linspace(MIN_X, MAX_X, RES)
    y = np.linspace(MIN_Y, MAX_Y, RES)
    X, Y = np.meshgrid(x, y)
    Z = random_contour(X, Y)
    return X, Y, Z


def plot_landscape(X, Y, Z):
    cs = plt.contour(X, Y, Z)
    plt.clabel(cs, inline=1, fontsize=6)
    plt.imshow(Z, extent=[MIN_X, MAX_X, MIN_Y, MAX_Y], origin="lower", alpha=0.3)


def compute_fitness(X_i):
    pos_x, pos_y = X_i
    _, j = np.unravel_index((np.abs(X - pos_x)).argmin(), Z.shape)
    i, _ = np.unravel_index((np.abs(Y - pos_y)).argmin(), Z.shape)
    return np.fabs(Z[i, j] - np.max(Z))


def create_fish(n):
    W = W_SCALE / 2 * np.ones((N, 1))
    P = np.array([(np.random.uniform(MIN_X, MAX_X), np.random.uniform(MIN_Y, MAX_Y)) for _ in range(n)])
    F = np.expand_dims(np.array([compute_fitness(P[row, :]) for row in range(n)]), axis=1)
    return W, P, F
    

def plot_school(W, P):
    for idx in range(N):
        plt.plot(P[idx, 0], P[idx, 1], 'r*', markersize=W[idx])


def bound_positions(P_new, P_old):
    idx_out_x = np.bitwise_or(P_new[:, 0] > MAX_X, P_new[:, 0] < MIN_X)
    idx_out_y = np.bitwise_or(P_new[:, 1] > MAX_Y, P_new[:, 1] < MIN_Y)
    P_new[idx_out_x, :] = P_old[idx_out_x, :]
    P_new[idx_out_y, :] = P_old[idx_out_y, :]
    return P_new


def run_fish_school(W, P, F, step_ind, step_vol):
    # Individual movement
    P_ind = np.zeros(P.shape)
    P_ind = P + step_ind * np.random.uniform(-1, 1, size=(N, 2))
    P_ind = bound_positions(P_ind, P)

    # Feeding
    F_next = np.expand_dims(np.array([compute_fitness(P_ind[row, :]) for row in range(N)]), axis=1)
    delta_F = F_next - F
    P_ind[(delta_F < 0).flatten()] = P[(delta_F < 0).flatten()]  # Don't move if new position decreases fitness
    delta_F[delta_F < 0] = 0
    W_next = W + delta_F / np.max(np.fabs(delta_F)) if np.max(delta_F) != 0 else W
    W_next[W_next > W_SCALE] = W_SCALE
    W_next[W_next < 1] = 1

    # Collective-instinctive movement
    P_ins = np.zeros(P.shape)
    P_ins = P_ind + np.tile(np.sum((P_ind - P) * np.tile(delta_F, 2), axis=0) / np.sum(delta_F), (N, 1)) if np.sum(delta_F) != 0 else P_ind
    P_ins = bound_positions(P_ins, P_ind)

    # Collective-volitive movement
    Bar = np.tile(np.sum(P_ins * np.tile(W_next, 2), axis=0) / np.sum(W_next, axis=0), (N, 1))
    if np.mean(W_next) >= np.mean(W):
        P_next = P_ins - np.multiply(step_vol * np.random.uniform(0, 1, size=(N, 2)), (P_ins - Bar))
    else:
        P_next = P_ins + np.multiply(step_vol * np.random.uniform(0, 1, size=(N, 1)), (P_ins - Bar))
    P_next = bound_positions(P_next, P_ins)

    # Breeding
    idx_new = np.argmin(W_next)
    idx_parent1, idx_parent2 = W_next.flatten().argsort()[::-1][:2]
    W_next[idx_new] = (W_next[idx_parent1] + W_next[idx_parent2]) / 2
    P_next[idx_new, :] = (P_next[idx_parent1] + P_next[idx_parent2]) / 2

    # Update values
    W = W_next
    P = P_next
    F = F_next

    return W, P, F


def main():
    plt.figure(figsize=(8, 5))
    W, P, F = create_fish(N)
    plot_landscape(X, Y, Z)
    plot_school(W, P)
    plt.colorbar(shrink=0.75)
    plt.title(f"Total fitness: {np.sum(F)}")
    plt.ion()
    iteration = 1
    step_ind = 0.5
    step_vol = 0.1
    for _ in range(MAX_ITER):
        W, P, F = run_fish_school(W, P, F, step_ind, step_vol)
        step_ind -= step_ind / MAX_ITER
        step_vol -= step_vol / MAX_ITER
        # Update plot
        plt.cla()
        plot_landscape(X, Y, Z)
        plot_school(W, P)
        plt.title(f"Total fitness: {np.sum(F):.2f}")
        plt.draw()
        plt.pause(0.01)
        iteration += 1
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    X, Y, Z = random_meshgrid()
    main()
