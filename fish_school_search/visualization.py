import matplotlib.pyplot as plt


class VisualizeSearch:
    @staticmethod
    def show_all(algorithm, num_iterations=100):
        """
        Shows the evolution of the solutions in the landscape.

        Args:
            algorithm (FishSchoolSearch): Fish school search initialized.
            num_iterations (int): Number of iterations to run the algorithm.
        """
        plt.figure(figsize=(8, 5))
        algorithm.plot()
        plt.colorbar(shrink=0.75)
        plt.ion()
        for _ in range(num_iterations):
            algorithm.update(num_iterations)
            plt.cla()
            algorithm.plot()
            plt.draw()
            plt.pause(0.01)
        plt.ioff()
        plt.show()

    @staticmethod
    def show_last(algorithm, num_iterations=100):
        """
        Shows the last evolution of the solutions in the landscape.

        Args:
            algorithm (FishSchoolSearch): Fish school search initialized.
            num_iterations (int): Number of iterations to run the algorithm.
        """
        for _ in range(num_iterations):
            algorithm.update(num_iterations)
        plt.figure(figsize=(8, 5))
        algorithm.plot()
        plt.colorbar(shrink=0.75)
        plt.show()
