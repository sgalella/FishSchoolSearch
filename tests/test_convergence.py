import unittest

import fish_school_search as fs

limits = [-5, 5, -3, 3]
resolution = 100
num_iterations = 500
num_individuals = 30


def run_search(algorithm, num_iterations=100):
    for _ in range(num_iterations):
        algorithm.update(num_iterations)


class TestConvergence(unittest.TestCase):

    def test_sphere(self):
        landscape = fs.SphereLandscape(limits, resolution)
        search = fs.FishSchoolSearch(landscape, num_individuals)
        run_search(search, num_iterations)
        self.assertAlmostEqual(search.best_fitness, 1)

    def test_grickwank(self):
        landscape = fs.GrickwankLandscape(limits, resolution)
        search = fs.FishSchoolSearch(landscape, num_individuals)
        run_search(search, num_iterations)
        self.assertAlmostEqual(search.best_fitness, 1)

    def test_himmelblau(self):
        landscape = fs.HimmelblauLandscape(limits, resolution)
        search = fs.FishSchoolSearch(landscape, num_individuals)
        run_search(search, num_iterations)
        self.assertAlmostEqual(search.best_fitness, 1)

    def test_ackley(self):
        landscape = fs.AckleyLandscape(limits, resolution)
        search = fs.FishSchoolSearch(landscape, num_individuals)
        run_search(search, num_iterations)
        self.assertAlmostEqual(search.best_fitness, 1)

    def test_rastringin(self):
        landscape = fs.RastringinLandscape(limits, resolution)
        search = fs.FishSchoolSearch(landscape, num_individuals)
        run_search(search, num_iterations)
        self.assertAlmostEqual(search.best_fitness, 1)


if __name__ == '__main__':
    unittest.main()
