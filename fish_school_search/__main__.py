import numpy as np

import optimization_functions as opt
from fish_school_search import FishSchoolSearch
from visualization import VisualizeSearch


np.random.seed(1234)
resolution = 100
limits = [-5, 5, -3, 3]  # x_min, x_max, y_min, y_max
landscape = opt.SphereLandscape(limits, resolution)
search = FishSchoolSearch(landscape, num_individuals=10)
VisualizeSearch.show_all(search, num_iterations=20)
