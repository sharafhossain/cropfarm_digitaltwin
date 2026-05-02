"""Student package for Project8 plant-growth GA assignment."""

from . import data_generation
from . import genetic_algorithm
from . import simulation

DEFAULT_TRUE_PARAMS = data_generation.DEFAULT_TRUE_PARAMS
create_forcing_signals = data_generation.create_forcing_signals
generate_dataset = data_generation.generate_dataset

GeneticAlgorithmEstimator = genetic_algorithm.GeneticAlgorithmEstimator
normalized_mse = genetic_algorithm.normalized_mse

build_grid_positions = simulation.build_grid_positions
simulate_heights = simulation.simulate_heights

__all__ = [
    "DEFAULT_TRUE_PARAMS",
    "create_forcing_signals",
    "generate_dataset",
    "GeneticAlgorithmEstimator",
    "normalized_mse",
    "build_grid_positions",
    "simulate_heights",
]
