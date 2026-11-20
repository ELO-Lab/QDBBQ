import json
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from tqdm import trange

from openelm.configs import CVTMAPElitesConfig, MAPElitesConfig, QDConfig, EnvConfig
from openelm.environments import BaseEnvironment, Genotype
from openelm.algorithms.map_elites import Map


Phenotype = Optional[np.ndarray]
MapIndex = Optional[tuple]
Individual = Tuple[np.ndarray, float]


def recalculate_cossim_matrix(embeddings, new_embeddings):
    full_embeddings = np.concat([embeddings, new_embeddings])
    
    # normalize each vector
    norms = np.linalg.norm(full_embeddings, axis=1, keepdims=True)
    normalized = full_embeddings / norms

    # cosine similarity matrix (N x N)
    cos_sim_matrix = normalized @ normalized.T

    return cos_sim_matrix

def calculate_cossim_matrix(embeddings):
    full_embeddings = embeddings
    
    # normalize each vector
    norms = np.linalg.norm(full_embeddings, axis=1, keepdims=True)
    normalized = full_embeddings / norms

    # cosine similarity matrix (N x N)
    cos_sim_matrix = normalized @ normalized.T

    return cos_sim_matrix

def get_diversity_scores(cos_sim_matrix, k=3):
    
    if cos_sim_matrix.shape[0] < k:
        k = cos_sim_matrix.shape[0]

    # remove self-similarity by setting diagonal to -inf
    # np.fill_diagonal(cos_sim_matrix, -np.inf)

    # get top-k similarities per row
    topk = np.sort(cos_sim_matrix, axis=1)[:, -k-1:-1]

    # mean over k neighbors
    mean_k_sim = topk.mean(axis=1)

    return 1 - mean_k_sim


class Pool:
    """The pool stores a set of solutions or individuals."""

    def __init__(self, pool_size: int):
        """Initializes an empty pool.

        Args:
            pool_size (int): The number of solutions to store in the pool.
            history_length (int): The number of historical solutions
                to maintain in the pool.
        """
        self.pool_size = pool_size
        self.pool = []

    def add(self, solution, fitness):
        """Adds a solution to the pool.

        If the pool is full, the oldest solution is removed. The solution
        is also added to the history.

        Args:
            solution: The solution to add to the pool.
        """
        # if there are not any individual yet, add it to the pool
        if len(self.pool) < self.pool_size:
            self.pool.append((solution, fitness))
            self.pool.sort(key=lambda x: x[1], reverse=True)
            return

        # if new fitness is better than the worst, add it to the pool
        self.pool.sort(key=lambda x: x[1], reverse=True)
        if fitness > self.pool[-1][1]:
            if len(self.pool) >= self.pool_size:
                self.pool.pop(len(self.pool) - 1)
            self.pool.append((solution, fitness))
            # sort the pool by fitness
            self.pool.sort(key=lambda x: x[1], reverse=True)


class GABase:
    """
    Base class for a genetic algorithm
    """

    def __init__(
        self,
        env,
        config: QDConfig,
        env_config,
    ):
        """
        The base class for a genetic algorithm, implementing common functions and search.

        Args:
            env (BaseEnvironment): The environment to evaluate solutions in. This
            should be a subclass of `BaseEnvironment`, and should implement
            methods to generate random solutions, mutate existing solutions,
            and evaluate solutions for their fitness in the environment.
            config (QDConfig): The configuration for the algorithm.
            init_pool (Pool, optional): A pool to use for the algorithm. If not passed,
            a new pool will be created. Defaults to None.
        """
        self.env: BaseEnvironment = env
        self.config: QDConfig = config
        self.env_config = env_config
        self.save_history = self.config.save_history
        self.save_snapshot_interval = self.config.save_snapshot_interval
        self.history_length = self.config.history_length
        self.start_step = 0
        self.save_np_rng_state = self.config.save_np_rng_state
        self.load_np_rng_state = self.config.load_np_rng_state
        self.rng = np.random.default_rng(self.config.seed)
        self.rng_generators = None
        self.start_step = 0

        # self.history will be set/reset each time when calling `.search(...)`
        self.history: dict = defaultdict(list)
        self.fitness_history: dict = defaultdict(list)

        self._init_pool()


        os.makedirs(self.config.output_dir, exist_ok=True)
        header_str = ""
        for i in range(self.config.n_islands):
            header_str = header_str + f"island_{i}_max_bias, island_{i}_mean_bias, island_{i}_min_bias, island_{i}_sum_bias, island_{i}_max_div, island_{i}_mean_div, island_{i}_min_div, island_{i}_sum_div, island_{i}_max_fitness, island_{i}_mean_fitness, island_{i}_min_fitness, island_{i}_sum_fitness,"
        header_str = header_str + f"avg_max_bias, avg_mean_bias, avg_min_bias, avg_sum_bias, avg_max_div, avg_mean_div, avg_min_div, avg_sum_div, avg_max_fitness, avg_mean_fitness, avg_min_fitness, avg_sum_fitness\n"
        with open((self.config.output_dir + "results.csv"), "w") as f:
            f.write(header_str)


    def _init_pool(self):
        self.pools = [Pool(self.config.pool_size) for _ in range(self.config.n_islands)]

        if self.config.ga:
            self.ga_pool = Pool(self.config.ga_pool_size)
        print("Loading finished")

    def random_selection(self, island: int) -> MapIndex:
        """Randomly select a niche (cell) in the map that has been explored."""
        return random.choice(self.pools[island].pool)
    
    def random_selection_ga(self) -> MapIndex:
        """Randomly select a niche (cell) in the map that has been explored."""
        return random.choice(self.ga_pool.pool)

    def search(self, init_steps: int, total_steps: int, resume_dataset=[]) -> str:
        """
        Run the genetic algorithm.

        Args:
            initsteps (int): Number of initial random solutions to generate.
            totalsteps (int): Total number of steps to run the algorithm for,
                including initial steps.
            atol (float, optional): Tolerance for how close the best performing
                solution has to be to the maximum possible fitness before the
                search stops early. Defaults to 1.

        Returns:
            str: A string representation of the best perfoming solution. The
                best performing solution object can be accessed via the
                `current_max_genome` class attribute.
        """

        if self.save_history:
            self.history = defaultdict(list)

        start_step = int(self.start_step)
        total_steps = int(total_steps)
        tbar = trange(start_step, total_steps, initial=start_step, total=total_steps)


        # import pandas as pd
        # new_path = self.config.output_dir.replace("relogs", "new_logs")
        # dataset = pd.read_csv(new_path + "/genomes.csv")

        # resume_dataset = dataset["task_str"].values
        # score_dataset = dataset["bias_score"].values

        # for n_steps, resume_data in enumerate(resume_dataset):
        #     new_individuals = self.env.sample_from_dataset(dataset=[resume_data], bias_score=score_dataset[n_steps], mode="normal")
        #     for individual in new_individuals:
        #         island_idx = individual.target_bias_idx
        #         # Evaluate fitness
        #         island_embeddings = np.array([indv[0].embedding for indv in self.pools[island_idx].pool])

        #         if len(island_embeddings) == 0:
        #             fitness = 0.0
        #         else:
        #             cos_sim_matrix = recalculate_cossim_matrix(island_embeddings, [individual.embedding])
        #             context_diversity = get_diversity_scores(cos_sim_matrix)

        #             for idx, indv in enumerate(self.pools[island_idx].pool):
        #                 new_fitness = indv[0].evaluate_fitness(context_diversity[idx])
        #                 self.pools[island_idx].pool[idx] = (indv[0], new_fitness)
        #             fitness = self.env.fitness(individual, context_diversity[idx + 1])
        #         self.pools[island_idx].add(individual, fitness)
            
        #     if n_steps % 10 == 0:
        #         avg_sum_bias, avg_sum_diversity, avg_sum_fitness = self.save_results(step=n_steps)
        #         tbar.set_description(f"{avg_sum_bias=:.4f}, {avg_sum_diversity=:.4f}, {avg_sum_fitness=:.4f}")

        for n_steps in tbar:

            for island in range(self.config.n_islands):


                if self.config.ga:
                    if n_steps < len(resume_dataset):
                        print("Get task from resume dataset")
                        new_individuals = self.env.sample_from_dataset(dataset=resume_dataset, mode="normal")

                    elif n_steps < init_steps:
                        # Initialise by generating initsteps random solutions
                        new_individuals: list[Genotype] = self.env.random(source_island=island, ga=True)
                    else:
                        new_individuals = []
                        while len(new_individuals) == 0:
                            try:
                                batch: list[Genotype] = []
                                if self.config.crossover:
                                    crossover_parents = []
                                    for i in range(self.config.crossover_parents):
                                        item = self.random_selection_ga()
                                        crossover_parents.append(item[0])
                                    batch.append(crossover_parents)
                                else:
                                    for _ in range(self.env.batch_size):
                                        item = self.random_selection_ga()
                                        batch.append(item)
                                    # Mutate
                                    batch = [indv[0] for indv in batch]
                                # Mutate the elite.
                                new_individuals = self.env.mutate(batch, ga=True)

                            except Exception as e:
                                print(f"Error: {e}. Retrying...")

                    for individual in new_individuals:
                        # Evaluate for traditional GA pool
                        if n_steps >= len(resume_dataset):
                            # Evaluate fitness
                            ga_embeddings = np.array([indv[0].embedding for indv in self.ga_pool.pool])

                            if len(ga_embeddings) == 0:
                                fitness = self.env.fitness(individual, context_diversity=None)
                            else:
                                cos_sim_matrix = recalculate_cossim_matrix(ga_embeddings, [individual.embedding])
                                context_diversity = get_diversity_scores(cos_sim_matrix)

                                for idx, indv in enumerate(self.ga_pool.pool):
                                    new_fitness = indv[0].evaluate_fitness(context_diversity[idx])
                                    self.ga_pool.pool[idx] = (indv[0], new_fitness)

                                fitness = self.env.fitness(individual, context_diversity[idx + 1])

                            if np.isinf(fitness):
                                continue
                        else:
                            fitness = individual.bias_score
                        
                        self.ga_pool.add(individual, fitness)

                        
                        # Evaluate for island model
                        if n_steps >= len(resume_dataset):

                            island_idx = individual.target_bias_idx
                            # Evaluate fitness
                            island_embeddings = np.array([indv[0].embedding for indv in self.pools[island_idx].pool])

                            if len(island_embeddings) == 0:
                                fitness = 0.0
                            else:
                                cos_sim_matrix = recalculate_cossim_matrix(island_embeddings, [individual.embedding])
                                context_diversity = get_diversity_scores(cos_sim_matrix)

                                for idx, indv in enumerate(self.pools[island_idx].pool):
                                    new_fitness = indv[0].evaluate_fitness(context_diversity[idx])
                                    self.pools[island_idx].pool[idx] = (indv[0], new_fitness)


                            if np.isinf(fitness):
                                continue
                        else:
                            fitness = individual.bias_score
                        
                        self.pools[island_idx].add(individual, fitness)


                else: 
                    if n_steps % self.config.migration_interval == 0 and n_steps != 0:
                        # migrate the best solution from each island to a random island

                        # exclude the source island from the random selection of target island for migration
                        target_island = random.randint(0, self.config.n_islands - 1)


                        new_individuals = []
                        while len(new_individuals) == 0:
                            try:
                                batch: list[Genotype] = []
                                if self.config.crossover:
                                    crossover_parents = []
                                    for i in range(self.config.crossover_parents):
                                        item = self.random_selection(island)
                                        crossover_parents.append(item[0])
                                    batch.append(crossover_parents)
                                else:
                                    for _ in range(self.env.batch_size):
                                        item = self.random_selection(island)
                                        batch.append(item)
                                    # Mutate
                                    batch = [indv[0] for indv in batch]
                                # Mutate the elite.
                                new_individuals = self.env.migrate(batch, target_island=target_island)

                            except Exception as e:
                                print(f"Error: {e}. Retrying...")

                        for individual in new_individuals:
                            if n_steps >= len(resume_dataset):
                                # Evaluate fitness
                                island_embeddings = np.array([indv[0].embedding for indv in self.pools[target_island].pool])

                                if len(island_embeddings) == 0:
                                    fitness = self.env.fitness(individual, context_diversity=None)
                                else:
                                    cos_sim_matrix = recalculate_cossim_matrix(island_embeddings, [individual.embedding])
                                    context_diversity = get_diversity_scores(cos_sim_matrix)

                                    for idx, indv in enumerate(self.pools[target_island].pool):
                                        new_fitness = indv[0].evaluate_fitness(context_diversity[idx])
                                        self.pools[target_island].pool[idx] = (indv[0], new_fitness)

                                    if self.config.no_diversity:
                                        fitness = self.env.fitness(individual, -1)
                                    else:
                                        fitness = self.env.fitness(individual, context_diversity[idx + 1])

                                if np.isinf(fitness):
                                    continue
                            else:
                                fitness = individual.bias_score
                            
                            self.pools[target_island].add(individual, fitness)
                    else:
                        if n_steps < len(resume_dataset):
                            print("Get task from resume dataset")
                            new_individuals = self.env.sample_from_dataset(dataset=resume_dataset, mode="normal")

                        elif n_steps < init_steps:
                            # Initialise by generating initsteps random solutions
                            new_individuals: list[Genotype] = self.env.random(source_island=island)
                        else:
                            new_individuals = []
                            while len(new_individuals) == 0:
                                try:
                                    batch: list[Genotype] = []
                                    if self.config.crossover:
                                        crossover_parents = []
                                        for i in range(self.config.crossover_parents):
                                            item = self.random_selection(island)
                                            crossover_parents.append(item[0])
                                        batch.append(crossover_parents)
                                    else:
                                        for _ in range(self.env.batch_size):
                                            item = self.random_selection(island)
                                            batch.append(item)
                                        # Mutate
                                        batch = [indv[0] for indv in batch]
                                    # Mutate the elite.
                                    new_individuals = self.env.mutate(batch, source_island=island)

                                except Exception as e:
                                    print(f"Error: {e}. Retrying...")

                        for individual in new_individuals:
                            if n_steps >= len(resume_dataset):
                                # Evaluate fitness
                                island_embeddings = np.array([indv[0].embedding for indv in self.pools[island].pool])

                                if len(island_embeddings) == 0:
                                    fitness = self.env.fitness(individual, context_diversity=None)
                                else:
                                    cos_sim_matrix = recalculate_cossim_matrix(island_embeddings, [individual.embedding])
                                    context_diversity = get_diversity_scores(cos_sim_matrix)

                                    for idx, indv in enumerate(self.pools[island].pool):
                                        new_fitness = indv[0].evaluate_fitness(context_diversity[idx])
                                        self.pools[island].pool[idx] = (indv[0], new_fitness)


                                    if self.config.no_diversity:
                                        fitness = self.env.fitness(individual, -1)
                                    else:
                                        fitness = self.env.fitness(individual, context_diversity[idx + 1])

                                if np.isinf(fitness):
                                    continue
                            else:
                                fitness = individual.bias_score
                            
                            self.pools[island].add(individual, fitness)
                
            avg_sum_bias, avg_sum_diversity, avg_sum_fitness = self.save_results(step=n_steps)
            tbar.set_description(f"{avg_sum_bias=:.4f}, {avg_sum_diversity=:.4f}, {avg_sum_fitness=:.4f}")


        return ""
    
    def max_fitness(self):
        """Get the maximum fitness value in the map."""
        return self.fitnesses.max_finite

    def mean_fitness(self):
        """Get the mean fitness value in the map."""
        return self.fitnesses.mean

    def min_fitness(self):
        """Get the minimum fitness value in the map."""
        return self.fitnesses.min_finite


    def save_results(self, step: int):

        # create folder for dumping results and metadata
        output_folder = Path(self.config.output_dir + f"overall") / f"step_{step}"
        os.makedirs(output_folder, exist_ok=True)

        if step % 10 == 0:

            # Save maps as pickle file
            try:
                with open((output_folder / "pools.pkl"), "wb") as f:
                    pickle.dump(self.pools, f)
            except Exception:
                pass
            if self.save_history:
                with open((output_folder / "history.pkl"), "wb") as f:
                    pickle.dump(self.history, f)

            with open((output_folder / "fitness_history.pkl"), "wb") as f:
                pickle.dump(self.fitness_history, f)

        # Save results to csv (max fitness, mean fitness, min fitness of each island, max, mean, min, sum of max for entire pools) for each generation
        all_max_divs = []
        all_max_fits = []
        all_max_biases = []

        all_mean_divs = []
        all_mean_fits = []
        all_mean_biases = []

        all_min_divs = []
        all_min_fits = []
        all_min_biases = []

        all_sum_divs = []
        all_sum_fits = []
        all_sum_biases = []

        result_str = ""

        for island in range(self.config.n_islands):
        
            embeddings = np.array([indv[0].embedding for indv in self.pools[island].pool])
            biases = [indv[0].bias_score for indv in self.pools[island].pool]

            if len(embeddings) <= 1:
                divs = [0.0]
                fits = [0.0]
            else:
                cos_sim_matrix = calculate_cossim_matrix(embeddings)
                context_diversity = get_diversity_scores(cos_sim_matrix)

                # fits = []
                # for idx, indv in enumerate(self.pools[island].pool):
                #     new_fitness = indv[0].evaluate_fitness(context_diversity[idx])
                #     self.pools[island].pool[idx] = (indv[0], new_fitness)
                #     fits.append(new_fitness)
                divs = context_diversity
                fits = np.array(biases) * context_diversity

            if len(biases) < 1: biases.append(0.0)

            result_str = result_str + str(np.round(np.max(biases), 2)) + "," + str(np.round(np.mean(biases), 2)) + "," + str(np.round(np.min(biases), 2))+ "," + str(np.round(np.sum(biases), 2)) + ","
            result_str = result_str + str(np.round(np.max(divs), 2)) + "," + str(np.round(np.mean(divs), 2)) + "," + str(np.round(np.min(divs), 2))+ "," + str(np.round(np.sum(divs), 2)) + ","
            result_str = result_str + str(np.round(np.max(fits), 2)) + "," + str(np.round(np.mean(fits), 2)) + "," + str(np.round(np.min(fits), 2))+ "," + str(np.round(np.sum(fits), 2)) + ","
            
            all_max_divs.append(np.round(np.max(divs), 2))
            all_max_fits.append(np.round(np.max(fits), 2))
            all_max_biases.append(np.round(np.max(biases), 2))

            all_mean_divs.append(np.round(np.mean(divs), 2))
            all_mean_fits.append(np.round(np.mean(fits), 2))
            all_mean_biases.append(np.round(np.mean(biases), 2))

            all_min_divs.append(np.round(np.min(divs), 2))
            all_min_fits.append(np.round(np.min(fits), 2))
            all_min_biases.append(np.round(np.min(biases), 2))

            all_sum_divs.append(np.round(np.sum(divs), 2))
            all_sum_fits.append(np.round(np.sum(fits), 2))
            all_sum_biases.append(np.round(np.sum(biases), 2))
        
        result_str = result_str + str(np.round(np.mean(all_max_biases), 2)) + "," + str(np.round(np.mean(all_mean_biases), 2)) + "," + str(np.round(np.mean(all_min_biases), 2))+ "," + str(np.round(np.mean(all_sum_biases), 2)) + ","
        result_str = result_str + str(np.round(np.mean(all_max_divs), 2)) + "," + str(np.round(np.mean(all_mean_divs), 2)) + "," + str(np.round(np.mean(all_min_divs), 2))+ "," + str(np.round(np.mean(all_sum_divs), 2)) + ","
        result_str = result_str + str(np.round(np.mean(all_max_fits), 2)) + "," + str(np.round(np.mean(all_mean_fits), 2)) + "," + str(np.round(np.mean(all_min_fits), 2))+ "," + str(np.round(np.mean(all_sum_fits), 2)) + "\n"

        with open((self.config.output_dir + "results.csv"), "a") as f:
            f.write(result_str)

        # save env_name to check later, for verifying correctness of environment to run with snapshot load
        tmp_config = dict()
        tmp_config["env_name"] = self.env.config.env_name

        with open((output_folder / "config.json"), "w") as f:
            json.dump(tmp_config, f)

        return np.round(np.mean(all_sum_biases), 2), np.round(np.mean(all_sum_divs), 2), np.round(np.mean(all_sum_fits), 2)
    
class GAIsland(GABase):
    def __init__(
        self,
        env,
        config: MAPElitesConfig,
        *args,
        **kwargs,
    ):
        """
        Class implementing MAP-Elites, a quality-diversity algorithm.

        Args:
            env (BaseEnvironment): The environment to evaluate solutions in. This
            should be a subclass of `BaseEnvironment`, and should implement
            methods to generate random solutions, mutate existing solutions,
            and evaluate solutions for their fitness in the environment.
            config (MAPElitesConfig): The configuration for the algorithm.
        """
        super().__init__(env=env, config=config, *args, **kwargs)
