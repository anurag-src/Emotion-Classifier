# -*- coding: utf-8 -*-
"""Genetic-Algorithm

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rhdsptO7OUIhxoJ40f3Ri27La8EBeDmc
"""

import random
import numpy as np
import pandas as pd
from ANN import ANN
import argparse


class GA:

    def __init__(self, num_pop_size, dataset_path):

        self.NeuralNet = ANN(dataset_path)
        
        self.num_features = self.NeuralNet.x_train.shape[1]
        self.crossover_threshold = 5
        self.num_pop_size = num_pop_size
        self.population = np.empty([self.num_pop_size, self.num_features])
        self.fitness = None
        self.best_parents = []

    def generate_random(self):
        """
            generate_random returns a 2d array filled with zero and ones
            where the row represents the genomes and the column a particular
            characteristic

        """

        for i in range(0, self.num_pop_size):
            n = random.randint(0, self.num_features)
            m = self.num_features - n
            a = np.array([1] * n + [0] * m)
            np.random.shuffle(a)
            self.population[i] = a

    def selection(self):
        """
            fitness : contains the fitness of each of the genome

            The function discards the k worst genome based on the fitness
            returns an array of genome which can be used for cross over and mutation

            we have used roulette wheel selection method
            but for it's implementation we have used the
            stochastic acceptance method

            https://arxiv.org/abs/1109.3627

        """

        n = self.population.shape[0]

        arr = np.empty(self.population.shape)

        i = 0
        temp = [True, False]

        # print(fitness)

        while i < n:

            tmp = random.randint(0, n - 1)
            prob = [self.fitness[tmp], 1 - self.fitness[tmp]]
            choice = np.random.choice(temp, p=prob)

            if choice:
                arr[i] = self.population[tmp]
                i += 1

        self.population = arr

    def normalization(self, fitness):
        """
            :param fitness: list of Neural Net fitness
            :return: normalised fitness

            return normalised weight
            new weight = old_weight/max_weight

        """

        n = len(fitness)
        n_sum = sum(fitness)

        for i in range(0, n):
            fitness[i] = fitness[i] / n_sum
        return fitness

    def crossover(self, crossover_threshold):
        """
            :param crossover_threshold: the number of crossovers that will happen
                                        to generate the children

            crossover function implements the crossover part of GA,
            it takes two arguments of parents and crossover_threshold and
            returns the children with crossover done. In case the number of parents
            is odd, it takes a random parent for a pair.

        """

        parent_count, feature_count = self.population.shape
        odd_parent = False

        parents = self.population

        if parent_count % 2 != 0:
            # parents = np.vstack([self.population, self.population[random.randint(0, parent_count - 1)]])
            parents = np.concatenate((parents, [parents[random.randint(0, parent_count - 1)]]))
            odd_parent = True

        if crossover_threshold > feature_count:
            raise Exception("Points for Crossover greater than maximum possible!")

        elif crossover_threshold < 1:
            raise Exception("Points for Crossover should be greater than 0!")

        crossover_points = set()
        while len(crossover_points) != crossover_threshold:
            crossover_points.add(random.randint(0, feature_count - 1))

        crossover_points = list(crossover_points)
        crossover_points.sort()

        for i in range(0, int((parent_count + 1) / 2)):

            parent_a = parents[(2 * i)]
            parent_b = parents[((2 * i) + 1)]

            for j in crossover_points:
                tmp = parent_a.copy()
                parent_a[j:] = parent_b[j:]
                parent_b[j:] = tmp[j:]

            parents[2 * i] = parent_a
            parents[2 * i + 1] = parent_b

        if odd_parent:
            parents = np.delete(parents, parent_count, 0)

        self.population = parents

    def mutate(self, mutation_count=1):
        """
            :param mutation_count: number of mutation a parent will have

            mutation function some of the genes in  every genome

        """

        if mutation_count > self.num_features:
            raise Exception("Points for Crossover greater than maximum possible!")
        elif mutation_count < 1:
            raise Exception(
                "No mutation is useless, don't call the function unnecessary! Else the BOGEYMAN will get you")

        parents = self.population

        for i in parents:

            mutation_points = set()

            while len(mutation_points) != mutation_count:
                mutation_points.add(random.randint(0, self.num_features - 1))

            for j in mutation_points:
                i[j] = 1 if i[j] == 0 else 0

        self.population = parents

    def store_best_parents(self):
        df = pd.DataFrame(self.best_parents)
        df.to_csv("best_parents.csv")

    def get_current_bestfit(self):

        self.fitness = self.NeuralNet.get_fitness(self.population)
        self.fitness = self.normalization(self.fitness)

        best_par = self.fitness.index(max(self.fitness))
        num_best_par_features = sum(self.population[best_par])

        self.best_parents.append(self.population[best_par])

        return max(self.fitness), num_best_par_features
        
    def populate_onegen(self):
        
        self.selection()
        self.crossover(self.crossover_threshold)
        self.mutate()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("-p", "--path_dataset", required=True, type=str, help="Path to the feature CSV")
    args.add_argument("-n", "--num_pop", default=10, help="Population size", type=int)
    args = vars(args.parse_args())

    num_generations = 10

    genAlg = GA(args["num_pop"], args["path_dataset"])
    print("==== Creating initial population ====\n")
    genAlg.generate_random()
    print("==== Training initial population ====\n")
    genAlg.get_current_bestfit()

    num_features = []

    for _ in range(num_generations):

        print("\n==== Training one generation of parents ====\n")

        genAlg.populate_onegen()
        curr_bestfit, best_num_features = genAlg.get_current_bestfit()
        print("\nNumber of features in best parent: {}".format(best_num_features))
        print("Normalised loss of best parent:    {}".format(curr_bestfit))

        num_features.append(best_num_features)

    print("\n==== Variation of best features across all generations ====")
    print(*num_features)

    genAlg.store_best_parents()



