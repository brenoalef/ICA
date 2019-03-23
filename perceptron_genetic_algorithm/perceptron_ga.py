import numpy as np
import random

class PerceptronGA:
    def __init__(self, population_size=50,  elite_size=10, mutation_rate=0.005, generations=250):
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.w = None

    def __initial_population(self, X):
        return [np.random.uniform(-1, 1, (X.shape[1], 1)) for i in range(0, self.population_size)]

    def __activation(self, individual, X):
        y_hat = X.dot(individual)
        y_hat[y_hat >= 0] = 1
        y_hat[y_hat != 1] = 0
        return y_hat

    def fitness(self, individual, X, y):
        y_hat = self.__activation(individual, X)
        return 1 - np.mean(np.abs(y_hat - y))

    def __sort_weights(self, population, X, y):
        fitnesses = {i: self.fitness(individual, X, y) for (i, individual) in enumerate(population)}
        return sorted(fitnesses.items(), key=lambda x:x[1], reverse=True)

    def __selection(self, sorted_pop):
        selected = []
        fitnesses = [y for (x, y) in sorted_pop]
        cumulative_freq = np.cumsum(fitnesses)/np.sum(fitnesses)
        for i in range(0, self.elite_size):
            selected.append(sorted_pop[i][0])
        for i in range(0, len(sorted_pop) - self.elite_size):
            pick = random.random()
            for i in range(0, len(sorted_pop)):
                if pick <= cumulative_freq[i]:
                    selected.append(sorted_pop[i][0])
                    break
        return selected

    def __mating_pool(self, population, selected):
        pool = []
        for i in range(0, len(selected)):
            pool.append(population[selected[i]])
        return pool

    def __breed(self, parent1, parent2):
        child = np.zeros((len(parent1), 1))
        
        gene_a = int(random.random()*len(parent1))
        gene_b = int(random.random()*len(parent1))

        start = min(gene_a, gene_b)
        end = max(gene_a, gene_b)

        child[0:start] = parent1[0:start]
        child[start:end] = parent2[start:end]
        child[end:] = parent1[end:]

        return child

    def __breed_population(self, matingpool):
        children = []
        pool = random.sample(matingpool, len(matingpool))

        for i in range(0, self.elite_size):
            children.append(matingpool[i])
        
        for i in range(0, len(matingpool) - self.elite_size):
            child = self.__breed(pool[i], pool[len(matingpool) - i - 1])
            children.append(child)
        return children

    def __mutate(self, individual, mutation_rate):
        for chromosome in range(len(individual)):
            if (random.random() < mutation_rate):
                to_swap = np.random.choice(np.linspace(-1, 80, num=5000))
                index=int(random.random()*len(individual[chromosome]))
                individual[chromosome][index] = to_swap
        return individual

    def __mutate_population(self, population):
        mutated = [self.__mutate(population[i], self.mutation_rate) for i in range(0, len(population))]
        return mutated

    def __next_generation(self, current_gen, X, y):
        sorted_pop = self.__sort_weights(current_gen, X, y)
        selected = self.__selection(sorted_pop)
        matingpool = self.__mating_pool(current_gen, selected)
        children = self.__breed_population(matingpool)
        next_gen = self.__mutate_population(children)
        return next_gen

    def fit(self, X, y):
        X = np.hstack((np.ones((X.shape[0],1)), X))
        pop = self.__initial_population(X)
        progress = []
        progress.append(1 / self.__sort_weights(pop, X, y)[0][1])
        for i in range(0, self.generations):
            pop = self.__next_generation(pop, X, y)
            progress.append(1 / self.__sort_weights(pop, X, y)[0][1])
        self.w = pop[self.__sort_weights(pop, X, y)[0][0]]
        return self.w, progress

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0],1)), X))
        return self.__activation(self.w, X)
