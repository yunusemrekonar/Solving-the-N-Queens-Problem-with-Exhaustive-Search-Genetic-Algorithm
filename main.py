import random
import time

# Depth-First Search (DFS) solution
def dfs(n):
    solution = []

    def solve(row):
        if row == n:
            return solution[:]
        for col in range(n):
            if is_safe(row, col, solution):
                solution.append(col)
                result = solve(row + 1)
                if result:
                    return result
                solution.pop()
        return None

    def is_safe(row, col, solution):
        for r, c in enumerate(solution):
            if c == col or abs(r - row) == abs(c - col):
                return False
        return True

    return solve(0)

# Genetic Algorithm (GA) solution
def fitness_function(individual, n):
    clashes = 0
    row_col_clashes = abs(len(individual) - len(set(individual)))
    clashes += row_col_clashes
  
    for i in range(len(individual)):
        for j in range(i + 1, len(individual)):
            if abs(individual[i] - individual[j]) == abs(i - j):
                clashes += 1
    return n - clashes  

def create_population(size, n):
    return [random.sample(range(n), n) for _ in range(size)]

def select_parents(population, fitness_scores):
    probabilities = [score / sum(fitness_scores) for score in fitness_scores]
    return random.choices(population, probabilities, k=2)

def crossover(parent1, parent2, n):
    crossover_point = random.randint(0, n - 1)
    child = parent1[:crossover_point] + [gene for gene in parent2 if gene not in parent1[:crossover_point]]
    return child

def mutate(individual, mutation_rate, n):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(n), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

def genetic_algorithm(n, population_size=100, generations=1000, mutation_rate=0.05):
    population = create_population(population_size, n)
    
    for generation in range(generations):
        fitness_scores = [fitness_function(ind, n) for ind in population]
        
        if n in fitness_scores:
            solution = population[fitness_scores.index(n)]
            return solution, generation
        
        next_generation = []
        for _ in range(population_size):
            parent1, parent2 = select_parents(population, fitness_scores)
            child = crossover(parent1, parent2, n)
            child = mutate(child, mutation_rate, n)
            next_generation.append(child)        
        population = next_generation  
    return None, generations  

# Functions to print solution board
def print_board(solution, n):
    print("\nSolution found:")
    for row in range(n):
        board_row = ['Q' if solution[row] == col else '.' for col in range(n)]
        print(" ".join(board_row))
    print("\n" + "-" * (2 * n - 1))

# Testing functions
def test_genetic_algorithm(n):
    print(f"Testing Genetic Algorithm for N={n}...")
    start_time = time.time()
    solution, generations = genetic_algorithm(n)
    end_time = time.time()

    if solution:
        print(f"- Solution found in generation {generations}")
        print_board(solution, n)
    else:
        print("- No Solution Found")
    print(f"- Time: {end_time - start_time:.2f}s\n")


def test_exhaustive_search(n):
    print(f"Testing Exhaustive Search for N={n}...")
    start_time = time.time()
    solution = dfs(n)
    end_time = time.time()

    if solution:
        print(f"- Solution found!")
        print_board(solution, n)
    else:
        print("- No Solution Found")
    print(f"- Time: {end_time - start_time:.2f}s\n")


def run_tests():
    test_sizes = [10, 50, 100]
    
    for n in test_sizes:
        test_genetic_algorithm(n)
        test_exhaustive_search(n)

if __name__ == "__main__":
    run_tests()
