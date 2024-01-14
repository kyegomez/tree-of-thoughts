# Comprehensive Documentation and Changelog
This document provides a comprehensive overview of the changes made to the TreeofThoughts class and its methods to improve readability and understandability. The changes include updating variable names to be more meaningful and descriptive, as well as modifying the structure of the code for better readability.

## Changelog
1. TreeofThoughts Class
Updated the class definition to include a more descriptive docstring.
2. __init__ Method
No changes were made to the __init__ method.
3. solve Method
Updated variable names:
x -> initial_prompt
k -> num_thoughts
T -> max_steps
b -> max_states
vth -> value_threshold
4. tot_bfs Method
Updated variable names:
x -> initial_prompt
k -> num_thoughts
T -> max_steps
b -> max_states
S0 -> current_states
S0_t -> generated_states
Vt -> state_values
St -> selected_states
5. tot_dfs Method
Updated variable names:

x -> initial_prompt
k -> num_thoughts
T -> max_steps
vth -> value_threshold
s -> state
t -> step
s_prime -> next_state
child -> child_state

### Added optional parameters for better control over the search process:
pruning_threshold
confidence_threshold
max_iterations
convergence_threshold
convergence_count
6. save_tree_to_json Method
No changes were made to the save_tree_to_json method.
7. print_tree Method
No changes were made to the print_tree method.

# Documentation
TreeofThoughts Class
The TreeofThoughts class is designed to solve problems using a tree-based search algorithm. It takes a model and a search algorithm (either 'BFS' or 'DFS') as input and provides methods to solve problems using the chosen algorithm.

## Initialization
The __init__ method initializes the TreeofThoughts class with the given model and search algorithm. It also initializes an empty tree structure to store the search results.

## Solve Method
The solve method is the main entry point for solving problems using the TreeofThoughts class. It takes the following parameters:

initial_prompt: The initial problem or prompt to be solved.
num_thoughts: The number of thoughts to generate at each step.
max_steps: The maximum number of steps to perform in the search.
max_states: The maximum number of states to consider at each step (for BFS).
value_threshold: The threshold value for pruning states (for DFS).
timeout: The maximum time allowed for the search process.
confidence_threshold: The confidence threshold for stopping the search.
max_iterations: The maximum number of iterations allowed for the search.
convergence_threshold: The threshold for determining convergence.
convergence_count: The number of consecutive convergences required to stop the search.
Based on the chosen search algorithm, the solve method calls either the tot_bfs or tot_dfs method to perform the search.

## tot_bfs Method
The tot_bfs method performs a breadth-first search to solve the problem. It takes the following parameters:

initial_prompt: The initial problem or prompt to be solved.
num_thoughts: The number of thoughts to generate at each step.
max_steps: The maximum number of steps to perform in the search.
max_states: The maximum number of states to consider at each step.
pruning_threshold: The threshold value for pruning states.
The method generates and evaluates states at each step, selecting the best states based on their values. The search continues until the maximum number of steps is reached, and the best state is returned.

## tot_dfs Method
The tot_dfs method performs a depth-first search to solve the problem. It takes the following parameters:

initial_prompt: The initial problem or prompt to be solved.
num_thoughts: The number of thoughts to generate at each step.
max_steps: The maximum number of steps to perform in the search.

value_threshold: The threshold value for pruning states.
pruning_threshold: The threshold value for pruning states based on their values.
confidence_threshold: The confidence threshold for stopping the search.
max_iterations: The maximum number of iterations allowed for the search.
convergence_threshold: The threshold for determining convergence.
convergence_count: The number of consecutive convergences required to stop the search.
The method uses a recursive depth-first search approach to explore the state space. It generates and evaluates states at each step, and if a state's value is above the value_threshold and pruning_threshold, it continues the search with the new state. The search stops when the maximum number of steps is reached, the confidence threshold is met, or the convergence criteria are satisfied. The best state is then returned.

## save_tree_to_json Method
The save_tree_to_json method saves the current tree structure and metrics to a JSON file. It takes the following parameter:

file_name: The name of the JSON file to save the tree structure and metrics.
This method is useful for logging the search process and analyzing the results later.

## print_tree Method
The print_tree method prints the tree structure in a human-readable format. It takes the following parameters:

node: The current node in the tree.
depth: The depth of the current node in the tree (default is 0).
This method is useful for visualizing the tree structure and understanding the search process.

## Usage
To use the TreeofThoughts class, follow these steps:

Initialize the class with a model and a search algorithm (either 'BFS' or 'DFS').
Call the solve method with the required parameters to perform the search and obtain the best state.
(Optional) Use the save_tree_to_json method to save the tree structure and metrics to a JSON file.
(Optional) Use the print_tree method to visualize the tree structure.
Here's an example of how to use the TreeofThoughts class:



# V2 with Monte Carlo, A* Search Algorithm, BFS, Best First Search
### Class: TreeofThoughts
This class represents the base class for the Tree of Thoughts search algorithm. It contains the following methods:

- `__init__(self, model)`: Initializes the TreeofThoughts object with the given model.
- `save_tree_to_json(self, file_name)`: Saves the tree to a JSON file with the given file name.
- `logNewState(self, state, evaluation)`: Logs a new state and its evaluation to the tree.
- `adjust_pruning_threshold_precentile(self, evaluated_thoughts, percentile)`: Adjusts the pruning threshold based on the percentile of evaluated thoughts.
- `adjust_pruning_threshold_moving_average(self, evaluated_thoughts, window_size)`: Adjusts the pruning threshold based on the moving average of evaluated thoughts.

### Class: TreeofThoughtsBFS
This class represents the Breadth-First Search (BFS) variant of the Tree of Thoughts search algorithm. It inherits from the TreeofThoughts class and contains the following method:

- `solve(self, initial_prompt, num_thoughts, max_steps, max_states, value_threshold, pruning_threshold=0.5)`: Solves the problem using BFS with the given parameters.

### Class: TreeofThoughtsDFS
This class represents the Depth-First Search (DFS) variant of the Tree of Thoughts search algorithm. It inherits from the TreeofThoughts class and contains the following method:

- `solve(self, initial_prompt, num_thoughts, max_steps, value_threshold, pruning_threshold=0.5)`: Solves the problem using DFS with the given parameters.

### Class: TreeofThoughtsBEST
This class represents the Best-First Search variant of the Tree of Thoughts search algorithm. It contains the following methods:

- `__init__(self, model)`: Initializes the TreeofThoughtsBEST object with the given model.
- `save_tree_to_json(self, file_name)`: Saves the tree to a JSON file with the given file name.
- `log_new_state(self, state, evaluation)`: Logs a new state and its evaluation to the tree.
- `solve(self, initial_prompt, num_thoughts, max_steps, pruning_threshold)`: Solves the problem using Best-First Search with the given parameters.

### Class: TreeofThoughtsASearch
This class represents the A* Search variant of the Tree of Thoughts search algorithm. It contains the following methods:

- `__init__(self, model)`: Initializes the TreeofThoughtsASearch object with the given model.
- `solve(self, initial_prompt, num_thoughts=5, max_steps=30, pruning_threshold=0.4)`: Solves the problem using A* Search with the given parameters.
- `is_goal(self, state, score)`: Determines if the given state is a goal state based on its score.
- `reconstruct_path(self, came_from, current_state, initial_prompt)`: Reconstructs the path from the initial state to the current state using the came_from dictionary.

### Class: MonteCarloTreeofThoughts
This class represents the Monte Carlo Tree Search variant of the Tree of Thoughts search algorithm. It inherits from the TreeofThoughts class and contains the following methods:

- `__init__(self, model, objective="balance")`: Initializes the MonteCarloTreeofThoughts object with the given model and objective.
- `optimize_params(self, num_thoughts, max_steps, max_states)`: Optimizes the search parameters based on the objective.
- `solve(self, initial_prompt, num_thoughts, max_steps, max_states, pruning_threshold)`: Solves the problem using

 Monte Carlo Tree Search with the given parameters.
- `monte_carlo_search(self, initial_prompt, num_thoughts, max_steps, max_states, pruning_threshold)`: Performs the Monte Carlo Tree Search with the given parameters.

### Class: OptimizedTreeofThoughts
This class represents an optimized version of the Tree of Thoughts search algorithm. It inherits from the TreeofThoughts class and contains the following method:

- `solve(self, x, k=None, T=None, b=None, vth=None, timeout=None, confidence_threshold=None, max_iterations=None, convergence_threshold=None, convergence_count=None)`: Solves the problem using an optimized search algorithm with the given parameters.