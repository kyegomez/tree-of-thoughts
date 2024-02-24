
# TreeofThoughts Documentation

## Overview

The `TreeofThoughts` library is a Python-based framework that uses depth-first search (DFS) algorithms for generative modeling. It is specifically designed for applications that require the generation of coherent and contextually relevant thoughts or states based on provided initial prompts. The library is most applicable in areas such as AI-driven creative writing, computational creativity, and automated reasoning.

## Class Definition

### DFS (`TreeofThoughts` subclass)

The `DFS` class inherits from the `TreeofThoughts` base class and implements a depth-first search algorithm to generate a sequence of thoughts based on a given prompt.

#### Constructor Arguments

| Argument             | Type             | Description                                                        | Default Value |
|----------------------|------------------|--------------------------------------------------------------------|---------------|
| `initial_prompt`     | `str`            | The initial prompt used to start the thought generation process.   | `None`        |
| `num_thoughts`       | `int`            | The number of thoughts to generate at each step in the process.    | `None`        |
| `max_steps`          | `int`            | The maximum number of steps in the thought generation process.     | `4`           |
| `value_threshold`    | `float`          | The minimum value a thought must have to be considered viable.     | `0.9`         |
| `pruning_threshold`  | `float`          | The cutoff value to prune less promising thoughts during search.   | `0.5`         |

#### Methods

| Method        | Description                                                |
|---------------|------------------------------------------------------------|
| `solve`       | Executes the depth-first search algorithm to find a sequence of thoughts with high evaluation values based on the provided prompt. Returns a final solution or the best thought based on the value threshold. |

### Usage Examples

#### Example 1: Initializing and Using DFS

```python
from treeofthoughts import DFS

# Assuming TreeofThoughts model implementation
class TreeofThoughts:
    def generate_thoughts(self, state, num_thoughts, initial_prompt):
        pass
    
    def evaluate_states(self, states, initial_prompt):
        pass
    
    def generate_solution(self, initial_prompt, best_state):
        pass

# Create DFS instance with the TreeofThoughts base class
thoughts_dfs = DFS()

# Call the solve method with initial arguments
solution = thoughts_dfs.solve(
    initial_prompt="Seed of ideas?",
    num_thoughts=5,
    max_steps=5,
    value_threshold=0.95,
    pruning_threshold=0.7
)

print(f"Solution: {solution}")
```

#### Example 2: Handling Errors in DFS

```python
from treeofthoughts import DFS
import logging

# Configure logger
logger = logging.getLogger()
logging.basicConfig(level=logging.ERROR)

# Use DFS class as described before
thoughts_dfs = DFS()
initial_prompt = "Beginning of a creative story."

try:
    solution = thoughts_dfs.solve(initial_prompt=initial_prompt)
    print(solution)
except Exception as e:
    logger.error(f"DFS Error: {e}")
```

#### Example 3: Extending DFS for Custom Functionality

```python
from treeofthoughts import DFS

# Custom model class that adds new methods to assess thoughts
class CustomModel(TreeofThoughts):
    # Define additional methods or override existing ones here
    pass

# Instantiate custom model and use it with DFS
custom_thoughts_dfs = DFS(CustomModel())
custom_solution = custom_thoughts_dfs.solve(
    "Exploring the depths of consciousness.",
    num_thoughts=10
)

print(f"Custom Solution: {custom_solution}")
```
