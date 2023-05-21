# Tree of Thoughts 🌳🌲🌴🌿🍃

![tree of thoughts banner](tree-of-thoughts.jpeg)

Tree of Thoughts (ToT) is a powerful and flexible algorithm for leveraging pre-trained language models to solve various problems by exploring multiple reasoning paths. It's designed to be plug-and-play, allowing users to easily connect their models and use the Tree of Thoughts method.

## Getting started
Clone this repository with ```git clone https://github.com/kyegomez/tree-of-thoughts```

Navigate to the repository folder: ``` cd tree-of-thoughts```

```pip install openai transformers```

Create a Python script (e.g., example.py) and import the necessary classes:

``` 
from tree_of_thoughts import OpenAILanguageModel, CustomLanguageModel, TreeofThoughts

model = OpenAILanguageModel('api key')

#choose search algorithm('bfs or 'dfs)
search_algorithm = "BFS"

#cot or propose
strategy="cot"

# value or vote
evaluation_strategy = "value"

#create an instance of the tree of thoughts class
tree_of_thoughts= TreeofThoughts(model, search_algorithm)

input_problem = "What is 2 + 2"
k = 5
T = 3
b = 5
vth = 0.5

#call the solve method with the inpit problem and other params
solution = tree_of_thoughts.solve(input_problem, k, T, b, vth, )

#use the solution in your production environment
print(solution)
```

Or Integrate your own custom language model:

``` 
class CustomLanguageModel(AbstractLanguageModel):
    def __init__(self, model):
        self.model = model

    def generate_thoughts(self, state, k):
        #implement the thought generation logic using self.model
        pass

    def evaluate_states(self, states):
        #implement state evaluation logic using self.model
        pass

```


Run the example script

🌟 Features:
- General problem-solving framework for language models
- Supports both breadth-first search (BFS) and depth-first search (DFS) algorithms
- Easy integration with popular language models like OpenAI and Hugging Face
- Extensible and adaptable to different problem properties and resource constraints

## Algorithmic Pseudocode

1. Define the thought decomposition based on the problem properties.
2. Create a thought generator function G(pθ, s, k) with two strategies:
   a. Sample i.i.d. thoughts from a CoT prompt.
   b. Propose thoughts sequentially using a "propose prompt".
3. Create a state evaluator function V(pθ, S) with two strategies:
   a. Value each state independently.
   b. Vote across states.
4. Choose a search algorithm (BFS or DFS) based on the tree structure.
5. Implement the chosen search algorithm.
6. Execute the chosen search algorithm with the input problem, thought generator, state evaluator, and other required parameters.

## Tree of Thoughts Class
```
class TreeofThoughts:
    
    def __init__(self, model, search_algorithm):
        self.model = model
        self.search_algorithm = search_algorithm

    def solve(self, x, k, T, b, vth):
        if self.search_algorithm == 'BFS':
            return self.tot_bfs(x, k, T, b)
        elif self.search_algorithm == 'DFS':
            return self.tot_dfs(x, k, T, vth)
        else:
            raise ValueError("Invalid search algorithm. Choose 'BFS' or 'DFS'.")

    def tot_bfs(self, x, k, T, b):
        S0 = {x}
        for t in range(1, T + 1):
            S0_t = {(*s, z) for s in S0 for z in self.model.generate_thoughts(s, k)}
            Vt = self.model.evaluate_states(S0_t)
            St = sorted(S0_t, key=lambda s: Vt[s], reverse=True)[:b]
            S0 = set(St)
        return self.model.generate_thoughts(max(St, key=lambda s: Vt[s]), 1)

    def tot_dfs(self, x, k, T, vth):
        output = []

        def dfs(s, t):
            if t > T:
                output.append(self.model.generate_thoughts(s, 1))
                return
            for s_prime in sorted(self.model.generate_thoughts(s, k)):
                if self.model.evaluate_states({s_prime})[s_prime] > vth:
                    dfs((*s, s_prime), t + 1)

        dfs(x, 1)
        return output
    
```


## Usage Examples

### OpenAI API

To use Tree of Thoughts with OpenAI's API, create a custom model class that inherits from `AbstractLanguageModel` and implements the required methods using OpenAI's API. Then, create an instance of the `TreeOfThoughts` class with the custom model and the desired search algorithm ('BFS' or 'DFS').

### Hugging Face Transformers

To use Tree of Thoughts with Hugging Face Transformers, create a custom model class that inherits from `AbstractLanguageModel` and implements the required methods using Hugging Face Transformers. Then, create an instance of the `TreeOfThoughts` class with the custom model and the desired search algorithm ('BFS' or 'DFS').

## Roadmap

Provide ready to use generate thoughts function

Provide ready to use evaluate states function

now
Implement a more sophisticated prompt engineering strategy to guide the model's reasoning process more effectively.

Introduce a reinforcement learning approach to fine-tune the model based on feedback from the Tree of Thoughts algorithm.



multi-modality tree of thoughts

multi-modality forest of thoughts

multi-modality world of thoughts



### Multi-Modality Tree of Thoughts 🌐🌳

The next big advancement for the Tree of Thoughts algorithm is to extend it to multi-modality, enabling it to handle not only text but also images, audio, and other data types. This will bring us closer to multi-modal superintelligence.

#### Actionable Steps

1. Research and identify suitable multi-modal pre-trained models that can handle various data types (e.g., text, images, audio).
2. Adapt the thought decomposition, thought generator, and state evaluator functions to handle multi-modal data.
3. Develop a method for combining different modalities in the search tree, allowing the algorithm to reason across different data types.
4. Implement and test the multi-modal Tree of Thoughts algorithm with various problems and datasets.
5. Optimize the algorithm for performance and resource usage, ensuring it scales well with large multi-modal datasets.
6. Publish the results and gather feedback from the community to further improve the multi-modal Tree of Thoughts algorithm.

Join us on this exciting journey to advance the Tree of Thoughts algorithm to multi-modality superintelligence! 🚀