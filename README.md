# Tree of Thoughts 🌳🌲🌴🌿🍃

![Discord](https://img.shields.io/discord/999382051935506503)![![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts)](https://twitter.com/intent/tweet?text=Check%20out%20this%20amazing%20project%20on%20improving%20AI%20reasoning%20-%20Tree%20of%20Thoughts!%20https://github.com/kyegomez/tree-of-thoughts)
[![LinkedIn](https://img.shields.io/badge/Share-LinkedIn-blue?style=social&logo=linkedin)](https://www.linkedin.com/sharing/share-offsite/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts)
[![Facebook](https://img.shields.io/badge/Share-Facebook-blue?style=social&logo=facebook)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts)
[![Reddit](https://img.shields.io/badge/Share-Reddit-orange?style=social&logo=reddit)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts&title=Check%20out%20this%20amazing%20project%20on%20improving%20AI%20reasoning%20-%20Tree%20of%20Thoughts%21)
[![Hacker News](https://img.shields.io/badge/Share-Hacker%20News-orange?style=social&logo=y-combinator)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts&t=Check%20out%20this%20amazing%20project%20on%20improving%20AI%20reasoning%20-%20Tree%20of%20Thoughts%21)
[![Pinterest](https://img.shields.io/badge/Share-Pinterest-red?style=social&logo=pinterest)](https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts&media=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts%2Fraw%2Fmain%2Ftree-of-thoughts.jpeg&description=Check%20out%20this%20amazing%20project%20on%20improving%20AI%20reasoning%20-%20Tree%20of%20Thoughts%21)
[![WhatsApp](https://img.shields.io/badge/Share-WhatsApp-green?style=social&logo=whatsapp)](https://api.whatsapp.com/send?text=Check%20out%20this%20amazing%20project%20on%20improving%20AI%20reasoning%20-%20Tree%20of%20Thoughts%21%20https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts)


![tree of thoughts banner](tree-of-thoughts.png)

[Paper link](https://arxiv.org/pdf/2305.10601.pdf)

Tree of Thoughts (ToT) is an all-new powerful and flexible algorithm that advances model reasoning by a whopping 70%. This is an plug in and play verision, connect your own models and enjoy superintelligence!

## 🔥 Updates

* Langchain TOT

* MonteCarlo

* A* Search 

* Best First Search

#### Coming soon!
* Iterative Depth Search 

* Any search algorithms you like?? Open an issue 😊 

# Basic Prompts:
No complex implementations, just pass in one of these prompts to your model: head over to `prompts.txt`

'Three experts with exceptional logical thinking skills are collaboratively answering a question using a tree of thoughts method. Each expert will share their thought process in detail, taking into account the previous thoughts of others and admitting any errors. They will iteratively refine and expand upon each other's ideas, giving credit where it's due. The process continues until a conclusive answer is found. Organize the entire response in a markdown table format. The question is...'

## Getting started

## Method1
Clone this repository with 

```git clone https://github.com/kyegomez/tree-of-thoughts```

Set Openai key in an environment file,

first create an file called: `.env` 

Then get your openai key and input it inside the '' as `OPENAI_API_KEY='SK-YOUR KEY'`

``` 

cd tree-of-thoughts
python3 -m pip install -r requirements.txt
cd tree_of_thoughts
```
Then go to `montecarlo_example.py` and fill in your api key! 

# 🔥 For much improved performance provide custom few prompt shots in the generate thoughts and generate states 🔥 

And in the `examples` folder we have other examples for huggingface transformers + hugginggface pipelines

## Method2
or:

```pip install tree-of-thoughts ```


Create a Python script (e.g., example.py) and import the necessary classes:

``` python
import os
from tree_of_thoughts import OpenAILanguageModel
from tree_of_thoughts import MonteCarloTreeofThoughts


api_model= "gpt-3.5-turbo"


model = OpenAILanguageModel(api_key='api key', api_model=api_model)


# Initialize the MonteCarloTreeofThoughts class with the model
tree_of_thoughts = MonteCarloTreeofThoughts(model)

# Note to reproduce the same results from the tree of thoughts paper if not better, 
# craft an 1 shot chain of thought prompt for your task below

initial_prompt =  """


Input: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 /  2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)
Input: use 4 numbers and basic arithmetic operations (+-*/) to obtain 24 in 1 equation
Possible next steps:



"""
num_thoughts = 1
max_steps = 3
max_states = 4
pruning_threshold = 0.5




solution = tree_of_thoughts.solve(
    initial_prompt=initial_prompt,
    num_thoughts=num_thoughts, 
    max_steps=max_steps, 
    max_states=max_states, 
    pruning_threshold=pruning_threshold,
    # sleep_time=sleep_time
)

print(f"Solution: {solution}")

```

Or Integrate your own custom language model:

```python

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

## 🌟 Features:
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


## Usage Examples

### OpenAI API

To use Tree of Thoughts with OpenAI's API, create a custom model class that inherits from `AbstractLanguageModel` and implements the required methods using OpenAI's API. Then, create an instance of the `TreeOfThoughts` class with the custom model and the desired search algorithm ('BFS' or 'DFS').

### Hugging Face Transformers
To run huggingface transformers with Tree of Thoughts

``` 
git clone https://github.com/kyegomez/tree-of-thoughts
cd tree-of-thoughts
python3 huggingfaceExample.py
```

```python
from tree_of_thoughts import HuggingLanguageModel

model_name="gpt2"
model_tokenizer="your tokenizer"

huggingface_model = HuggingLanguageModel(model_name, model_tokenizer)
```



```python
class HuggingLanguageModel(AbstractLanguageModel):
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_thoughts(self, state, k):
        state_text = ' '.join(state)
        prompt = f"Given the current state of reasoning: '{state_text}', generate {k} coherent thoughts to achieve the reasoning process:"

        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, num_return_sequences=k)
        thoughts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        return thoughts

    def evaluate_states(self, states, initial_prompt):
        state_values = {}
        for state in states:
            state_text = ' '.join(state)
            prompt = f"Given the current state of reasoning: '{state_text}', pessimitically evaluate its value as a float between 0 and 1 based on it's potential to achieve {initial_prompt}"

            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, num_return_sequences=1)
            value_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            try:
                value = float(value_text)
            except ValueError:
                value = 0  # Assign a default value if the conversion fails

            state_values[state] = value

        return state_values


```


# Contributing
This algorithm is still infant yet it's potential remains unimaginable, let's advance the reasoning of AI's together under this banner.

# Share With Your Network

You can easily share this repository by clicking on the following buttons:

[![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts)](https://twitter.com/intent/tweet?text=Check%20out%20this%20amazing%20project%20on%20improving%20AI%20reasoning%20-%20Tree%20of%20Thoughts!%20https://github.com/kyegomez/tree-of-thoughts)
[![LinkedIn](https://img.shields.io/badge/Share-LinkedIn-blue?style=social&logo=linkedin)](https://www.linkedin.com/sharing/share-offsite/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts)

For Instagram, while it doesn't directly support sharing of web links, you can share the screenshot of our project and the link in your caption or bio. You can download the project screenshot by clicking the image below:

[![Tree of Thoughts](https://github.com/kyegomez/tree-of-thoughts/raw/main/tree-of-thoughts.jpeg)](https://github.com/kyegomez/tree-of-thoughts/raw/main/tree-of-thoughts.jpeg)

We greatly appreciate any help in spreading the word about our project. Thank you for your support!

# Roadmap:

* Resilient Prompting: Teach model how to think rather than what to think.

* Add pruning treshold management for precise bad state cutoff

* Evaluating each thought as soon as thought generated then evaluating an chain of thoughts or the state of thoughts by averaging out the values of each thought evaluation.

* Add Traversal method, which "will incapsulate the run of either dfs or bfs under the hood so that the issue of different args is solved from @ivanzhovannik

* Add Delay between generate solutions and generate values

* Dynamic and adaptive parameters, like max steps, num_thoughts, max_states and value threshold that shift depending on the complexity of the user objective.

* Add Rejected reasoning metadata (thought, state, reasoning_on_state) into  generate solutions

* any other ideas? Please pr request this algorithm is very infant and it's potential is limitless

* Chain of Thought Hub Evaluation tests!

# Documentation:
# Search Algorithms in Tree of Thoughts 

The Tree of Thoughts library supports a variety of search algorithms that can be employed for different problem-solving contexts. Here's a brief overview of each search algorithm along with their primary benefits and use-cases.

## 1. Breadth-First Search (BFS)

BFS explores all the nodes at the present depth before going on to the nodes at the next depth level. It is an excellent choice when the depth of the tree is relatively small, and solutions are spread out evenly.

**Benefits:**
- It guarantees to find the shallowest goal, i.e., the solution with fewer steps.
- It is a simple and straightforward algorithm for traversing trees or graphs.

**Use-cases:**
- Ideal for problems where the depth of the tree/graph is not very large.
- Useful when the goal is close to the root.

## 2. Depth-First Search (DFS)

DFS explores as far as possible along each branch before backing up. It is suitable when the tree depth is significant, and solutions are located deep in the tree.

**Benefits:**
- It uses less memory compared to BFS as it needs to store only a single path from the root to a leaf node, along with remaining unexplored sibling nodes for each node on the path.
- It can explore deeper solutions that are not accessible with BFS.

**Use-cases:**
- It is often used in simulations due to its more aggressive (deeper) search.
- Ideal for searching through a big search space.

## 3. Best-First Search

Best-First Search uses an evaluation function to decide which adjacent node is most promising and then explores. It is suitable for problems where we have some heuristic information about the distance from the current state to the goal.

**Benefits:**
- It can provide a more efficient solution by using heuristics.
- It does not explore unnecessary paths, thus saving resources.

**Use-cases:**
- Suitable for a large dataset where the goal node's location is unknown.
- Ideal for problems where some heuristic can guide the search to the goal.

## 4. A* Search

A* Search finds the least-cost path from the given initial node to one goal node (out of one or more possible goals). It uses a best-first search and finds the least-cost path to a goal.

**Benefits:**
- It is complete, optimal, optimally efficient, and uses heuristics to guide itself.
- A* balances between BFS and DFS and avoids expanding paths that are already expensive.

**Use-cases:**
- Widely used in pathfinding and graph traversal, the process of plotting an efficiently directed path between multiple points.
- Suitable for games, mapping apps, and routing paths for vehicles where we need an optimal solution.

## 5. Monte Carlo Tree Search (MCTS)

MCTS uses random sampling of the search space and uses the results to guide the search. It is best when the search space is vast and cannot be completely traversed.

**Benefits:**
- It can provide good solutions for extremely complex problems with a large search space, where traditional methods fail.
- It uses statistical analysis of the results for decision making, which can handle the uncertainty and variability in the problem.

**Use-cases:**
- Suitable for "perfect information" games, which are games where players have complete knowledge of all events and states.
- Also useful in real-time video games and other domains where the decision-making time is limited.


## input_problem (str): 
The initial problem statement or prompt for which the Tree of Thoughts algorithm will generate a solution.

## num_thoughts (int, default=5): 
The number of thoughts to generate at each state. 
A higher value of k will result in more thoughts being generated, potentially leading to a more diverse set of solutions. However, increasing k may also increase the computational complexity and time required to find a solution.

## max_steps (int, default=3): 
The maximum depth of the search tree. 
A higher value of T allows the algorithm to explore deeper states, potentially leading to better solutions. However, increasing T may also increase the computational complexity and time required to find a solution.

## max_states (int, default=5): 
The branching factor of the search tree, which determines the maximum number of child nodes for each parent node.
A higher value of b allows the algorithm to explore more states, potentially leading to better solutions. However, increasing b may also increase the computational complexity and time required to find a solution.

## value_threshold (float, default=0.5): 
The value threshold for pruning states. 
States with a value below this threshold will be discarded, reducing the search space. A higher value of vth will result in a more aggressive pruning strategy, potentially speeding up the search process. However, setting vth too high may cause the algorithm to discard promising states, leading to suboptimal solutions.


## LangchainTOT class

`LangchainTOT` is the main class you'll interact with. It acts as a wrapper for the Large Language Model and allows you to set a problem description, add thoughts, and check the validity of these thoughts using a specified checker.

### Initialization

You initialize a `LangchainTOT` object with an optional problem description and a checker class:

```python
problem_description = """
3,*,*,2|1,*,3,*|*,1,*,3|4,*,*,1
- This is a 4x4 Sudoku puzzle.
- The * represents a cell to be filled.
- The | character separates rows.
- At each step, replace one or more * with digits 1-4.
- There must be no duplicate digits in any row, column or 2x2 subgrid.
- Keep the known digits from previous valid thoughts in place.
- Each thought can be a partial or the final solution.
""".strip()

langchain_tot = LangchainTOT(problem_description=problem_description, checker_class=lambda: my_checker)
```

If you want to change the problem description or checker class later, you can use the `set_problem_description` and `set_checker_class` methods.

### Adding thoughts

Once you have your `LangchainTOT` object, you can add thoughts to it. A thought is a string representing a possible solution or step towards a solution. You can add thoughts using the `add_thought` method:

```python
langchain_tot.add_thought("3,*,*,2|1,*,3,*|*,1,*,3|4,*,*,1")
```

### Checking thoughts

Once you've added one or more thoughts, you can check their validity using the `check_thoughts` method:

```python
print(langchain_tot.check_thoughts())
```

This method will return a `ThoughtValidity` value representing whether the latest thought is a final valid solution (`VALID_FINAL`), an intermediate valid step (`VALID_INTERMEDIATE`), or invalid (`INVALID`).

## MyChecker class

`MyChecker` is a class for creating custom checkers. It inherits from `ToTChecker` and must implement the `evaluate` method.

### Initialization

You initialize a `MyChecker` object with a validation function:

```python
my_checker = MyChecker(validate_fn=lambda p, t: validate_sudoku(p, t, sudoku_solution))
```

## Custom validation function

The validation function is a callable that takes a problem description and a tuple of thoughts and returns a `ThoughtValidity` value. It defines how to check the validity of thoughts for a specific problem.

In this code, we're using the `validate_sudoku` function as our validation function. This function takes a problem description, a tuple of thoughts, and a solution string, and returns the validity of the last thought based on whether it matches the solution and whether it's a possible step towards the solution.

The `validate_sudoku` function is specific to 4x4 Sudoku puzzles and you'll need to create your own validation function for different types of problems.


# Acknowledgements

Thanks to: Shunyu Yao Princeton University, Dian Yu Google DeepMind, Jeffrey Zhao, Google DeepMind, Izhak Shafran Google DeepMind, Thomas L. Griffiths, Princeton University, Yuan Cao Google DeepMind, Karthik Narasimha, Princeton University for sharing this amazing work with the world!

And, thanks to Phil Wang or Lucidrains for inspiring me to devote myself to open source AI Research
