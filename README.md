# Tree of Thoughts 🌳🌲🌴🌿🍃

![tree of thoughts banner](tree-of-thoughts.jpeg)

[Paper link](https://arxiv.org/pdf/2305.10601.pdf)

Tree of Thoughts (ToT) is an all-new powerful and flexible algorithm that advances model reasoning by a whopping 70%. This is an plug in and play verision, connect your own models and enjoy superintelligence!


Share this repository by clicking on the following buttons 😊 

[![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts)](https://twitter.com/intent/tweet?text=Check%20out%20this%20amazing%20project%20on%20improving%20AI%20reasoning%20-%20Tree%20of%20Thoughts!%20https://github.com/kyegomez/tree-of-thoughts)
[![LinkedIn](https://img.shields.io/badge/Share-LinkedIn-blue?style=social&logo=linkedin)](https://www.linkedin.com/sharing/share-offsite/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts)

# Join Agora, Creators United
This implementation of Tree of Thoughts is brought to you by Agora, Agora advances Humanity with open source SOTA Multi-Modality AI research! We plan on combating Humanity's grandest root problems like food insecurity, planetary insecurity, and disease, and hopefully death itself.

[Join our Discord and contribute to this project](https://discord.gg/qUtxnK2NMf)

## Getting started

## Method1
Clone this repository with 

```git clone https://github.com/kyegomez/tree-of-thoughts```

``` 
cd tree-of-thoughts
cd tree_of_thoughts
python3 treeofthoughts.py --problem "design an new transportation system for an all-new city" --search_algorithm="BFS"
```
Add ` OPENAI_API_KEY='API KEY'` in the .env!



## Method2
or:

```pip install tree-of-thoughts ```


Navigate to the repository folder: ```cd tree-of-thoughts```

```pip install openai```

Create a Python script (e.g., example.py) and import the necessary classes:

``` python
from tree_of_thoughts.treeofthoughts import OpenAILanguageModel, CustomLanguageModel, TreeofThoughts, OptimizedOpenAILanguageModel, OptimizedTreeofThoughts, HuggingLanguageModel

use_v2 = False

api_key=""

api_base= "" # leave it blank if you simply use default openai api url

if not use_v2:
    #v1
    model = OpenAILanguageModel(api_key=api_key, api_base=api_base # api_model="gpt4" # for higher performance base model is not smart
    ) 
else:
    #v2 parallel execution, caching, adaptive temperature
    model = OptimizedOpenAILanguageModel(api_key=api_key, api_base=api_base, # api_model="gpt4" # for higher performance base model is not smart
    )

#huggingface
# model_name="gpt2"
# model = HuggingLanguageModel(model_name)


#choose search algorithm('BFS' or 'DFS')
search_algorithm = "BFS"

#cot or propose
strategy="cot"

# value or vote
evaluation_strategy = "value"

# if not use_v2:
#     #create an instance of the tree of thoughts class v1


tree_of_thoughts = TreeofThoughts(model, search_algorithm)


# else:
#     #or v2 -> dynamic beam width -< adjust the beam width [b] dynamically based on the search depth quality of the generated thoughts # does not work now use regular tree of thoughts
#     tree_of_thoughts= OptimizedTreeofThoughts(model, search_algorithm)

input_problem = "use 4 numbers and basic arithmetic operations (+-*/) to obtain 24" #note for super intelligent responses you'll have to be more explicit in your prompt and select a better model
    


input_problem = "What are the best reasoning methods to advance Large Language Models"
k = 5 #number of thoughts to generate
T = 3 # maximum depth of the search tree
b = 5 # branching factor -< number of child nodes for each branch
vth = 0.5 # pruning state -> any evaluated thought below this is eliminated
timeout = 10 #10 seconds timeout before stop
confidence = 0.8 #cmodel is confident on performance
max_iterations = 40 #tree branh nodes 
convergence_threshold = 0.01 #determining when the search process has converged
convergence_count = 5 # number of searchers to be considered converged
#read documentation for more

#call the solve emthod with the input problem and other params
solution = tree_of_thoughts.solve(input_problem, k, T, b, vth, timeout, confidence, max_iterations, convergence_threshold, convergence_count)

    


# # Save the tree and metrics to a JSON file
# file_name = "logs/tree_of_thoughts_output.json"
# tree_of_thoughts.save_tree_to_json(file_name)

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

## Tree of Thoughts Class
``` python
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
To run huggingface transformers
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

    def evaluate_states(self, states, inital_prompt):
        state_values = {}
        for state in states:
            state_text = ' '.join(state)
            prompt = f"Given the current state of reasoning: '{state_text}', pessimitically evaluate its value as a float between 0 and 1 based on it's potential to achieve {inital_prompt}"

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


## Roadmap

now:
Generate suite of evaluations used in the paper testing AI agents with other reasoning methods like COT and self consistency and run them in parallel to conduct evaluation experiments.

Implement a more sophisticated prompt engineering strategy to guide the model's reasoning process more effectively.

Make TreeofThoughts class completely customizable with a config yml file with params like
chatbot:
    type: "openai"
    max_context_length: 8000
    include_chat_history_in_query: false
openai:
    model: <model_name>
    api_key: <your_open_ai_api_key>


Script that generates an dataset based on a topic input, -> set of questions are asked, then multiple trees of thoughts are run concurrently to generate the decision making rich dataset


Introduce a reinforcement learning, distillment, and finetuning scripts to finely tune the model based on feedback from the Tree of Thoughts algorithm.

Integrate heuristics that autonomously determine the search algorithm based on indicators

Integrate heuristics that autonomously determine the strategy cos or propose

Integrate heuristics that autonomously set the input params:

k = 
T = 
b = 
vth = 


# multi modal 
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

# Here's the documentation for the inputs of the optimized Tree of Thoughts model:

## x (str): 
The initial problem statement or prompt for which the Tree of Thoughts algorithm will generate a solution.

## k (int, default=5): 
The number of thoughts to generate at each state. 
A higher value of k will result in more thoughts being generated, potentially leading to a more diverse set of solutions. However, increasing k may also increase the computational complexity and time required to find a solution.

## T (int, default=3): 
The maximum depth of the search tree. 
A higher value of T allows the algorithm to explore deeper states, potentially leading to better solutions. However, increasing T may also increase the computational complexity and time required to find a solution.

## b (int, default=5): 
The branching factor of the search tree, which determines the maximum number of child nodes for each parent node.
A higher value of b allows the algorithm to explore more states, potentially leading to better solutions. However, increasing b may also increase the computational complexity and time required to find a solution.

## vth (float, default=0.5): 
The value threshold for pruning states. 
States with a value below this threshold will be discarded, reducing the search space. A higher value of vth will result in a more aggressive pruning strategy, potentially speeding up the search process. However, setting vth too high may cause the algorithm to discard promising states, leading to suboptimal solutions.

## timeout (int, default=10): 
The maximum time (in seconds) allowed for the search process. If the search process exceeds this time limit, the algorithm will return the best solution found so far.

## confidence_threshold (float, default=0.8): 
The confidence threshold for determining when a solution is satisfactory. If the algorithm finds a solution with a confidence value above this threshold, it will return the solution immediately.

## max_iterations (int, default=40): 
The maximum number of iterations allowed for the search process. If the search process exceeds this number of iterations, the algorithm will return the best solution found so far.

## convergence_threshold (float, default=0.01): 
The convergence threshold for determining when the search process has converged. If the difference in confidence values between consecutive iterations is below this threshold for a specified number of iterations (convergence_count), the algorithm will return the best solution found so far.

## convergence_count (int, default=5): 
The number of consecutive iterations required for the search process to be considered converged. If the difference in confidence values between consecutive iterations is below the convergence_threshold for this number of iterations, the algorithm will return the best solution found so far.


# Acknowledgements

Thanks to: Shunyu Yao Princeton University, Dian Yu Google DeepMind, Jeffrey Zhao, Google DeepMind, Izhak Shafran Google DeepMind, Thomas L. Griffiths, Princeton University, Yuan Cao Google DeepMind, Karthik Narasimha, Princeton University for sharing this amazing work with the world!

And, thanks to Phil Wang or Lucidrains for inspiring me to devote myself to open source AI Research


The TreeofThoughts class is a search algorithm that aims to solve problems by generating and evaluating thoughts in a tree-like structure. The algorithm can be configured to use either Breadth-First Search (BFS) or Depth-First Search (DFS) as the search strategy. The algorithm consists of four main components:

## Thought Decomposition: The problem is decomposed based on its properties.

Thought Generator: A function G(p0, s, k) is created to generate thoughts using two strategies: a. Sample independent and identically distributed (iid) thoughts from a cot prompt. b. Propose thoughts sequentially using a propose prompt.

State Evaluator: A function V(p0, S) is created to evaluate states using two strategies: a. Value each state independently. b. Vote across states.

Search Algorithm: Choose a search algorithm based on the tree structure (BFS or DFS).
The algorithm is implemented as follows:

### For BFS (tot_bfs method):

Initialize S0 with the input x.
For each step t from 1 to T (step limit): a. Generate candidate thoughts for each state in St-1. b. Evaluate the candidate states using the state evaluator V. c. Select the b most promising states for St.
Return the final output by generating the thought for the best state in St.
For DFS (tot_dfs method):

### Define a recursive DFS function with the current state s, step t, and other required parameters.
If t > T, record the output by generating the thought for the current state s.
For each candidate state s in the sorted list of generated thoughts for s: a. If the evaluated value of s is greater than the threshold vth, call the DFS function recursively with s and t + 1.
The solve method executes the chosen search algorithm with the input problem, thought generator, state evaluator, and other required parameters. The OptimizedTreeofThoughts class is a subclass of TreeofThoughts that aims to improve the performance of the algorithm but currently does not output the state after each thought.

# Here is the documentation for the variables used in the TreeofThoughts class:

x: The input problem statement or initial state that needs to be solved.
k: The number of thoughts to be generated by the thought generator function G(p0, s, k).
T: The maximum number of steps (step limit) for the search algorithm.
b: The number of most promising states to be selected at each step in the BFS algorithm.
vth: The value threshold used in the DFS algorithm to decide whether to explore a candidate state further.
s: A state in the search space, which is a tuple representing the sequence of thoughts generated so far.
S0: The set of states at the current step of the search algorithm. In the BFS algorithm, it is initialized with the input x. In the DFS algorithm, it is updated recursively as the search progresses.
St: The set of most promising states selected at step t in the BFS algorithm.
S0_t: The set of candidate states generated at step t in the BFS algorithm by combining the states in S0 with the generated thoughts.
Z: A thought generated by the thought generator function G(p0, s, k).

# Here's a brief explanation of each variable:

x: The input problem statement that the algorithm aims to solve.
k: Controls the number of thoughts generated at each step, which affects the branching factor of the search tree.
T: Limits the depth of the search tree, preventing the algorithm from running indefinitely.
b: Determines the number of promising states to explore at each step in the BFS algorithm, which affects the breadth of the search.
vth: A threshold used in the DFS algorithm to decide whether a candidate state is worth exploring further, helping to prune the search tree.
s: Represents a state in the search space, which is a sequence of thoughts generated so far.
S0: The set of states at the current step of the search algorithm, used to keep track of the search progress.
St: The set of most promising states selected at each step in the BFS algorithm, guiding the search towards better solutions.
S0_t: The set of candidate states generated at each step in the BFS algorithm, which are evaluated and used to update S0.
Z: A thought generated by the thought generator function, which is combined with the current state to create a new candidate state.


