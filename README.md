# Tree of Thoughts 🌳🌲🌴🌿🍃

[Paper link](https://arxiv.org/pdf/2305.10601.pdf)

Disclaimer: this is not the official implementation for the paper, so do not expect to replicate paper results using this repo. Instead, refer to the official implementation at https://github.com/ysymyth/tree-of-thought-llm to replicate paper results.

![tree of thoughts banner](tree-of-thoughts.png)



Tree of Thoughts (ToT) is an all-new powerful and flexible algorithm that advances model reasoning by a whopping 70%. This is an plug in and play verision, connect your own models and enjoy superintelligence!


Share this repository by clicking on the following buttons 😊 

[![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts)](https://twitter.com/intent/tweet?text=Check%20out%20this%20amazing%20project%20on%20improving%20AI%20reasoning%20-%20Tree%20of%20Thoughts!%20https://github.com/kyegomez/tree-of-thoughts)
[![LinkedIn](https://img.shields.io/badge/Share-LinkedIn-blue?style=social&logo=linkedin)](https://www.linkedin.com/sharing/share-offsite/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts)

# Join Agora, Creators United
This implementation of Tree of Thoughts is brought to you by Agora, Agora advances Humanity with open source SOTA Multi-Modality AI research!


[Join our Discord and contribute to this project](https://discord.gg/qUtxnK2NMf)


# Basic Prompts:
No complex implementations, just pass in one of these prompts to your model: head over to `prompts.txt`

'Three experts with exceptional logical thinking skills are collaboratively answering a question using a tree of thoughts method. Each expert will share their thought process in detail, taking into account the previous thoughts of others and admitting any errors. They will iteratively refine and expand upon each other's ideas, giving credit where it's due. The process continues until a conclusive answer is found. Organize the entire response in a markdown table format. The question is...'

## Getting started
Set Openai key in an environment file,

first create an file called: `.env` 

Then get your openai key and input it inside the '' as `OPENAI_API_KEY='SK-YOUR KEY'`


## Method1
Clone this repository with 

```git clone https://github.com/kyegomez/tree-of-thoughts```

``` 
cd tree-of-thoughts
python3 -m pip install -r requirements.txt
cd tree_of_thoughts
python3 treeofthoughts.py --problem "design an new transportation system for an all-new city" --search_algorithm="BFS"
```
Add ` OPENAI_API_KEY='API KEY'` in the .env!

!!!! For much improved performance provide custom few prompt shots in the generate thoughts and generate states! !!!!!

And in the `examples` folder we have other examples for huggingface transformers + hugginggface pipelines

## Method2
or:

```pip install tree-of-thoughts ```


Create a Python script (e.g., example.py) and import the necessary classes:

``` python
import os
from tree_of_thoughts import OptimizedOpenAILanguageModel
from tree_of_thoughts import TreeofThoughts


api_key = os.getenv("OPENAI_API_KEY")
model = OptimizedOpenAILanguageModel(api_key=api_key) # api_model="gpt4" # for higher performance base model is not smart


#choose search algorithm('BFS' or 'DFS')
search_algorithm = "BFS"

#cot or propose
strategy="cot"

# value or vote
evaluation_strategy = "value"

#initialize the class
tree_of_thoughts = TreeofThoughts(model, search_algorithm)

#enter an problem if you want!
input_problem = "use 4 numbers and basic arithmetic operations (+-*/) to obtain 24" #note for superior intelligent responses you'll have to be more explicit in your prompt and select a better model
    

num_thoughts = 2
max_steps= 3
max_states = 5
value_threshold= 0.5

#call the solve emthod with the input problem and other params

solution = tree_of_thoughts.solve(input_problem, 
    num_thoughts=num_thoughts,
    max_steps=max_states,
    max_states=5,
    value_threshold=value_threshold,
    )

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

# Acknowledgements

Thanks to: Shunyu Yao Princeton University, Dian Yu Google DeepMind, Jeffrey Zhao, Google DeepMind, Izhak Shafran Google DeepMind, Thomas L. Griffiths, Princeton University, Yuan Cao Google DeepMind, Karthik Narasimha, Princeton University for sharing this amazing work with the world!

And, thanks to Phil Wang or Lucidrains for inspiring me to devote myself to open source AI Research
