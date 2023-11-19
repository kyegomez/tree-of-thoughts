[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

![Tree of Thoughts Banner](treeofthoughts.png)

![Discord](https://img.shields.io/discord/999382051935506503)
[![Twitter](https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts)](https://twitter.com/intent/tweet?text=Check%20out%20this%20amazing%20project%20on%20improving%20AI%20reasoning%20-%20Tree%20of%20Thoughts!%20https://github.com/kyegomez/tree-of-thoughts)
[![LinkedIn](https://img.shields.io/badge/Share-LinkedIn-blue?style=social&logo=linkedin)](https://www.linkedin.com/sharing/share-offsite/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts)
[![Facebook](https://img.shields.io/badge/Share-Facebook-blue?style=social&logo=facebook)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts)
[![Reddit](https://img.shields.io/badge/Share-Reddit-orange?style=social&logo=reddit)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts&title=Check%20out%20this%20amazing%20project%20on%20improving%20AI%20reasoning%20-%20Tree%20of%20Thoughts%21)
[![Hacker News](https://img.shields.io/badge/Share-Hacker%20News-orange?style=social&logo=y-combinator)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts&t=Check%20out%20this%20amazing%20project%20on%20improving%20AI%20reasoning%20-%20Tree%20of%20Thoughts%21)
[![Pinterest](https://img.shields.io/badge/Share-Pinterest-red?style=social&logo=pinterest)](https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts&media=https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts%2Fraw%2Fmain%2Ftree-of-thoughts.jpeg&description=Check%20out%20this%20amazing%20project%20on%20improving%20AI%20reasoning%20-%20Tree%20of%20Thoughts%21)
[![WhatsApp](https://img.shields.io/badge/Share-WhatsApp-green?style=social&logo=whatsapp)](https://api.whatsapp.com/send?text=Check%20out%20this%20amazing%20project%20on%20improving%20AI%20reasoning%20-%20Tree%20of%20Thoughts%21%20https%3A%2F%2Fgithub.com%2Fkyegomez%2Ftree-of-thoughts)


[Paper link](https://arxiv.org/pdf/2305.10601.pdf)
[Author's implementation](https://github.com/princeton-nlp/tree-of-thought-llm)

## Introduction

Tree of Thoughts (ToT) is a powerful and flexible algorithm that significantly advances model reasoning by up to 70%. This plug-and-play version allows you to connect your own models and experience superintelligence!


## Install

```bash
pip install tree-of-thoughts
```

## Usage
```python
import os
from tree_of_thoughts import OpenAILanguageModel, MonteCarloTreeofThoughts

api_model = "gpt-3.5-turbo"
model = OpenAILanguageModel(api_key='api key', api_model=api_model)

# Initialize the MonteCarloTreeofThoughts class with the model
tree_of_thoughts = MonteCarloTreeofThoughts(model)

# To reproduce the same results from the Tree of Thoughts paper, or even better,
# craft a one-shot chain of thought prompt for your task below:

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
)

print(f"Solution: {solution}")
```


### ToT with HF LLM

To run Hugging Face Transformers with Tree of Thoughts:
```python
from tree_of_thoughts import TreeofThoughts, HuggingLanguageModel, MonteCarloTreeofThoughts

model_name="01-ai/Yi-34B"

model = HuggingLanguageModel(model_name, 
                             model_tokenizer=model_name, 
                             verbose=True)
                             

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

### Basic Prompts
- Copy and paste this into your llm!

```
"Three experts with exceptional logical thinking skills are collaboratively answering a question using the tree of thoughts method. Each expert will share their thought process in detail, taking into account the previous thoughts of others and admitting any errors. They will iteratively refine and expand upon each other's ideas, giving credit where it's due. The process continues until a conclusive answer is found. Organize the entire response in a markdown table format. The task is:
```



# Acknowledgements

Thanks to: Shunyu Yao Princeton University, Dian Yu Google DeepMind, Jeffrey Zhao, Google DeepMind, Izhak Shafran Google DeepMind, Thomas L. Griffiths, Princeton University, Yuan Cao Google DeepMind, Karthik Narasimha, Princeton University for sharing this amazing work with the world!

And, thanks to Phil Wang or Lucidrains for inspiring me to devote myself to open source AI Research

# License
Apache