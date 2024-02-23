[![Multi-Modality](imags/agorabanner.png)](https://discord.gg/qUtxnK2NMf)

![Tree of Thoughts Banner](images/treeofthoughts.png)

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
from tree_of_thoughts import ToTAgent, MonteCarloSearch
from dotenv import load_dotenv
from swarms import Agent, OpenAIChat

load_dotenv()

# Get the API key from the environment
api_key = os.environ.get("OPENAI_API_KEY")

# Initialize an agent from swarms
agent = Agent(
    agent_name="tree_of_thoughts",
    agent_description="This agent uses the tree_of_thoughts library to generate thoughts.",
    system_prompt=None,
    llm = OpenAIChat(),   
)

# Initialize the ToTAgent class with the API key
model = ToTAgent(
    agent,
    strategy="cot",
    evaluation_strategy="value",
    enable_react=True,
    k=3,
)


# Initialize the MonteCarloSearch class with the model
tree_of_thoughts = MonteCarloSearch(model)

# Define the initial prompt
initial_prompt = """


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

# Define the number of thoughts to generate
num_thoughts = 1
max_steps = 3
max_states = 4
pruning_threshold = 0.5


# Generate the thoughts
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


### ToT with HF LLM

To run Hugging Face Transformers with Tree of Thoughts:
```python
import os
from tree_of_thoughts import ToTAgent, MonteCarloSearch
from dotenv import load_dotenv
from swarms import Agent, HuggingfaceLLM

load_dotenv()

# Get the API key from the environment
api_key = os.environ.get("OPENAI_API_KEY")

# Initialize an agent from swarms
agent = Agent(
    agent_name="tree_of_thoughts",
    agent_description="This agent uses the tree_of_thoughts library to generate thoughts.",
    system_prompt=None,
    llm = HuggingfaceLLM(model),   
)

# Initialize the ToTAgent class with the API key
model = ToTAgent(
    agent,
    strategy="cot",
    evaluation_strategy="value",
    enable_react=True,
    k=3,
)


# Initialize the MonteCarloSearch class with the model
tree_of_thoughts = MonteCarloSearch(model)

# Define the initial prompt
initial_prompt = """


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

# Define the number of thoughts to generate
num_thoughts = 1
max_steps = 3
max_states = 4
pruning_threshold = 0.5


# Generate the thoughts
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