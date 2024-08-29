[![Multi-Modality](images/agorabanner.png)](https://discord.gg/qUtxnK2NMf)

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
$ pip3 install -U tree-of-thoughts
```

## Example
```python
from tree_of_thoughts import TotAgent, DFSWithTotAgent

tot_agent = TotAgent()

# Create the DFSWithTotAgent class with a threshold, max steps, and pruning threshold
dfs_agent = DFSWithTotAgent(
    agent=tot_agent,
    threshold=0.8,
    max_loops=1,
    prune_threshold=0.5,  # Branches with evaluation < 0.5 will be pruned
    number_of_agents=4,
)

# Starting state for the DFS algorithm
initial_state = """

Your task: is to use 4 numbers and basic arithmetic operations (+-*/) to obtain 24 in 1 equation, return only the math

"""

# Run the DFS algorithm to solve the problem
final_thought = dfs_agent.run(initial_state)

# Outputs json which is easy to read
print(final_thought)


```

### Basic Prompts
```txt

Imagine three different experts are answering this question. All experts will write down 1 step of their thinking, then share it with the group. Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave. The question is...



################ 2nd ################

Simulate three brilliant, logical experts collaboratively answering a question. Each one verbosely explains their thought process in real-time, considering the prior explanations of others and openly acknowledging mistakes. At each step, whenever possible, each expert refines and builds upon the thoughts of others, acknowledging their contributions. They continue until there is a definitive answer to the question. For clarity, your entire response should be in a markdown table. The question is...


################ ################

Imagine three highly intelligent experts working together to answer a question. They will follow a tree of thoughts approach, where each expert shares their thought process step by step. They will consider the input from others, refine their thoughts, and build upon the group's collective knowledge. If an expert realizes their thought is incorrect, they will acknowledge it and withdraw from the discussion. Continue this process until a definitive answer is reached. Present the entire response in a markdown table. The question is...


################ 2nd ################

Three experts with exceptional logical thinking skills are collaboratively answering a question using a tree of thoughts method. Each expert will share their thought process in detail, taking into account the previous thoughts of others and admitting any errors. They will iteratively refine and expand upon each other's ideas, giving credit where it's due. The process continues until a conclusive answer is found. Organize the entire response in a markdown table format. The question is...
################ 2nd ################


Envision a group of three experts working in unison to tackle a question by employing a tree of thoughts strategy. Each expert will thoroughly explain their line of thinking at every step, while also considering the insights provided by their peers. They will openly recognize any mistakes and build upon the group's shared understanding. This iterative process will continue until a definitive solution is reached. Structure the entire response as a markdown table. The question is...


################ 2nd ################

"Three experts with exceptional logical thinking skills are collaboratively answering a question using the tree of thoughts method. Each expert will share their thought process in detail, taking into account the previous thoughts of others and admitting any errors. They will iteratively refine and expand upon each other's ideas, giving credit where it's due. The process continues until a conclusive answer is found. Organize the entire response in a markdown table format. The task is:
```

## Todo
- [ ] Finish implementing the depth or max_loops feature in the dfs class
- [ ] Finish the new BFS search algorithm
- [ ] Implement montecarlo search algorithm
- [ ] Make a function that can intake json and make a tree out of it visually to visualize the tree of thoughts! 


# Acknowledgements

Thanks to: Shunyu Yao Princeton University, Dian Yu Google DeepMind, Jeffrey Zhao, Google DeepMind, Izhak Shafran Google DeepMind, Thomas L. Griffiths, Princeton University, Yuan Cao Google DeepMind, Karthik Narasimha, Princeton University for sharing this amazing work with the world!

And, thanks to Phil Wang or Lucidrains for inspiring me to devote myself to open source AI Research

# License
Apache
