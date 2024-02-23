import os
from tree_of_thoughts.openai_models import OpenAILanguageModel
from tree_of_thoughts.treeofthoughts import MonteCarloTreeofThoughts
from dotenv import load_dotenv

load_dotenv()


api_key = os.environ.get("OPENAI_API_KEY")

# Initialize the OpenAILanguageModel class with the API key
model = OpenAILanguageModel(api_key=api_key)


# Initialize the MonteCarloTreeofThoughts class with the model
tree_of_thoughts = MonteCarloTreeofThoughts(model)

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
