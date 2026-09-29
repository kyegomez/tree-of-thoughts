from swarms import TreeOfThoughts

# Create a Tree of Thoughts agent that searches depth-first
tot = TreeOfThoughts(
    model_name="gpt-5.4",  # Any LiteLLM model that supports function calling
    search_algorithm="dfs",  # "bfs" (beam search) or "dfs" (backtracking)
    num_thoughts=3,  # Candidate thoughts generated per expanded state
    max_depth=3,  # Three operations combine four numbers into one
    value_threshold=0.5,  # Thoughts scoring below 0.5 are pruned
    thought_description=(
        "One arithmetic operation on two of the remaining numbers, "
        "followed by the numbers left, e.g. '13 - 9 = 4 (left: 4 4 10)'."
    ),
    evaluation_criteria="Can the remaining numbers still reach 24?",
)

# Run the search and print the final answer
answer = tot.run("Use 4, 9, 10 and 13 with + - * / to obtain 24.")
print(answer)

# Print the best reasoning path and whether it passed evaluation
result = tot.last_result
for number, step in enumerate(result.steps, 1):
    print(f"{number}. {step}")
print(f"solved={result.solved} llm_calls={result.llm_calls}")
