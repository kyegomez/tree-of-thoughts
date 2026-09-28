"""Game of 24 with depth-first search.

Depth-first search follows the most promising thought first and backtracks
when a state scores below value_threshold, i.e. when the evaluator judges
that the remaining numbers can no longer reach 24. max_expansions bounds the
cost of a search that backtracks often.

Expected answer: (10 - 4) * (13 - 9) = 24.
"""

from swarms import TreeOfThoughts

tot = TreeOfThoughts(
    name="Game-of-24-DFS",
    model_name="gpt-5.4",
    search_algorithm="dfs",
    generation_strategy="propose",
    evaluation_strategy="value",
    num_thoughts=3,
    max_depth=3,
    value_threshold=0.5,
    max_expansions=10,
    thought_description=(
        "One arithmetic operation on two of the remaining numbers, "
        "followed by the numbers left, e.g. '13 - 9 = 4 (left: 4 4 10)'."
    ),
    evaluation_criteria=(
        "Is the arithmetic correct, and is every input number used exactly "
        "once? Can the remaining numbers still reach 24?"
    ),
)

answer = tot.run("Use 4, 9, 10 and 13 with + - * / to obtain 24.")
print(f"Answer: {answer}\n")

result = tot.last_result
for number, step in enumerate(result.steps, 1):
    print(f"{number}. {step}")
print(
    f"\nsolved={result.solved} nodes_expanded={result.nodes_expanded} "
    f"llm_calls={result.llm_calls} tokens={result.usage['total_tokens']}"
)
