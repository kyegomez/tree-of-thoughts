"""Game of 24 with breadth-first search.

The benchmark task from Yao et al. (2023): combine four numbers with
+ - * / to obtain 24. Each thought applies one operation to two of the
remaining numbers, so a solution takes exactly three thoughts. The beam keeps
the three most promising partial solutions at each depth.

Expected answer: (10 - 4) * (13 - 9) = 24.
"""

from swarms import TreeOfThoughts

tot = TreeOfThoughts(
    name="Game-of-24-BFS",
    model_name="gpt-5.4",
    search_algorithm="bfs",
    generation_strategy="propose",
    evaluation_strategy="value",
    num_thoughts=3,
    breadth=3,
    max_depth=3,
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
