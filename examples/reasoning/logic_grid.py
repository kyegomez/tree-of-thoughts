"""Logic grid: match four researchers to floors, fields and drinks.

Breadth-first search makes one deduction per step and keeps three partial
assignments alive, so a placement that later contradicts a clue does not
sink the search. The puzzle has exactly one solution. Expected answer:
Ben, floor 1, mathematics, tea; Dev, floor 2, biology, juice; Cara, floor 3,
chemistry, coffee; Ava, floor 4, physics, water.
"""

from swarms import TreeOfThoughts

tot = TreeOfThoughts(
    name="Logic-Grid-Solver",
    model_name="gpt-5.4",
    search_algorithm="bfs",
    generation_strategy="propose",
    evaluation_strategy="value",
    num_thoughts=3,
    breadth=3,
    max_depth=5,
    thought_description=(
        "One deduction: fix one attribute of one person, or rule out "
        "options for a person, naming the clues that force it."
    ),
    evaluation_criteria=(
        "Does the partial assignment break any clue, including one that "
        "the remaining options can no longer satisfy? A final answer "
        "gives every person a distinct floor, field and drink and "
        "satisfies all nine clues."
    ),
)

answer = tot.run(
    "Ava, Ben, Cara and Dev each have an office on a different floor of a "
    "four-floor institute (floors 1 to 4). Each works in a different field "
    "(physics, chemistry, biology, mathematics) and drinks something "
    "different (coffee, tea, water, juice). Clues: (1) Ben works on floor "
    "1. (2) Ava works on a higher floor than Dev. (3) The physicist works "
    "on the floor directly above the chemist. (4) Cara is not the "
    "biologist. (5) The mathematician drinks tea. (6) Cara drinks coffee. "
    "(7) The person on floor 4 drinks water. (8) The biologist drinks "
    "juice. (9) Dev is not the mathematician. Give each person's floor, "
    "field and drink."
)
print(f"Answer: {answer}\n")

result = tot.last_result
for number, step in enumerate(result.steps, 1):
    print(f"{number}. {step}")
print(
    f"\nsolved={result.solved} nodes_expanded={result.nodes_expanded} "
    f"llm_calls={result.llm_calls} tokens={result.usage['total_tokens']}"
)
