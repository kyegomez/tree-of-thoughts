"""Special relativity: the threshold energy for antiproton production.

A proton beam strikes protons at rest. The reaction p + p -> p + p + p + p̄
needs enough energy in the centre-of-mass frame to create two extra proton
masses, and a non-relativistic treatment gets the answer badly wrong.
Depth-first search with a raised threshold prunes any step that confuses
kinetic with total energy or skips the invariant mass. Expected answer:
6 m_p c^2, about 5.63 GeV.
"""

from swarms import TreeOfThoughts

tot = TreeOfThoughts(
    name="Relativistic-Kinematics-Solver",
    model_name="gpt-5.4",
    search_algorithm="dfs",
    generation_strategy="propose",
    evaluation_strategy="value",
    num_thoughts=3,
    max_depth=4,
    value_threshold=0.6,
    thought_description=(
        "One step of relativistic kinematics: write an invariant such as "
        "s = (sum of four-momenta)^2, evaluate it in one frame, apply the "
        "threshold condition, or solve for one quantity."
    ),
    evaluation_criteria=(
        "Is the invariant mass evaluated correctly in both the lab frame "
        "and the centre-of-mass frame? At threshold, what must the "
        "products be doing in the centre-of-mass frame? Is kinetic energy "
        "kept distinct from total energy? Penalize non-relativistic "
        "formulas."
    ),
)

answer = tot.run(
    "A beam of protons strikes protons at rest in a fixed target. What is "
    "the minimum kinetic energy of a beam proton for the reaction "
    "p + p -> p + p + p + p̄ (antiproton production) to occur? Take the "
    "proton rest energy to be 938.3 MeV."
)
print(f"Answer: {answer}\n")

result = tot.last_result
for number, step in enumerate(result.steps, 1):
    print(f"{number}. {step}")
print(
    f"\nsolved={result.solved} nodes_expanded={result.nodes_expanded} "
    f"llm_calls={result.llm_calls} tokens={result.usage['total_tokens']}"
)
