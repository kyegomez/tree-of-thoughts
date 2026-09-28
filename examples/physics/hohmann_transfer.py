"""Orbital mechanics: a Hohmann transfer from low Earth orbit to GEO.

Breadth-first search keeps three partial calculations alive and averages two
evaluator ratings per candidate, so a single misjudged burn cannot steer the
beam. Expected answer: about 2.43 km/s for the first burn and 1.47 km/s for
the second, 3.90 km/s in total, with a transfer time of about 5.27 hours.
"""

from swarms import TreeOfThoughts

tot = TreeOfThoughts(
    name="Orbital-Mechanics-Solver",
    model_name="gpt-5.4",
    search_algorithm="bfs",
    generation_strategy="propose",
    evaluation_strategy="value",
    num_thoughts=3,
    breadth=3,
    max_depth=5,
    n_evaluate_samples=2,
    thought_description=(
        "One step of the calculation with units: an orbital radius, a "
        "circular or transfer-orbit speed from the vis-viva equation, one "
        "burn's delta-v, or the transfer time from Kepler's third law."
    ),
    evaluation_criteria=(
        "Are radii measured from Earth's centre, not its surface? Is the "
        "vis-viva equation applied with the right semi-major axis? Are "
        "speeds in consistent units? Is the transfer time half an orbital "
        "period of the transfer ellipse?"
    ),
)

answer = tot.run(
    "A satellite is in a circular equatorial orbit 300 km above Earth's "
    "surface. Using a Hohmann transfer, what total delta-v is needed to "
    "reach geostationary orbit, whose radius is 42,164 km, and how long "
    "does the transfer take? Take Earth's gravitational parameter to be "
    "398,600 km^3/s^2 and its radius to be 6,371 km."
)
print(f"Answer: {answer}\n")

result = tot.last_result
for number, step in enumerate(result.steps, 1):
    print(f"{number}. {step}")
print(
    f"\nsolved={result.solved} nodes_expanded={result.nodes_expanded} "
    f"llm_calls={result.llm_calls} tokens={result.usage['total_tokens']}"
)
