"""Quantum mechanics: the photon emitted by an electron in an infinite well.

Breadth-first search that samples each candidate step in a separate call, so
the beam holds different routes to the answer, such as computing energies in
electronvolts first or simplifying the wavelength symbolically first.
Expected answer: E_1 is about 0.376 eV, the transition releases about
3.01 eV, and the photon wavelength is about 412 nm.
"""

from swarms import TreeOfThoughts

tot = TreeOfThoughts(
    name="Quantum-Solver",
    model_name="gpt-5.4",
    search_algorithm="bfs",
    generation_strategy="sample",
    evaluation_strategy="value",
    num_thoughts=3,
    breadth=2,
    max_depth=4,
    thought_description=(
        "One step with units: write the energy levels of the infinite "
        "well, evaluate one energy, take an energy difference, or convert "
        "an energy to a photon wavelength."
    ),
    evaluation_criteria=(
        "Are the energy levels E_n = n^2 h^2 / (8 m L^2) used with the right "
        "quantum numbers? Are joules and electronvolts converted correctly? "
        "Is the wavelength physically sensible for the energy found?"
    ),
)

answer = tot.run(
    "An electron is confined to a one-dimensional infinite square well of "
    "width 1.0 nm. What is the wavelength of the photon emitted when it "
    "drops from the n = 3 state to the n = 1 state? Use h = 6.626e-34 J s, "
    "m_e = 9.109e-31 kg and c = 2.998e8 m/s."
)
print(f"Answer: {answer}\n")

result = tot.last_result
for number, step in enumerate(result.steps, 1):
    print(f"{number}. {step}")
print(
    f"\nsolved={result.solved} nodes_expanded={result.nodes_expanded} "
    f"llm_calls={result.llm_calls} tokens={result.usage['total_tokens']}"
)
