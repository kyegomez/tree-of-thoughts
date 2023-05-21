# Tree of Thoughts 🌳🌲🌴🌿🍃

Tree of Thoughts (ToT) is a powerful and flexible algorithm for leveraging pre-trained language models to solve various problems by exploring multiple reasoning paths. It's designed to be plug-and-play, allowing users to easily connect their models and use the Tree of Thoughts method.

🌟 Features:
- General problem-solving framework for language models
- Supports both breadth-first search (BFS) and depth-first search (DFS) algorithms
- Easy integration with popular language models like OpenAI and Hugging Face
- Extensible and adaptable to different problem properties and resource constraints

## Algorithmic Pseudocode

1. Define the thought decomposition based on the problem properties.
2. Create a thought generator function G(pθ, s, k) with two strategies:
   a. Sample i.i.d. thoughts from a CoT prompt.
   b. Propose thoughts sequentially using a "propose prompt".
3. Create a state evaluator function V(pθ, S) with two strategies:
   a. Value each state independently.
   b. Vote across states.
4. Choose a search algorithm (BFS or DFS) based on the tree structure.
5. Implement the chosen search algorithm.
6. Execute the chosen search algorithm with the input problem, thought generator, state evaluator, and other required parameters.

## Usage Examples

### OpenAI API

To use Tree of Thoughts with OpenAI's API, create a custom model class that inherits from `AbstractLanguageModel` and implements the required methods using OpenAI's API. Then, create an instance of the `TreeOfThoughts` class with the custom model and the desired search algorithm ('BFS' or 'DFS').

### Hugging Face Transformers

To use Tree of Thoughts with Hugging Face Transformers, create a custom model class that inherits from `AbstractLanguageModel` and implements the required methods using Hugging Face Transformers. Then, create an instance of the `TreeOfThoughts` class with the custom model and the desired search algorithm ('BFS' or 'DFS').

## Roadmap

Provide ready to use generate thoughts function

Provide ready to use evaluate states function



### Multi-Modality Tree of Thoughts 🌐🌳

The next big advancement for the Tree of Thoughts algorithm is to extend it to multi-modality, enabling it to handle not only text but also images, audio, and other data types. This will bring us closer to multi-modal superintelligence.

#### Actionable Steps

1. Research and identify suitable multi-modal pre-trained models that can handle various data types (e.g., text, images, audio).
2. Adapt the thought decomposition, thought generator, and state evaluator functions to handle multi-modal data.
3. Develop a method for combining different modalities in the search tree, allowing the algorithm to reason across different data types.
4. Implement and test the multi-modal Tree of Thoughts algorithm with various problems and datasets.
5. Optimize the algorithm for performance and resource usage, ensuring it scales well with large multi-modal datasets.
6. Publish the results and gather feedback from the community to further improve the multi-modal Tree of Thoughts algorithm.

Join us on this exciting journey to advance the Tree of Thoughts algorithm to multi-modality superintelligence! 🚀