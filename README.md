# dLLM Cache Zoo

A field guide to key–value (KV) cache policies for diffusion language models, and a project to make those policies interchangeable in an existing research pipeline.

**The project is in its design phase.** You can compare what four policies reuse, read their papers, and jump to a specific revision of each official implementation. The first planned integration pairs LLaDA-8B-Instruct with a full-recomputation baseline and Fast-dLLM v1 prefix caching.

## Compare two policies

Consider a masked diffusion model filling in the middle block of a response. Attention spans the **prompt and finished blocks**, the **active block**, and the **future masked blocks**, in that order.

| Policy | At the start of a block | On later steps in the same block |
| --- | --- | --- |
| Full recomputation | Compute all three regions | Compute all three again |
| Fast-dLLM v1 prefix | Compute all three regions | Reuse the prefix. Recompute the active block and the suffix. |
| Fast-dLLM v1 dual | Compute all three regions | Reuse the prefix and suffix. Recompute the active block. |

The suffix can change its internal representation while its input tokens are still masked. Reusing it saves computation and can change the generated answer. A useful comparison therefore needs both elapsed time and task accuracy. [Authors’ LLaDA implementation](https://github.com/NVlabs/Fast-dLLM/blob/a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b/v1/llada/generate.py)

## Use the project

- **Choosing a method?** Find the reuse rule, refresh rule, supported upstream model families, paper, code, and licensing notes in the [method catalog](docs/methods.md).
- **Designing an experiment?** Use the same prompts and decoding rules in both arms. The [benchmark protocol](docs/benchmarking.md) defines the controls, metrics, and evidence to publish.
- **Building on the Zoo?** Read the [project scope and first milestones](docs/project.md). A method reference is a useful first contribution and takes one Markdown edit. [Contribution guide](CONTRIBUTING.md)

Original project materials are available under Apache-2.0. The catalog links each method to its authors. GitHub’s **Cite this repository** menu provides a software citation. [License](LICENSE) · [Citation metadata](CITATION.cff)
