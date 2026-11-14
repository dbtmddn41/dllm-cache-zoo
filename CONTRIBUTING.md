# Contributing

Contributions can bring a new cache method into the Zoo or connect an existing method to another model or inference runtime. During the design phase, a concrete upstream example helps identify the denoising step or model operation where the method can attach.

## Add or correct a method

1. Check the [method table](README.md#method-zoo) for the family and variant. Start from a concrete combination, such as dKV-Cache-Decode on LLaDA-8B-Instruct.
2. Add a variant to its family page or create a page under `docs/methods/`. Follow the Fast-dLLM page as an example. Describe what gets reused, how to select the method, the settings that affect behavior, and any sampler or batch-size requirement. Link the paper, a pinned implementation, an example for the exact checkpoint, and its license evidence. [Example method page](docs/methods/fast-dllm.md)
3. Link the variant in the README and the example in the [checkpoint index](docs/methods.md). A link to an author-maintained implementation belongs in an upstream example column. A Zoo integration should link its method, host-side example, configuration, tested runtime, and checkpoint revision.

For an integration proposal, point to the host loop or native hook that provides the required step information. Describe what the adapter must read or update, which sampler settings it requires, and how a user can switch back to the baseline. Measured comparisons can be contributed independently. [Library design](docs/project.md) · [Benchmark protocol](docs/benchmarking.md)

## Set up a working copy

Install Git and uv, clone the repository, and run the commands below from the checkout. The lockfile prepares Ruff, mypy, and pytest for future Python development. The environment uses Python 3.11 and has no GPU prerequisite. [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)

```sh
git clone https://github.com/dbtmddn41/dllm-cache-zoo.git
cd dllm-cache-zoo
uv sync --locked
```

## Check a documentation change

GitHub checks Markdown style and relative links on each pull request. To run the same checks locally, use markdownlint-cli2 0.24.0 and lychee 0.24.2 from the checkout. The first command uses npm to run the pinned linter. The second uses the lychee binary. [markdownlint-cli2](https://github.com/DavidAnson/markdownlint-cli2) · [lychee installation](https://github.com/lycheeverse/lychee#installation)

```sh
npx --yes markdownlint-cli2@0.24.0 '*.md' 'docs/**/*.md' '.github/**/*.md'
lychee --offline --include-fragments '*.md' 'docs/**/*.md' '.github/**/*.md'
```

Open a pull request for a ready edit or use the method proposal form to discuss scope. [Issue forms](https://github.com/dbtmddn41/dllm-cache-zoo/issues/new/choose) · [Maintainer workflow](docs/maintaining.md)
