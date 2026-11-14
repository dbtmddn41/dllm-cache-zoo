# Contributing

Contributions can bring a new cache method into the Zoo, connect an existing method to another model family or host, or add a measured result. During the design stage, a concrete upstream example helps identify the denoising step or model operation where the method can attach.

## Add a method to the catalog

The smallest contribution is one row in the catalog. Give the paper version, a description of what the method stores or skips, the models it ran on, and the code revision with its license. A row needs no method page. [Method catalog](docs/catalog.md)

## Add or correct a method

1. Check the [method table](README.md#method-zoo) and the [method catalog](docs/catalog.md) for the family and variant. Start from a concrete combination, such as dKV-Cache-Decode on LLaDA-8B-Instruct.
2. Add a variant to its family page or create a page under `docs/methods/`. Follow the Fast-dLLM page as an example. Describe what gets reused, how to select the method, the settings that affect behavior, and any sampler or batch-size requirement. If the method ships with its own decoding rule, say which settings belong to the cache and which to decoding. Link the paper, a pinned implementation, an example for the exact checkpoint, and its license evidence. [Example method page](docs/methods/fast-dllm.md)
3. Link the variant in the README and the example in the [checkpoint index](docs/methods.md) under its model family. A link to an author-maintained implementation belongs in an upstream example column. A Zoo integration should link its method, host-side example, configuration, tested host, and checkpoint revision.

## Propose a policy or an adapter

In the planned library, a method is a policy and each model family has an adapter. A policy proposal states which positions the method recomputes between refreshes, when it refreshes, which signals it reads, and what state it stores. Add a row to the design table with those four answers. An adapter proposal points to the host loop or native hook that provides the step information, describes what the adapter must read or update, and explains how a user switches back to the baseline. [How published methods fit](docs/project.md#how-published-methods-fit) · [Hosts](docs/project.md#hosts)

Write a policy from the paper when the reference code has no license that permits reuse, and say so on the method page. When code is adapted from a licensed source, keep its notice and link the revision.

## Contribute a measured result

Measured comparisons can be contributed independently of code. Follow the comparison protocol, name the decoding condition, and attach the run manifest and raw records. A result that differs from a paper is welcome when the settings are stated. [Comparison protocol](docs/benchmarking.md)

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
