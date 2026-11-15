# Contributing

Contributions can bring a new cache method into the Zoo, connect an existing method to another model family or host, or add a measured result. A concrete upstream example helps identify the denoising step or model operation where the method attaches.

## Add a method to the catalog

The smallest contribution is one row in the catalog. Give the paper version, a description of what the method stores or skips, the models it ran on, and the code revision with its license. A row needs no method page. [Method catalog](docs/catalog.md)

## Add or correct a method

1. Check the [method table](README.md#method-zoo) and the [method catalog](docs/catalog.md) for the family and variant. Start from a concrete combination, such as dKV-Cache-Decode on LLaDA-8B-Instruct.
2. Add a variant to its family page or create a page under `docs/methods/`. Follow the Fast-dLLM page as an example. Describe what gets reused, how to select the method, the settings that affect behavior, and any sampler or batch-size requirement. If the method ships with its own decoding rule, say which settings belong to the cache and which to decoding. Link the paper, a pinned implementation, an example for the exact checkpoint, and its license evidence. [Example method page](docs/methods/fast-dllm.md)
3. Link the variant in the README and the example in the [checkpoint index](docs/methods.md) under its model family. A link to an author-maintained implementation belongs in an upstream example column. A Zoo integration should link its method, host-side example, configuration, tested host, and checkpoint revision.

## Add a method to the library

A method is one policy class under `src/dllm_cache_zoo/methods/`. The design notes show the smallest one. [Add a method](docs/design/library.md#add-a-method)

1. Name the paper and the reference revision in the class, and say in the module docstring what was taken from where. Adapt code only from a source whose license permits it, and add it to the notices. When the reference has no license, write the policy from the paper and say so. [Third-party notices](NOTICE)
2. Add CPU tests on the toy models: the method fills every mask, and it matches full recomputation under the settings where its definition says it must. [Method tests](tests/test_methods.py)
3. Compare it with the authors' implementation on a released checkpoint and add the result file. The comparison runs the authors' own loop with their cache, then again with the model call answered by the Zoo, and reports whether the same tokens come out. It takes no timings. [Comparison script](checks/compare_upstream.py) · [Check report](docs/checks.md)
4. Add the method to its page and to the README table, with its settings and anything the Zoo does differently from the reference.

## Take part in the design

A proposal for a cache policy or a model adapter is welcome before any code lands, and so is a measured comparison. The design notes say what each should contain. [Propose a policy or an adapter](docs/design/README.md#propose-a-policy-or-an-adapter) · [Contribute a measured result](docs/design/README.md#contribute-a-measured-result) · [Roadmap](docs/design/roadmap.md)

## Set up a working copy

Install Git and uv, clone the repository, and run the commands below from the checkout. The lockfile provides PyTorch, Ruff, mypy and pytest. The environment uses Python 3.11, and the tests run on CPU in a few seconds. [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)

```sh
git clone https://github.com/dbtmddn41/dllm-cache-zoo.git
cd dllm-cache-zoo
uv sync --locked
uv run pytest -q
uv run ruff check src tests checks examples
uv run mypy
```

## Check a documentation change

GitHub checks Markdown style and relative links on each pull request. To run the same checks locally, use markdownlint-cli2 0.24.0 and lychee 0.24.2 from the checkout. The first command uses npm to run the pinned linter. The second uses the lychee binary. [markdownlint-cli2](https://github.com/DavidAnson/markdownlint-cli2) · [lychee installation](https://github.com/lycheeverse/lychee#installation)

```sh
npx --yes markdownlint-cli2@0.24.0 '*.md' 'docs/**/*.md' '.github/**/*.md'
lychee --offline --include-fragments '*.md' 'docs/**/*.md' '.github/**/*.md'
```

Open a pull request for a ready edit or use the method proposal form to discuss scope. [Issue forms](https://github.com/dbtmddn41/dllm-cache-zoo/issues/new/choose) · [Maintainer workflow](docs/design/maintaining.md)
