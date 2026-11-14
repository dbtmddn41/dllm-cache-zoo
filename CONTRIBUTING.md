# Contributing

The easiest first contribution is a **method reference**. A researcher who knows a paper can document its cache rule and point readers to the exact upstream implementation. Small corrections to a method description or model claim are welcome.

## Add or correct a method

1. Look for the policy in the [catalog](docs/methods.md). Identify a specific variant, the model family, and the positions or layers whose states it reuses.
2. Edit the summary row and the corresponding entry. Describe how the cache is populated and refreshed, name the knobs that affect comparisons, and link the paper, an upstream commit permalink, and the license evidence. If the license is unspecified, record that observation and link to the revision.
3. Open a focused pull request. Explain which claim a reader can verify from each source. Use **Reference** for a literature entry. The [project scope](docs/project.md) describes the evidence for stronger status claims.

For an integration proposal, describe a real downstream use case, the baseline, and the smallest runnable example that would establish support. A benchmark contribution should include the paired settings and raw records described in the [benchmark protocol](docs/benchmarking.md).

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
