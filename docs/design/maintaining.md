# Maintaining the Zoo

## A one-person workflow

The unit of work is one supported pair: a method on a checkpoint, with its example and evidence. Track the next pair in Issues. Work on one pair at a time, and update the method page alongside it. A working, checked entry is what people come to a zoo for. Follow the roadmap order, and move a later item forward only when a user asks for it with a concrete checkpoint. [Roadmap](roadmap.md) · [Comparison protocol](benchmarking.md)

| When | Maintainer action |
| --- | --- |
| During triage | Group requests for the same method. Look for the checkpoint a team uses, a pinned upstream example, and the smallest change that would let that team use the method. |
| For each change | Use a short-lived branch and one focused pull request. State the effect on a researcher, link its evidence, and review the diff after checks pass. A solo maintainer can merge their own reviewed PR. |
| When a source changes | Compare the new revision with the pinned one. Update the affected entry and its evidence in one PR. Tie existing benchmark artifacts to the revision originally measured. |
| When a method or checkpoint is released | Add it to the catalog with its paper, pinned revision and license. Note whether it bundles a decoding rule and which model family it targets. Prompt support for a new release is how people find the Zoo, so prefer a small, correct entry within days over a complete one later. |
| Monthly | Review grouped dependency proposals and prioritize reproducibility reports. Repeat the catalog survey for new cache methods and block diffusion checkpoints, and update the survey date. Check which pairs teams have asked for or linked from downstream repositories. Stars and clones are useful signals of discovery. A reported reproduction or downstream use helps choose what to maintain. |

GitHub Actions checks the Markdown and its local links and installs the locked development environment on Linux. Dependabot proposes grouped monthly updates for actions and Python development dependencies. Once model code exists, the planned PR checks add Ruff, mypy, and pytest on CPU. A GPU run belongs with changes that can affect model behavior, and can be reused to produce the release evidence. [Method catalog](../catalog.md#keep-the-catalog-current)

## Releasing an integration

1. Check that the documented example works from a fresh checkout and that the model, baseline, tests, and support status agree. A measured claim should link the full comparison artifacts.
2. Choose a version and tag the reviewed main-branch commit. Use the GitHub Release notes to name the supported method and checkpoint pairs, user-visible changes, known limitations, and reproduction command. Link the precise commits and any third-party notices.
3. Open the tagged example as a new user would and verify that the artifacts are accessible. Record changes that affect comparison with older results. The first installable release should also document the distribution name and its tested installation command.

## Working with method authors

Each integrated method should have one page a team can send to a colleague. Include the minimal example, the checkpoint it works with, and a matched baseline result when available. Credit the method authors alongside the implementation.

Before publishing a measured result, send the method's authors the result, the settings and the reproduction command, and ask whether the configuration is the one they would choose. Record their reply or the lack of one on the method page. An author who confirms their entry is also the most likely person to contribute the next method. Once the Zoo covers both model families, describe the policy formulation and the matched comparison in a technical report so the Zoo can be cited as a whole. [Citation metadata](../../CITATION.cff) · [Contribution guide](../../CONTRIBUTING.md)
