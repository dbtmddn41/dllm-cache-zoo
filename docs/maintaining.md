# Maintaining the Zoo

## A one-person workflow

Keep the current milestone in the [project roadmap](project.md) and use Issues as the queue. Work on one integration at a time. A reference entry, a model integration, and its measured result can land independently, so a paper correction never has to wait for GPU access.

| When | Maintainer action |
| --- | --- |
| During triage | Group duplicate requests. Prefer a concrete downstream use case, an identifiable upstream revision, and a comparison that fits the available hardware. Point a scoped literature-only contribution to the catalog. |
| For each change | Use a short-lived branch and one focused pull request. State the effect on a researcher, link its evidence, and review the diff after checks pass. A solo maintainer can merge their own reviewed PR. |
| When a source changes | Compare the new revision with the pinned one. Update the affected entry and its evidence in one PR. Tie existing benchmark artifacts to the revision originally measured. |
| Monthly | Review grouped dependency proposals and prioritize reproducibility reports. Check which integrations teams have asked for or linked from downstream repositories. Stars and clones are useful signals of discovery. A reported reproduction or downstream use helps choose what to maintain. |

GitHub Actions checks the Markdown and its local links and installs the locked development environment on Linux. Dependabot proposes grouped monthly updates for actions and Python development dependencies. Once model code exists, the planned PR checks add Ruff, mypy, and pytest on CPU. A GPU run belongs with changes that can affect model behavior, and can be reused to produce the release evidence.

## Releasing an integration

1. Check that the documented example works from a fresh checkout and that the model, baseline, tests, and support status agree. A measured claim should link the full comparison artifacts.
2. Choose a version and tag the reviewed main-branch commit. Use the GitHub Release notes to name the supported policy/model pairs, user-visible changes, known limitations, and reproduction command. Link the precise commits and any third-party notices.
3. Open the tagged example as a new user would and verify that the artifacts are accessible. Record changes that affect comparison with older results. The first installable release should also document the distribution name and its tested installation command.

## Helping another team adopt it

Each integrated policy should have one page a team can send to a colleague. It should include the minimal example, the checkpoint it works with, and a matched baseline result when available. A paper author can contribute a source-backed entry before an adapter exists. GitHub generates a citation from the repository metadata. Experiments using a specific policy should also credit its paper and upstream implementation. [Citation metadata](../CITATION.cff) · [Contribution guide](../CONTRIBUTING.md)
