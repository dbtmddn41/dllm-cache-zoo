# Maintaining the Zoo

## A one-person workflow

Track the next usable method or model configuration in Issues. Work on one integration at a time, and update the method page alongside it. A team should be able to adopt an integrated method as soon as its example works. Measurements and support for another checkpoint can follow independently. [First release target](project.md#proposed-first-release)

| When | Maintainer action |
| --- | --- |
| During triage | Group requests for the same method. Look for the checkpoint a team uses, a pinned upstream example, and the smallest change that would let that team use the method. |
| For each change | Use a short-lived branch and one focused pull request. State the effect on a researcher, link its evidence, and review the diff after checks pass. A solo maintainer can merge their own reviewed PR. |
| When a source changes | Compare the new revision with the pinned one. Update the affected entry and its evidence in one PR. Tie existing benchmark artifacts to the revision originally measured. |
| Monthly | Review grouped dependency proposals and prioritize reproducibility reports. Check which integrations teams have asked for or linked from downstream repositories. Stars and clones are useful signals of discovery. A reported reproduction or downstream use helps choose what to maintain. |

GitHub Actions checks the Markdown and its local links and installs the locked development environment on Linux. Dependabot proposes grouped monthly updates for actions and Python development dependencies. Once model code exists, the planned PR checks add Ruff, mypy, and pytest on CPU. A GPU run belongs with changes that can affect model behavior, and can be reused to produce the release evidence.

## Releasing an integration

1. Check that the documented example works from a fresh checkout and that the model, baseline, tests, and support status agree. A measured claim should link the full comparison artifacts.
2. Choose a version and tag the reviewed main-branch commit. Use the GitHub Release notes to name the supported policy/model pairs, user-visible changes, known limitations, and reproduction command. Link the precise commits and any third-party notices.
3. Open the tagged example as a new user would and verify that the artifacts are accessible. Record changes that affect comparison with older results. The first installable release should also document the distribution name and its tested installation command.

## Helping another team adopt it

Each integrated method should have one page a team can send to a colleague. Include the minimal example, the checkpoint it works with, and a matched baseline result when available. Credit the method authors alongside the implementation. [Citation metadata](../CITATION.cff) · [Contribution guide](../CONTRIBUTING.md)
