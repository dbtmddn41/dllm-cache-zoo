# Design notes

These pages describe how the Zoo works and what is planned next. The library, the method policies and the checks against the authors' implementations exist. Measured comparisons, the serving integration and the release process are plans. For what you can use today, start from the method table. [Method table](../../README.md#method-zoo) · [Check report](../checks.md) · [Method catalog](../catalog.md)

| Page | What it covers |
| --- | --- |
| [Roadmap](roadmap.md) | The four milestones and where each stands, and what the Zoo takes in or leaves to the host. |
| [Method selection](method-selection.md) | Which methods are planned and in what order, the evidence behind each choice, and what would bring a deferred method back. |
| [Library design](library.md) | Cache policies, model adapters and hosts, how the published methods map onto them, the interface, and how to add a method. |
| [Comparison protocol](benchmarking.md) | How a method is measured against a matched baseline, and which records are kept. |
| [Serving integration](serving.md) | How a method is carried into an existing inference engine and checked under load. |
| [Maintaining the Zoo](maintaining.md) | The maintainer's workflow for entries, releases and method authors. |

## Propose a policy or an adapter

A method is a policy and each model layout has an adapter. A policy proposal states which positions the method recomputes between refreshes, when it refreshes, which signals it reads, and what state it stores. Add a row to the design table with those four answers, or open a pull request with the policy class. An adapter proposal names the model class and its module layout, and says where its forward differs from the layouts already covered. [How published methods fit](library.md#how-published-methods-fit) · [Add a method](library.md#add-a-method) · [Hosts](library.md#hosts)

Write a policy from the paper when the reference code has no license that permits reuse, and say so on the method page. When code is adapted from a licensed source, keep its notice and link the revision.

## Contribute a measured result

Measured comparisons can be contributed independently of code. Follow the comparison protocol, name the decoding condition, and attach the run manifest and raw records. A result that differs from a paper is welcome when the settings are stated. [Comparison protocol](benchmarking.md)
