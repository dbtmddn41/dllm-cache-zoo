# Design notes

These pages describe how the Zoo is planned to work. Nothing in them is implemented yet, and they change as the milestones land. For what you can use today, start from the method table. [Method table](../../README.md#method-zoo) · [Method catalog](../catalog.md)

| Page | What it covers |
| --- | --- |
| [Roadmap](roadmap.md) | The four milestones, and what the Zoo takes in or leaves to the host. |
| [Library design](library.md) | Cache policies, model adapters and hosts, how the published methods map onto them, and the planned interface. |
| [Comparison protocol](benchmarking.md) | How a method is measured against a matched baseline, and which records are kept. |
| [Serving integration](serving.md) | How a method is carried into an existing inference engine and checked under load. |
| [Maintaining the Zoo](maintaining.md) | The maintainer's workflow for entries, releases and method authors. |

## Propose a policy or an adapter

In the planned library, a method is a policy and each model family has an adapter. A policy proposal states which positions the method recomputes between refreshes, when it refreshes, which signals it reads, and what state it stores. Add a row to the design table with those four answers. An adapter proposal points to the host loop or native hook that provides the step information, describes what the adapter must read or update, and explains how a user switches back to the baseline. [How published methods fit](library.md#how-published-methods-fit) · [Hosts](library.md#hosts)

Write a policy from the paper when the reference code has no license that permits reuse, and say so on the method page. When code is adapted from a licensed source, keep its notice and link the revision.

## Contribute a measured result

Measured comparisons can be contributed independently of code. Follow the comparison protocol, name the decoding condition, and attach the run manifest and raw records. A result that differs from a paper is welcome when the settings are stated. [Comparison protocol](benchmarking.md)
