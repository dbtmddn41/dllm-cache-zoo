# Comparing cache policies

Use this protocol when reporting the effect of switching cache methods in the Zoo. For example, on LLaDA-8B-Instruct a team can compare full recomputation, prefix caching, and dual caching on the same GSM8K questions. Each cache run shares the baseline's prompts and decoding rule. [LLaDA model](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) · [GSM8K dataset and answer format](https://github.com/openai/grade-school-math) · [Fast-dLLM settings](methods/fast-dllm.md#select-a-variant-upstream)

## Pair the runs

1. Pin the model, tokenizer, dataset split and revision, prompt template, and scoring rule. Save an ordered list of example IDs and seeds. For a small pilot, publish the subset and its size alongside the score.
2. Use the same device, dtype, attention implementation, batch size, maximum generation length, block length, denoising budget, token-transfer rule, sampling parameters, and stopping rule. Record the actual steps and generated lengths. If confidence-based parallel decoding is also under study, give that change a separate comparison.
3. For isolated model timings, warm up each configuration, measure identical batches, and synchronize the device at the timing boundaries. Alternate run order across repeats and isolate cache state between independent requests. For concurrent serving, timestamp requests at the client and leave the worker's asynchronous execution intact. Record warm-ups and measured repeats.
4. Save per-example outputs and per-batch timing before computing aggregates. Include timeouts, out-of-memory errors, and extraction failures with their IDs. Report how many examples completed in each arm.

## Read the numbers

| Measure | Definition |
| --- | --- |
| Answer accuracy | Correct / attempted on the same IDs. For GSM8K, compare the normalized final numeric answer with the reference. Fix the extraction and normalization rules in advance. Count missing answers as incorrect and give the numerator and denominator. |
| Generation latency | Wall time from input tensors ready on device through the final synchronized generation step. Include cache construction, refreshes, and token selection. Give median batch latency over repeated runs, with batch size and spread. Report cold loading or tokenization separately if timed. |
| Throughput | Sum of actual generated output tokens divided by total timed generation seconds. Use the same EOS convention and exclude prompt and padding tokens. Count each final output position once. |
| Peak memory | Reset the device allocator's peak counters after warm-up and before each measured batch. Report peak allocated memory in MiB including resident model weights. Record peak reserved memory separately when available. |
| Cache effect | Divide baseline time by policy time for the matched workload. If repetitions are used, give the median paired ratio and its spread. Report the accuracy difference in percentage points. Include forward-pass counts and refresh counts to explain changes in time. |

Lower latency can accompany shorter generations or a lower answer score. Present these columns together. Put a paper's reported results in a separately attributed entry with its original setup. A local speedup needs a measured local baseline.

## Evidence to retain

| Record | Minimum contents |
| --- | --- |
| Run manifest | Zoo commit, upstream policy commit, model and tokenizer revisions, dataset revision and IDs, command and resolved settings, seeds, GPU model/count, driver and runtime, library versions, warm-up and repeat counts |
| Per-example record | ID, seed, run and batch identifiers, policy/configuration, generated answer or permitted output artifact, output token count, score or failure, denoising steps and refreshes |
| Per-batch record | Run and batch identifiers, repeat number, synchronized elapsed seconds, peak memory, example IDs and completion status |
| Published comparison | Baseline and policy summaries, sample count, variability across repeats, raw records or a durable artifact link, and instructions to reconstruct the table |

Before publishing, replay the baseline from the recorded settings, check the all-refresh configuration against it, and inspect examples whose scores changed. Preserve the original result when adding a correction or a run with different hardware.

For a deployment claim, also report client-observed tail latency, deadline goodput per GPU, quality, memory and errors under the declared arrival pattern. Publish the stock server, adapter-off and cache-on results together. [Serving workload and release criteria](serving.md#decide-whether-to-enable-it)
