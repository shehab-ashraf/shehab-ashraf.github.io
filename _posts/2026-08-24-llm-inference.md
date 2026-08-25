---
layout: post
title: "LLM Inference Engine"
subtitle: "Building nano-infer from scratch — baseline, benchmarks, and the road ahead"
date: 2026-08-24
last_updated: 2026-08-25
category: systems
project: "llm-inference"
---

Let's build an LLM inference engine piece by piece, starting from the simplest possible one. This is the work-log where I will document every observation, every bottleneck, and every optimization.

An inference engine isn't magic. At its core, it's about scheduling requests on the CPU and executing them efficiently on the GPU with good kernels. That is exactly what we will do here.

You can follow along with all the code in my repo: [shehab-ashraf/llm-inference](https://github.com/shehab-ashraf/llm-inference).

All benchmarks run on an L4 GPU. The model is [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B).

## In this post

1. [The Engine](#the-engine)
2. [The Baseline](#the-baseline)
   - [Benchmark](#benchmark)

## The Engine
Here is the whole engine:

<p align="center">
  <img src="{{ site.baseurl }}/assets/img/engine.png" alt="nano-infer engine" width="750" />
</p>

**LLM**: what you touch. You give it text, it gives text back. That's it.<br>
**LLMEngine**: the CPU orchestrator. It tokenizes the input and runs the generation loop: prefill once, then decode step by step.<br>
**ModelRunner**: the GPU executor. Owns the model and sampler, runs the forward pass, returns logits. All the heavy math lives here.

## The Baseline

I started with the simplest possible engine that could generate text. No caching, no optimization, just a raw loop.

```python
# Prefill: the full prompt in one shot
logits = self._prefill(input_ids)                                # entire [B, S] through the model
next_token = self._sample_token(logits, sampling_params)         # sample the last position

# Decode: one token at a time
for _ in range(1, sampling_params.max_tokens):
    input_ids = torch.cat((input_ids, next_token.unsqueeze(1)), dim=1)  # grow the sequence
    logits = self._decode_step(input_ids)                               # re-runs the WHOLE sequence again
    next_token = self._sample_token(logits, sampling_params)
```
Inference happens in two phases. First is the prefill. The full prompt goes through the model in one shot, and we sample the first token from the logits of the last position.

Then we move to the second phase: the decode loop, where the tokens are generated one by one. But look closely at how we are doing it right now. We take the entire sequence, append the newly generated token, and run that entire growing sequence through the model again just to get one more token.

Let's see how this engine actually performs.

### Benchmark

The setup: 128-token prompts, generating 256 tokens, batch sizes from 1 to 64.

```text
Qwen3-0.6B | 596M params | torch.bfloat16 | cuda
  28L / 16H / 128D | loaded in 7.6s | 1.13 GB VRAM

Benchmark
Prompt length: 128 | Max tokens: 256

Batch | Time (s) | TTFT (s) | TPOT (s) | VRAM (GB) |    TPS
------------------------------------------------------------
    1 |    7.031 |    0.032 |    0.027 |     1.153 |   36.4
    4 |   10.123 |    0.033 |    0.040 |     1.199 |  101.2
    8 |   21.092 |    0.041 |    0.083 |     1.259 |   97.1
   16 |   50.554 |    0.079 |    0.198 |     1.379 |   81.0
   32 |  124.495 |    0.195 |    0.487 |     1.621 |   65.8
   64 |  275.020 |    0.497 |    1.077 |     2.104 |   59.6
```

Let's see what happens as the batch size goes up.

**Batch 1 to 4:** 36 tok/s at batch 1, 101 tok/s at batch 4. We added 4x the work, but the time only went up a little (7.0s → 10.1s). This means we are memory-bound. The GPU loads 1.1 GB of weights just to do a tiny bit of math. The compute cores finish their work instantly and then just sit there waiting. At batch 4, we load the exact same weights, but we use them on 4 sequences instead of 1. It is the same trip to memory, but we get 4x the tokens. We get this extra speed for free because the cores were just sitting idle anyway.

**After batch 4:** The free speed stops. Batch 8 stays around 97 TPS. By batch 64, throughput drops to 59 TPS and takes over 4.5 minutes. This happens because 64 sequences give the Tensor Cores enough math to keep them fully busy. The bottleneck completely flips. We are no longer waiting on memory. The memory is feeding weights just fine, but now the compute cores are working at maximum capacity. This means we are compute-bound.

**The VRAM:** 1.15 GB at batch 1, only 2.10 GB at batch 64: less than 2x memory for 64x the sequences. The weights dominate, and activations are tiny next to them. Memory is not our problem. The problem is pure waste: we recompute the full attention for every token, at every layer, every step.

This is the baseline. Slow, doing far too much work per token.


The code is on the [baseline branch](https://github.com/shehab-ashraf/llm-inference/tree/baseline).