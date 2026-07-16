# learning-llm-from-scratch

Every LLM mechanism, rebuilt blind from memory in raw PyTorch, then verified numerically against production references.

**The rule: `allclose` or it didn't happen.**

## Method

Each mechanism goes through a 5-step loop:

1. **Blind implementation** — no references, raw PyTorch only. Stuck points become gap notes.
2. **Make it work** — train on a tiny task until loss goes down and outputs are non-degenerate.
3. **Read the theory** — paper/lecture with gap notes in hand. Read for *why*, not *what*.
4. **Diff against production** — compare with PyTorch/HF/trl source. Every difference is an engineering lesson. Fix until `torch.allclose` passes.
5. **Explain it** — README writeup: what it does, why each design choice exists, what breaks without it.

## Index

| # | Mechanism | Category | Status | Blind time | Verified against | One-line takeaway |
|---|-----------|----------|--------|------------|------------------|-------------------|
| 1 | [autograd](foundations/autograd/) | foundations | 🔲 | — | `torch.autograd` | — |
| 2 | [training loop + SGD](training/training-loop/) | training | 🔲 | — | `torch.optim.SGD` | — |
| 3 | [cross-entropy](foundations/cross-entropy/) | foundations | 🔲 | — | `F.cross_entropy` | — |
| 4 | [AdamW](foundations/adamw/) | foundations | 🔲 | — | `torch.optim.AdamW` | — |
| 5 | [BPE tokenizer](tokenization/bpe/) | tokenization | 🔲 | — | HF `tokenizers` | — |
| 6 | [embeddings + RoPE](architecture/rope/) | architecture | 🔲 | — | Llama reference impl | — |
| 7 | [attention (single → multi-head)](architecture/attention/) | architecture | 🔲 | — | `F.scaled_dot_product_attention` | — |
| 8 | [LayerNorm & RMSNorm](architecture/layernorm-rmsnorm/) | architecture | 🔲 | — | `nn.LayerNorm` / Llama RMSNorm | — |
| 9 | [full GPT](architecture/gpt/) | architecture | 🔲 | — | nanoGPT | — |
| 10 | [sampling (greedy/temp/top-k/top-p)](inference/sampling/) | inference | 🔲 | — | HF `generate` | — |
| 11 | [KV-cache](inference/kv-cache/) | inference | 🔲 | — | no-cache generation (identical outputs) | — |
| 12 | [grad accumulation + clipping + LR schedules](training/grad-accumulation/) | training | 🔲 | — | math identity: N steps ≡ batch×N | — |
| 13 | [SFT with prompt masking](post-training/sft/) | post-training | 🔲 | — | trl `SFTTrainer` | — |
| 14 | [DPO](post-training/dpo/) | post-training | 🔲 | — | `trl.DPOTrainer` | — |
| 15 | [GRPO](post-training/grpo/) | post-training | 🔲 | — | trl / verl | — |

Status legend: 🔲 planned · 🟡 in progress · ✅ done (allclose passed) · 🎓 graduated (blind < target time, 3 clean drill passes)

## Structure

```
foundations/        math & optimization primitives
tokenization/       BPE (future: unigram-lm, byte-level)
architecture/       model components (future: GQA, MLA, SwiGLU, MoE)
training/           training mechanics (future: mixed precision, DDP, ZeRO)
inference/          decoding & serving (future: speculative decoding, paged attention)
post-training/      SFT / DPO / GRPO (future: reward models, PPO, KTO)
interpretability/   (future: logit lens, SAEs)
template/           copy this to start any new mechanism
slides/             one review deck per category
drills/             timed drill logs
```

