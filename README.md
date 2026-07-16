# learning-llm-from-scratch

## Loop

Per mechanism — read first, then learn by repetition:

1. Read — study the mechanism and its math from the reference. Skim if familiar, dig in if new.
2. Repeat until fluent — rebuild from memory in blind0.ipynb, then blind1, blind2, … Each pass is a fresh drill, checked against production and fixed. Repetition is where it sticks.
3. Promote & verify — the clean version becomes main.py; allclose @ rtol=1e-4 confirms it.
4. Keep — write the README summary; revisit questions.md later to confirm it stuck.

## Dependency graph

Build order, left to right: tokenization → architecture → training → post-training → inference → agent. Numbers match the [index](#index). Node color = category; the training block folds in the foundations primitives (blue). Agent is future work.

```mermaid
flowchart LR
    subgraph TOKENIZE["① tokenization"]
        direction TB
        M5[5 · BPE tokenizer]:::token
    end

    subgraph ARCH["② architecture"]
        direction TB
        M6[6 · embeddings + RoPE]:::arch
        M8[8 · LayerNorm & RMSNorm]:::arch
        M7[7 · attention]:::arch
        M9[9 · full GPT]:::arch
        M6 --> M7 --> M9
        M8 --> M9
    end

    subgraph TRAIN["③ training"]
        direction TB
        M1[1 · autograd]:::found
        M3[3 · cross-entropy]:::found
        M2[2 · training loop + SGD]:::train
        M4[4 · AdamW]:::found
        M12[12 · grad accum / clip / LR]:::train
        M1 --> M3 --> M2 --> M4 --> M12
    end

    subgraph POST["④ post-training"]
        direction TB
        M13[13 · SFT]:::post
        M14[14 · DPO]:::post
        M15[15 · GRPO]:::post
        M13 --> M14
        M13 --> M15
    end

    subgraph INFER["⑤ inference"]
        direction TB
        M10[10 · sampling]:::infer
        M11[11 · KV-cache]:::infer
    end

    subgraph AGENT["⑥ agent (future)"]
        direction TB
        A1[tool use / function calling]:::agent
        A2[ReAct loop]:::agent
        A3[multi-step + memory]:::agent
        A1 --> A2 --> A3
    end

    %% block-to-block flow
    TOKENIZE ==> ARCH ==> TRAIN ==> POST ==> INFER ==> AGENT

    classDef found fill:#e3f2fd,stroke:#1565c0,color:#0d1b2a
    classDef train fill:#fff3e0,stroke:#e65100,color:#0d1b2a
    classDef token fill:#f3e5f5,stroke:#6a1b9a,color:#0d1b2a
    classDef arch fill:#e8f5e9,stroke:#2e7d32,color:#0d1b2a
    classDef infer fill:#fce4ec,stroke:#c2185b,color:#0d1b2a
    classDef post fill:#ede7f6,stroke:#4527a0,color:#0d1b2a
    classDef agent fill:#e0f2f1,stroke:#00695c,color:#0d1b2a
    style TOKENIZE fill:#fafafa,stroke:#bbb,color:#333
    style ARCH fill:#fafafa,stroke:#bbb,color:#333
    style TRAIN fill:#fafafa,stroke:#bbb,color:#333
    style POST fill:#fafafa,stroke:#bbb,color:#333
    style INFER fill:#fafafa,stroke:#bbb,color:#333
    style AGENT fill:#fafafa,stroke:#bbb,color:#333
```

## Index

15 core mechanisms in build order; the [graph](#dependency-graph) above regroups them by category.

| # | Mechanism | Category | Status | Verified against | One-line takeaway |
|---|-----------|----------|--------|------------------|-------------------|
| 1 | [autograd](foundations/autograd/) | foundations | 🔲 | `torch.autograd` | — |
| 2 | [training loop + SGD](training/training-loop/) | training | 🔲 | `torch.optim.SGD` | — |
| 3 | [cross-entropy](foundations/cross-entropy/) | foundations | 🔲 | `F.cross_entropy` | — |
| 4 | [AdamW](foundations/adamw/) | foundations | 🔲 | `torch.optim.AdamW` | — |
| 5 | [BPE tokenizer](tokenization/bpe/) | tokenization | 🔲 | HF `tokenizers` | — |
| 6 | [embeddings + RoPE](architecture/rope/) | architecture | 🔲 | Llama reference impl | — |
| 7 | [attention (single → multi-head)](architecture/attention/) | architecture | 🔲 | `F.scaled_dot_product_attention` | — |
| 8 | [LayerNorm & RMSNorm](architecture/layernorm-rmsnorm/) | architecture | 🔲 | `nn.LayerNorm` / Llama RMSNorm | — |
| 9 | [full GPT](architecture/gpt/) | architecture | 🔲 | nanoGPT | — |
| 10 | [sampling (greedy/temp/top-k/top-p)](inference/sampling/) | inference | 🔲 | HF `generate` | — |
| 11 | [KV-cache](inference/kv-cache/) | inference | 🔲 | no-cache generation (identical outputs) | — |
| 12 | [grad accumulation + clipping + LR schedules](training/grad-accumulation/) | training | 🔲 | math identity: N steps ≡ batch×N | — |
| 13 | [SFT with prompt masking](post-training/sft/) | post-training | 🔲 | `trl.SFTTrainer` | — |
| 14 | [DPO](post-training/dpo/) | post-training | 🔲 | `trl.DPOTrainer` | — |
| 15 | [GRPO](post-training/grpo/) | post-training | 🔲 | `trl` / `verl` | — |

Status legend: 🔲 planned · 🟡 in progress · ✅ done (allclose passed) · 🎓 graduated (blind < target time, 3 clean drill passes)

## Setup

```bash
uv sync            # create .venv/ from pyproject.toml
uv run python ...  # run inside .venv
```

Data, checkpoints, and logs live on `/workspace` (uncommitted).
