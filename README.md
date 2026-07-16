# learning-llm-components

Method — read first, then learn by repetition: [docs/method.md](docs/method.md).

## Dependency graph

Category flow, left to right: tokenization → architecture → training → post-training → inference → agent. Numbers match the [index](#index) learning order. Node color = category; the training block folds in the foundations primitives (blue).

```mermaid
flowchart LR
    subgraph TOKENIZE["① tokenization"]
        direction TB
        M1[1 · BPE tokenizer]:::token
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
        M2[2 · autograd]:::found
        M3[3 · cross-entropy]:::found
        M4[4 · training loop + SGD]:::train
        M5[5 · AdamW]:::found
        M12[12 · grad accum / clip / LR]:::train
        M2 --> M3 --> M4 --> M5 --> M12
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

    subgraph AGENT["⑥ agent"]
        direction TB
        M16[16 · tool calling]:::agent
        M17[17 · ReAct loop]:::agent
        M18[18 · multi-step + memory]:::agent
        M16 --> M17 --> M18
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

18 core mechanisms in learning order; the [graph](#dependency-graph) above regroups them by category.

| # | Mechanism | Category | Status | Verified against | One-line takeaway |
|---|-----------|----------|--------|------------------|-------------------|
| 1 | [BPE tokenizer](tokenization/bpe/) | tokenization | 🟡 | HF `tokenizers` | — |
| 2 | [autograd](foundations/autograd/) | foundations | 🔲 | `torch.autograd` | — |
| 3 | [cross-entropy](foundations/cross-entropy/) | foundations | 🔲 | `F.cross_entropy` | — |
| 4 | [training loop + SGD](training/training-loop/) | training | 🔲 | `torch.optim.SGD` | — |
| 5 | [AdamW](foundations/adamw/) | foundations | 🔲 | `torch.optim.AdamW` | — |
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
| 16 | [tool calling](agents/tool-calling/) | agent | 🔲 | HF `transformers` tool chat template | — |
| 17 | [ReAct loop](react_loop/) | agent | 🟡 | `ysymyth/ReAct` | Interleave Thought → Action → Observation until Finish[answer] |
| 18 | [multi-step + memory](agents/memory/) | agent | 🔲 | LangGraph reference loop | — |

Status legend: 🔲 planned · 🟡 in progress · ✅ done (allclose passed)

## Setup

One venv per module, one `.env` at the root.

```bash
cp .env.example .env                          # once per clone: the shared, gitignored secrets file
code learning-llm-components.code-workspace   # always open this — never the repo folder
cd react_loop && uv sync                      # once per module: creates ./.venv from its pyproject.toml
```

Run `uv sync` from the module — the root has no `pyproject.toml`. Open the repo folder and VS Code can't see the module venvs (no interpreter, no kernel); the workspace file makes each module its own root. In notebooks: **Select Kernel → Python Environments → `.venv`**. [Details](docs/environment.md).

Docker available if needed. Data, checkpoints, and logs live on `/workspace` (uncommitted).
