# learning-llm-from-scratch

## GitHub

Every project needs a git `origin` at creation (rtx-config §13):

```bash
gh repo create <org>/learning-llm-from-scratch --private --source=. --remote=origin
git add -A && git commit -m "Initial scaffold"
git push -u origin main
```

## Python environment (uv)

Managed with `uv` — `pyproject.toml` + `uv.lock` define it, `.venv/` is the actual
environment (seeded from rtx-config's base ML stack: torch+cu121, transformers, …):

```bash
uv sync                  # run this first — creates .venv/ (+ uv.lock) from pyproject.toml
uv remove <package> ...  # trim base packages this project doesn't need
uv add <package> ...     # installs + pins project-specific packages
uv run python ...        # runs inside .venv — no activation needed
```

Commit `pyproject.toml` and `uv.lock` together; never `.venv/`.

## Structure
```text
learning-llm-from-scratch/
├── README.md                  # the index (see below) — this replaces tier ordering
├── gap-journal.md
├── template/                  # copy this to start any new mechanism
│   ├── README.md              # the 7-section Step-5 template
│   ├── blind.py
│   ├── final.py
│   └── test_allclose.py
├── slides/
│   ├── foundations.pptx
│   ├── architecture.pptx
│   ├── training.pptx
│   ├── inference.pptx
│   ├── post-training.pptx
│   └── ...                    # new deck when a new category appears
├── foundations/               # math & optimization primitives
│   ├── autograd/
│   ├── cross-entropy/
│   └── adamw/
├── tokenization/
│   └── bpe/                   # later: unigram-lm, byte-level, ...
├── architecture/              # model components
│   ├── attention/             # later: mqa/, gqa/, mla/, sliding-window/
│   ├── rope/                  # later: alibi/, yarn/
│   ├── layernorm-rmsnorm/
│   ├── ffn/                   # later: swiglu/, moe/
│   └── gpt/                   # full assembly
├── training/
│   ├── training-loop/
│   ├── lr-schedules/
│   └── grad-accumulation/     # later: mixed-precision/, ddp/, zero/
├── inference/
│   ├── sampling/
│   └── kv-cache/              # later: speculative-decoding/, paged-attention/, quantization/
├── post-training/
│   ├── sft/
│   ├── dpo/
│   └── grpo/                  # later: reward-model/, ppo/, kto/
├── interpretability/          # empty for now — Tier 5 lands here if you pivot
└── drills/
    └── drill-log.md
```

## Where things live

Only this directory is backed up (via its git `origin`) — everything else lives on
the shared `/workspace` store and is never committed:

| Path | What goes here |
|---|---|
| `/workspace/data/learning-llm-from-scratch/{raw,processed}/` | Datasets |
| `/workspace/models/checkpoints/learning-llm-from-scratch/` | Training checkpoints |
| `/workspace/logs/learning-llm-from-scratch/` | Experiment logs (`WANDB_DIR`) |

Mirror anything irreplaceable to HF Hub/cloud before deleting it (see rtx-config's
`docs/system_design.md` §13/§14).
