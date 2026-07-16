## Conventions (the contract every mechanism obeys)

- **`blind.py` is frozen.** The Step-1 attempt, never edited after the session. The diff between `blind.py` and `final.py` *is* the learning.
- **`final.py` matches the reference.** Post Step-4, numerically verified.
- **`test_allclose.py` must pass.** `rtol=1e-4` against the production reference. No test, no ✅.
- **`README.md` follows the 7-section template** (see `template/README.md`).
- **Tiny scale always.** `d_model=64`, 4 layers, char-level. Understanding doesn't require GPUs.
- **New mechanism = `cp -r template/ <category>/<name>/`** + a new row in the index above.

## Gap journal

Recurring stuck points, guesses, Step-4 diffs, and bug taxonomy live in [`gap-journal.md`](gap-journal.md). Reviewed weekly.

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
