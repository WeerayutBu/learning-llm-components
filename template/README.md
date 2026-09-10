# <mechanism name>

<one-line takeaway — the sentence that goes in the index row>

*<intuition — the mental model, not the mechanics. What is this actually like?>*

- Category: <tokenization · foundations · training · architecture · inference · post-training · agent>
- Verified against: <reference.impl> · scored by <allclose @ rtol=1e-4 · EM/F1 · exact value · token counts>
- Status: 🔲 planned · 🟡 in progress · ✅ done

## Summary

**Objective:** <one sentence>

**Input:** <small concrete example>

**Expected output:** <small concrete example>

```mermaid
flowchart LR
    X["input"] --> OP["the mechanism"] --> Y["output"]
```

1. **Why** — <the problem it solves>
2. **How** — <the mechanism, and the one insight that makes it click>
3. **Without it** — <the failure mode it prevents>

## Reference

- `main.ipynb`: <the arc, in the order the cells build it>
- <the parts you'll look up again — formula, shapes, key API>

## Verification

<How you confirmed it matches the reference, and on what inputs. Numbers here are
measured, never estimated.>

## Run

```bash
cp ../.env.example ../.env   # repo-root .env: add DEEPINFRA_API_KEY (shared, gitignored)
uv sync                      # per-module venv from this pyproject.toml
# run main.ipynb
```

## Links

<papers · reference code · docs read>
