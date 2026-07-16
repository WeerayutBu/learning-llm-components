# Tool calling

Give the model a typed escape hatch to real code: it asks, you execute.

*A brain in a jar — fluent, and unable to multiply or read a clock. The schema is a menu it
can point at; you are the hands.*

- Category: agent
- Verified against: OpenAI tool-call wire format (via DeepInfra) · scored by exact arithmetic, not allclose
- Status: 🟡 in progress

## Summary

```mermaid
flowchart LR
    Q["Question"] --> M["model + tools=[schema]"]
    M -->|"tool_calls"| X["execute in Python"] --> T["role: tool<br/>keyed by tool_call_id"] --> M
    M -->|"no tool_calls"| A["Answer"]
```

1. **Why** — LLMs predict tokens, they don't compute. Asked for `47281 * 918` the model returned `43295058`, then `43301338` on a re-run; the truth is `43403958`. Fluent, confident, wrong — and wrong differently each time.
2. **How** — pass tools as JSON Schema; the model replies with structured `tool_calls` (name + `arguments`) instead of prose; you run the real function and feed the result back as a `tool` turn. Loop until it stops asking.
3. **Without it** — either fluent wrong answers, or a hand-maintained regex over free text.

## Reference

- `main.ipynb`: the wrong answer, tool + schema, one call, the loop.
- A tool is **two things that must agree** — the Python implementation (the model never sees it) and the schema (the only thing it can reason about). `description` and parameter names are prompt text.

## Weaknesses

- **`finish_reason` is `"stop"`, not `"tool_calls"`** — DeepInfra doesn't set the standard value, so branching on it never fires. Test `message.tool_calls` instead.
- **Template fragments leak into `content`** — seen as `'function=multiply>{…}</function>'` on one run and bare `'>'` on the next. Harmless if you read `tool_calls`, confusing if you print `content`.
- **The schema is prompt, not a contract** — nothing forces the model to call the tool, call it once, or pass sane values. `IMPLS[name](**args)` trusts it completely: a wrong name is a `KeyError`, wrong args a `TypeError`.
- **`arguments` is a JSON *string*** — needs `json.loads`; malformed JSON raises rather than degrading.
- **The `tool` turn must echo `tool_call_id`** — drop it and results can't be matched to requests; with parallel calls it misattributes silently.

## Verification

Exact arithmetic, not `allclose`: the model alone gets `47281 * 918` wrong; with the tool it returns `43403958` exactly. `tool_calls` is checked against the schema (name resolves in `IMPLS`, `arguments` parse to the declared types).

## Run

```bash
cp ../.env.example ../.env   # repo-root .env: add DEEPINFRA_API_KEY (shared, gitignored)
uv sync                      # per-module venv from this pyproject.toml
# run main.ipynb
```

## Links

- [OpenAI function calling](https://platform.openai.com/docs/guides/function-calling) (the wire format DeepInfra implements) · [DeepInfra docs](https://deepinfra.com/docs)
