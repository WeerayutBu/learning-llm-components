# Adding a module

Process for an agent adding a mechanism. Read before creating a dir.

## 1. Read first

| where | what |
|---|---|
| `README.md` — index table | source of truth: category, "Verified against", status. Owns the learning order |
| `template/` | copy this |
| `agt_tool_calling/` | worked example of the same shape — if it disagrees with the template, it wins |
| `.env.example` | the one secrets file, at the root |

`ls` before you assume. This repo gets restructured mid-task; files vanish between commands.

Not modules: `questions/` (recall forms), `slides/`.

## 2. Name the dir

`<cat>_<mechanism>` — `tok` `fnd` `trn` `arc` `inf` `pst` `agt`.

**No numbers** — the index owns the order, so adding never renames a dir. Ask if the category
isn't obvious.

## 3. Layout

`README.md`, `main.ipynb`, `pyproject.toml` (+ `uv.lock` once synced), plus any data the
notebook loads. No `questions.md`, no `blind0.ipynb` — pruned.

## 4. Environment

- `uv sync` **from inside the module**.
- The root `pyproject.toml` is a stray copy of `pst_tool_sft`'s. Ignore it.
- **Check `VIRTUAL_ENV`** — it points at a root `.venv`, so `uv sync` installs there and strips
  other modules' deps. It removed `requests` and broke `agt_react_loop`. `deactivate` first.
- Keys: root `.env`, via `load_dotenv(find_dotenv(usecwd=True))`.
- Pin `ipykernel<7` — VS Code's Jupyter can't drive 7.x and blames a missing ipykernel.

## 5. Run it before you write it

**Never write a claim you haven't executed.**

- Every number measured. Paste real output.
- Re-run after changing inputs — removing one name moved `197 -> 189` tokens.
- Read the whole output, not the tail. A broken ReAct env scored *better* by reciting few-shot
  answers; the tail looked perfect.
- Ask what a passing check proved. `bool(os.getenv("KEY"))` proves *set*, not *valid* — the
  placeholder `sk-ant-...` passes it, then 401s.

## 6. README shape

Title → one-liner → *italic intuition* → Category / Verified against / Status → `## Summary`
(mermaid + why / how / without-it) → `## Reference` → `## Weaknesses` → `## Verification` →
`## Run` → `## Links`.

## 7. Weaknesses

A table — `| Weakness | What happens | Fix |` — in the README and the notebook's last cell.
Only what you hit, with the evidence. No fix? `none — <what to expect>`. If a re-run didn't
reproduce it, delete it.

## 8. Prose is the shortest part

Short everywhere: READMEs, markdown cells, replies. Assume your draft is twice as long as it
should be.

- One line of markdown per section; let the code explain itself.
- Cut anything restating the code, the table, or the diagram.
- Cut explanation, never evidence — numbers and quotes stay.
- Drop whole sentences; don't compress into fragments or arrows.

## 9. Close the loop

Index row: link, `🟡`, one-line takeaway. Bump the "N core mechanisms" count.

## 10. Validate

- notebook: valid JSON, code parses, runs top to bottom, no empty cells
- mermaid: balanced `[]` `{}` `""`, no dead ends
- markdown: links resolve; no unescaped `|` in a table cell (`<|im_end|>` splits it)
- no personal data — name, email, paths, keys. Check the Wikipedia `User-Agent` too
- `git status` before deleting anything you didn't create; confirm it's in `HEAD`

## Modules are separate projects

No cross-links. Comparisons belong in the index.

## Ask, don't guess

Category, naming and scope are the owner's calls — a wrong guess costs a repo-wide rename.
Anything a file or a command can settle, settle yourself.
