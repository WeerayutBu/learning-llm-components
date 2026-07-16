# Program

The agent's instructions, not the loop's. **This file is the lever — change the goal here.**

## Goal

Make `solve(n)` faster. Lower seconds is better.

It counts the primes below `n`. The sieve in `src/` is the one everyone writes first.

## What the agent sees

- `program.md` — this file, prepended to every prompt. It reads the goal; it cannot change it.
- `src/` — the code under search. **The only thing it may rewrite.**

## What the agent never sees

- `harness/` — `cases.json` (the data) and `measure.py` (the timer). Not hidden for tidiness:
  an agent that can read the verifier games it, and one that can edit it deletes the failing
  case instead of passing it.

## Constraints

- Signature stays `def solve(n)`. `n` is exclusive: count primes `< n`.
- Every case in `harness/cases.json` must pass exactly. A faster wrong answer scores nothing.
- Pure stdlib. No third-party imports, no I/O, no network.

## Stopping

- `MAX_EXPERIMENTS` reached, or no improvement for `PATIENCE` experiments.
