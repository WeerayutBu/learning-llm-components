# <mechanism name>

Recall form — answer from memory, file closed. Answer in your own words *and* a quick mermaid sketch wherever a diagram rereads faster than prose (data flow, tensor shapes, algorithm steps). Any blank → drill it, then log the gap in [gap-journal.md](../gap-journal.md).

- Verified against: <reference.impl> · allclose @ rtol=1e-4: <✅ / —>
- Last reviewed: <date> · recalled: <n/8>

## Questions

0. Brain dump — explain the whole thing in 2–3 sentences, no jargon, from memory.
   —

1. What problem does it solve, and why does it exist?
   —

2. Core mechanism — write the formula / algorithm, with input → output shapes.
   Text: —
   Sketch the forward pass (nodes = ops, edges = tensors with shapes):
   ```mermaid
   flowchart LR
       X["x [B, T, d]"] --> OP["op"] --> Y["y [B, T, d]"]
   ```

3. The one insight that makes it click (why it's built this way)?
   —

4. What did my blind attempt get wrong vs the production reference?
   —

5. The subtle detail the reference gets right that's easy to miss — scaling, masking, numerical stability, dtype, in-place?
   —

6. What did I fix to make it match the reference (allclose @ rtol=1e-4)?
   —

7. What breaks, and how does it fail, without it or if implemented wrong?
   —

8. Topic-specific — the question only this mechanism raises.
   (e.g. attention → why scale by 1/√dₖ? · RoPE → why rotate pairs? · RMSNorm → why drop mean-subtraction? · DPO → why the reference-model KL term?)
   —
