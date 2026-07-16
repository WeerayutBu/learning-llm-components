# EDITABLE — the OUTER loop rewrites this. EvoX's EvolvedProgramDatabase.


class EvolvedProgramDatabase(ProgramDatabase):   # noqa: F821 — injected by load()
    """The search strategy: which parent to mutate, and what to show alongside it.

    Greedy on purpose. Always the single best program, never any context — the
    fixed strategy EvoX exists to replace.
    """

    def add(self, program):
        self.programs.append(program)

    def sample(self, num_context_programs=4):
        parent = max(self.programs, key=lambda p: p.metrics["score"])
        return parent, []
