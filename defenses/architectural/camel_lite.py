"""CaMeL-lite: control flow from the user, data flow checked at the sinks.

CaMeL (Debenedetti et al., 2025, arXiv 2503.18813) has a privileged LLM write a
program from the user's request alone, runs it in an interpreter that tracks
the provenance of every value, and enforces per-tool policies on where values
may flow. This is the honest-effort subset, and the paper must say exactly that:

IMPLEMENTED
- Control/data separation for side effects: the plan (which side-effecting
  tools may run) comes from the trusted request only, via `planner.py`.
  Untrusted text can never add a side-effecting tool.
- A sink policy: every value reaching a sink argument (`policy.SIDE_EFFECTS`:
  money recipient, email recipients, URL posted to, user invited ...) must
  appear in the user's own request. Untrusted data may fill non-sink slots
  (an email body, an amount) freely.

NOT IMPLEMENTED
- No interpreter and no per-value provenance tags. The sink rule is a literal
  check against the request text, so a legitimate recipient that the user
  only *referred to* ("pay the bill in this file") is refused. CaMeL proper
  can sometimes allow that with an explicit policy; we report the utility cost.
- No quarantined LLM: the agent itself still reads untrusted text and chooses
  arguments. Only the sinks are guarded, so a goal with no sink (deleting a
  file the plan allows deleting) is not stopped by the data rule.
- No side-channel or read-path policies (see `policy.py` on `get_webpage`).
"""

from __future__ import annotations

from typing import Any

from defenses.architectural.allowlist import Allowlist
from defenses.base import DefenseClass, RunContext, Verdict
from defenses.policy import appears_in, sink_values


class CamelLite(Allowlist):
    name = "camel_lite"
    klass: DefenseClass = "C"

    def check_call(self, name: str, args: dict[str, Any], ctx: RunContext) -> Verdict:
        v = super().check_call(name, args, ctx)
        if v.block:
            return v
        untrusted = [s for s in sink_values(name, args) if not appears_in(s, ctx.user_task)]
        if untrusted:
            return Verdict(
                block=True,
                reason=f"sink value(s) {untrusted} not in the user's request",
                extra_tokens=v.extra_tokens,
            )
        return v
