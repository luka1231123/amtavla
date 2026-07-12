# Capabilities Roadmap — Toward a Personal Cognitive OS

How amtavla grows from a read-and-remember assistant into a system that can
**read anything, write freely to your own space, and touch the outside world
only with explicit approval.** This is the implementation plan for the
capability catalog; it slots under `AMTAVLA_ROADMAP.md` Phases 1–6 and inherits
its non-negotiables (typed turn loop, provenance, local-first, auditability).

---

## 0. Principles (read before adding any capability)

**Three trust tiers.** Every action is classified by its worst-case side effect.
The tier decides the gate, not the feature.

| Tier | Effect | Gate | Examples |
| --- | --- | --- | --- |
| **T0** | Read / compute, no persistent effect | none — runs freely | search, fetch, calculate, parse, code-eval-for-answer |
| **T1** | Local write to our own store or the user's sandbox | automatic, **reversible**, audited | memory write, file write, reminder, draft |
| **T2** | Leaves the device, spends money, or is irreversible | **approval required** before execution | send message, calendar invite, shell command, delete outside sandbox |

**Reversibility is a feature.** Every T1 write records a before-image (we already
do this via `memory_history`; files get `.bak` snapshots). T2 actions that can't
be rolled back (a sent email) are gated *before* the effect, not after.

**Provenance is preserved end-to-end.** Every action returns `ActionResult` with
`sources`; every artifact it produces carries where it came from. New capabilities
must not bypass the source contract (see `AMTAVLA_ROADMAP.md` "Ideal Answer
Contract").

**Nothing autonomous is unattributable.** Any action taken without a live user
turn (agents, durable jobs) writes a trace row and surfaces a one-line summary at
check-in.

---

## 1. The "add a capability" recipe (every milestone follows this)

Adding a capability is the *same seven steps* every time. The architecture is
already shaped for it — this recipe is why the roadmap is mostly parallelizable.

1. **Contract** — add a member to `ActionType` in `brain/contracts.py`. The
   planner's `SUPPORTED_ACTIONS` is auto-derived from the enum, so the action
   becomes plannable with no planner edit.
2. **Executor** — add one `elif` branch in `ActionRunner.run()`
   (`brain/action_runner.py`) that calls a dedicated tool module and returns a
   structured `ActionResult` (failures are `ok=False`, never exceptions —
   except permission/validation which raise and become structured errors).
3. **Tool module** — implement the actual work in `tools/<name>.py`, sandboxed
   and bounded (mirror `tools/localfiles.py`: resolve every path against an
   allowlisted root, cap sizes, refuse anything outside).
4. **Gate** — T0: none. T1: record a before-image. T2: raise to the approvals
   layer instead of executing (§4).
5. **Prompt** — teach the pathway/planner when to choose it
   (`brain/prompts/…`) and the generator how to report its result
   (`brain/prompts/tasks/generator.md` `<tool_results>` block).
6. **Config** — a feature flag in `brain/brain_config.json` (default off for
   anything T2), following `extraction.model_enabled` / `local_files.root`.
7. **Tests** — extend the offline fakes in `tests/` (fake the tool, assert the
   `ActionResult` shape, gating, and rollback) + one live `memory_eval.py`-style
   ground-truth case.

Exit criterion for any capability: a fake-backed unit test proves the happy
path, the failure path, and the gate; a live probe proves it end-to-end.

---

## 2. Sequencing (dependency-ordered)

```
M1 File write ──┐
M2 Read/ingest ─┼─► M4 Messaging + Calendar ──► M6 Agents & durable jobs
M3 Approvals ───┘        ▲                              ▲
   (T2 unlock) ──────────┴──── M5 Code/shell runner ───┘
```

- **M1, M2** are independent and safe (T1/T0) — start immediately, in parallel.
- **M3 (approvals) is the unlock** for the entire outbound half. Nothing in
  M4/M5/M6 ships without it.
- **M4, M5** depend on M3.
- **M6** composes everything.

---

## 3. Milestones

### M1 — File write / edit / create (T1) · _start here_
**Goal:** turn the read-only note tool into one that produces artifacts.

- **Actions:** `FILE_WRITE`, `FILE_EDIT` (create/overwrite; targeted replace).
- **Files:** `tools/localfiles.py` (add write methods reusing `_safe_path`),
  `action_runner.py` (branches), `contracts.py` (enum), config
  `local_files.writable_root` (separate from read root; default a scratch dir).
- **Gate (T1):** before overwrite, snapshot to `<file>.bak`; record the write in
  a `file_ops` audit row. Refuse writes outside `writable_root`, refuse binary,
  cap size.
- **Provenance:** generated files carry a header/sidecar noting the turn ID and
  sources that produced them.
- **Tests:** write→read round-trip; overwrite creates `.bak`; path-escape
  refused; size cap enforced.
- **Exit:** "save that summary to notes.md" writes a file and can be undone.
- **Effort:** S (1–2 days). No new infra.

### M2 — Read & ingest the world (T0)
**Goal:** make research and "read this" real beyond plain-text local files.

- **Actions:** `WEB_FETCH` (fetch one URL → readable text + citation),
  `FILE_PARSE` (PDF/CSV/docx/JSON → structured text), extend `CALCULATE` with
  unit/currency/time conversion.
- **Files:** `tools/webfetch.py`, `tools/fileparse.py`, `action_runner.py`,
  reuse the existing web source-id scheme (`web:<hash>`).
- **Gate (T0):** none, but enforce fetch timeouts, size caps, and a domain
  allowlist/denylist in config; strip scripts.
- **Provenance:** fetched/parsed content becomes source rows with stable IDs so
  answers cite them, exactly like `SEARCH`.
- **Tests:** fake fetcher returns fixed HTML→text; PDF/CSV fixtures parse; oversize
  refused; citation IDs flow to the answer.
- **Exit:** "read <url> and summarize" and "what's in this PDF" work with sources.
- **Effort:** M (3–4 days).

### M3 — Approvals substrate (the T2 unlock)
**Goal:** a safe boundary so anything that leaves the device pauses for a click.

- **Data model (catalog):** `approvals` (id, turn_id, action_type, payload,
  state, requested_at, decided_at, decision) and `action_audit` (full record per
  §0). Job states from the roadmap: `waiting_for_approval`, `waiting_for_user`,
  `waiting_for_time`.
- **Flow:** a T2 action in `ActionRunner` does **not** execute; it writes an
  `approvals` row (state `pending`) and returns an `ActionResult` that renders as
  "Awaiting your approval to <do X>." The UI (phone/CLI) shows an approve/deny
  control (reuse the `insight_feedback` yes/no channel). On approve, the action
  executes and the result is delivered proactively (reuse
  `set_proactive_hook`).
- **Files:** `brain/memory/catalog.py` (tables), `action_runner.py` (T2 short-
  circuit), `brain/orchestrator.py` (approval-aware turn state),
  `server/phone_server.py` (+ CLI) approval UI.
- **Tests:** a T2 action creates a pending approval and does **not** run; approve
  → runs once; deny → never runs; approval is idempotent (no double-send).
- **Exit:** a fake "send" action is blocked, queued, and only fires on approval.
- **Effort:** L (1 week). Load-bearing — build carefully, test adversarially.

### M4 — Communication + calendar (draft T1 / act T2)
**Goal:** first capabilities that do things in the world, safely.

- **Actions:** `DRAFT_MESSAGE` (T1 — compose, store as artifact, cite context),
  `SEND_MESSAGE` (T2), `CALENDAR_READ` (T0), `CALENDAR_WRITE` (T2).
- **Files:** `tools/connectors/{email,calendar}.py` behind a thin provider
  interface (start with one provider each; keep credentials in OS keychain, never
  in the repo or catalog). `action_runner.py` branches; all T2 route through M3.
- **Gate:** drafting is T1 (reversible artifact). Sending/inviting is T2 → M3
  approval, showing the exact payload (recipient, subject, body / event details).
- **Conflict handling:** `CALENDAR_WRITE` runs a pre-check for overlaps and
  surfaces conflicts before requesting approval (roadmap "conflict resolution").
- **Tests:** draft→approve→send fires exactly once with the shown payload;
  calendar conflict is detected pre-approval; deny path sends nothing.
- **Exit:** "email Nino the invoice" drafts, shows the message, and sends only on
  approval; "am I free Thursday?" reads the calendar.
- **Effort:** L (1–2 weeks, mostly connector/auth plumbing).

### M5 — Sandboxed code / shell runner (T2)
**Goal:** the general-purpose "do any computer task" primitive.

- **Actions:** `CODE_RUN` (evaluate code in a locked-down subprocess for a result),
  `SHELL_RUN` (T2, approval-gated per command).
- **Files:** `tools/coderun.py` — subprocess with no network, temp cwd, CPU/mem/
  time limits, output cap; deny filesystem access outside a scratch dir.
- **Gate:** pure computation (no side effects) can be T0/T1; any command that
  touches the filesystem outside scratch or the network is T2 → M3, showing the
  exact command.
- **Tests:** a compute snippet returns its value; a network attempt is blocked;
  timeout/output caps enforced; a mutating command requires approval.
- **Exit:** "compute X with a script" runs sandboxed; "run <cmd>" asks first.
- **Effort:** L (1 week) — security-sensitive; threat-model before shipping.

### M6 — Specialist agents & durable jobs (roadmap Phase 6)
**Goal:** compose the above into multi-step work that runs over hours/days.

- **Build:** specialist agents (Researcher, Drafter, Scheduler, Memory Librarian,
  Verifier, Privacy Guardian) as bounded units with an explicit contract:
  *input, context pack, allowed tools, budget, output schema, trace.* No single
  omni-agent.
- **Data model:** `agent_runs`, `tool_runs`, `job_checkpoints` (resumable: write
  state after every step); job states from M3.
- **Flow:** the orchestrator dispatches a goal to an agent; the agent uses only
  its allowed tools (each still tier-gated); a **Verifier** pass checks claims/
  sources before any T2 action; check-ins deliver one-line summaries via the
  proactive channel. Extend the existing background `RESEARCH` job as the first
  durable-job template.
- **Tests:** an agent run is fully replayable from its trace; a drafted outbound
  action stops at approval; a killed job resumes from its last checkpoint.
- **Exit:** amtavla runs a multi-day research/scheduling task, reports at a chosen
  time, and never sends without approval.
- **Effort:** XL (multi-week). Do last; it needs M1–M5 stable.

---

## 4. Cross-cutting concerns (apply to every milestone)

- **Audit trail:** every action (T0 included) writes to `action_audit`: user
  request, route, memory used, tool called, output, approval state, result.
  This is the "why did it do that?" record and the replay source.
- **Config flags:** every new capability ships **off by default** if it is T2, on
  if T0/T1 and reversible. One flag per capability, namespaced under `tools`.
- **Testing tiers** (established this cycle — keep them honest):
  - *Tier 1* pytest, offline, deterministic — logic, gating, rollback (~1s).
  - *Tier 2* `memory_eval.py`-style live ground-truth probes — end-to-end on
    messy phrasing (~1 min). Add cases per capability with `must_contain` /
    `must_not_contain` and explicit **approval-required** assertions.
  - *Tier 3* human reads the transcript for what assertions can't encode.
- **Failure honesty:** a failed action returns `ok=False`; the generator must say
  it failed and why (already enforced in `generator.md` `<tool_results>`). Never
  imply success.
- **Privacy Guardian first-class:** before any T2 payload is shown for approval,
  a policy check flags sensitive data (credentials, others' private info) in the
  outbound content.

---

## 5. Definition of done (per milestone)

A milestone is done when: the action(s) are plannable and dispatched; the
correct tier-gate is enforced and tested (including deny/rollback); provenance
flows to the answer/artifact; an audit row is written; Tier-1 and Tier-2 tests
pass; and `README.md` / `ARCHITECTURE.md` / the action table in `CLAUDE.md` are
updated to match.
