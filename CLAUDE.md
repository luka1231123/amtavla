# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

amtavla is a local-first LLM cognitive system ("second brain") that runs every request through one typed, observable turn loop backed by SQLite memory. It accepts already-parsed text (from CLI, phone bridge, or eventually sub-vocal input) — see `AMTAVLA_ROADMAP.md` for the long-term product vision and `ARCHITECTURE.md` / `README.md` for the current implementation, both of which this file summarizes and should stay consistent with.

## Commands

```bash
# Install
pip install -r requirements.txt
pip install -r requirements-dev.txt   # adds pytest

# Run (two terminals)
venv/bin/python server/phone_server.py   # phone UI, debug UI, memory review API on :8081
venv/bin/python main.py                  # assistant CLI loop

# Or both at once (main.py starts llama-server on launch; ./run kills it on exit)
./run

# Tests (fully offline — fakes model/embedding/search clients, no llama.cpp/Ollama/network needed)
venv/bin/python -m pytest tests
venv/bin/python -m pytest tests/test_orchestrator.py
venv/bin/python -m pytest tests/test_orchestrator.py::test_name -v

# Live conversation probe: scenario suites through the same TurnOrchestrator the
# live app uses, with per-turn gap annotation (needs llama.cpp + Ollama; wipes memory)
python3 raw_full_trace.py
# writes logs/raw_runs/<timestamp>/gap_report.md plus {session_transcript,brain_timeline,memory_chronological,web_searches,idle_processes,thoughts}.txt
```

Isolated memory DB for a scratch dashboard run: `AMTAVLA_CATALOG_DB=/tmp/amtavla-memory.db venv/bin/python server/phone_server.py`.

External runtime deps (not installed via pip): a locally-built `llama.cpp` server with a `.gguf` model in `~/llama.cpp/models`, and Ollama with `nomic-embed-text` pulled for embeddings.

## Architecture

`main.py` owns transport (CLI / Socket.IO) and operator commands. `brain/orchestrator.py` (`TurnOrchestrator`) owns all cognitive control flow for a turn — this split is the main seam in the codebase.

### Turn loop (`brain/orchestrator.py`)

1. Create a typed `Turn` (`brain/contracts.py`) with a stable ID; sample subsystem health.
2. `TurnResolver` (`brain/resolver.py`) rewrites a context-dependent follow-up ("look it up", "the continuation of that phrase") into a standalone request using the recent conversation, storing it as `turn.resolved_input`. Routing, recall, and SEARCH key off the resolved text; permission-gated actions (REMINDER, MEMORY_WRITE, NOTE_READ) and memory commit keep the original `user_input`. Self-contained utterances and explicit commands (`remember…`, `remind me…`) pass through untouched, and a fresh conversation (no recent turns) is never rewritten — this is the "know if we're continuing vs starting" signal. Gated by `routing.resolve_followups_enabled`.
3. `IntentRouter` (`brain/intent_router.py`) returns a `RouteDecision` — hybrid rules/regex + embedding retrieval + LLM rerank, with low-confidence fallback to `planner_full`. `brain_dump` is reachable only through explicit rule phrasing, never fuzzy embedding/LLM routing, so the store is never dumped unprompted.
4. `MemoryController` (`brain/memory_controller.py`) returns a source-aware `ContextPack` (semantic + episodic + insight context, plus a `conversation` buffer of the recent turns for continuity) without doing implicit web work.
5. Pathway determines planning: `DIRECT_PATHWAYS` (`direct_reply`, `creative_reply`, `remember_reply`, `memory_recall_reply`) skip planning; `SINGLE_ACTION_PATHWAYS` (`search_then_reply`, `summarize_reply`, `reminder_reply`, `notes_reply`, `research_reply`) get one deterministic action; everything else goes through `Planner` (`brain/planner.py`), bounded by `max_plan_steps`.
6. `ActionRunner` (`brain/action_runner.py`) executes each `Action` in order, always returning an `ActionResult` (even on failure — failures are structured, not exceptions).
7. `GroundedReasoner` (`brain/reasoner.py`) optionally runs one bounded evidence-synthesis pass between tools and generation — only for question-like, non-direct-pathway turns with at least one successful evidence action or `min_sources` supplied sources (`reasoning.enabled` in config). It emits validated claims tied to real source IDs, which the generator prompt must preserve verbatim rather than re-deriving.
8. `ResponseGenerator` (`generator.py`) builds a source-aware prompt from the plan, action results, context, and any reasoning pass, and replays the recent conversation as chat messages so follow-ups resolve; then generates the response.
9. The completed `Turn` + its `TraceEvent` list are queued for memory commit.
10. Idle workers run background synthesis, promote strong discoveries to insight LTM, and decay/expire episodic memory.

All contracts in `brain/contracts.py` are plain dataclasses that serialize to JSON-safe dicts for persistence/UI (`Turn`, `RouteDecision`, `ContextPack`, `Plan`, `Action`, `ActionResult`, `TraceEvent`, `SourceRef`, `SearchResult`).

### Action surface

| Action | Responsibility | External effect |
| --- | --- | --- |
| `THINK` | Bounded reasoning instruction to generation | None |
| `SEARCH` | Structured web rows via `tools/websearch.py` | Network read |
| `CALCULATE` | Arithmetic via restricted AST | None |
| `MEMORY_SEARCH` | Recall local semantic/episodic/insight context | Local read |
| `MEMORY_WRITE` | Store one explicit durable fact with provenance | Local write |
| `SUMMARIZE` | Collect recent catalog notes as cited material for the generator | Local read |
| `REMINDER` | Parse "remind me..." into a confirmed commitment with `due_at`; a dedicated ~2s reminder tick fires it via the proactive channel | Local write |
| `NOTE_READ` | List/read/find local files via `tools/localfiles.py` (sandboxed root, read-only, size-bounded) | Local read |
| `FILE_WRITE` | Create/overwrite a sandbox text file via `LocalFilesWriter` (`local_files.writable_root`); JSON detail `{path, content}`; reversible — keeps a `.bak` snapshot | Local write |
| `FILE_EDIT` | Targeted find/replace in a sandbox text file; JSON detail `{path, find, replace}`; also snapshots `.bak` | Local write |
| `WEB_FETCH` | Fetch one URL → readable text + `web:<hash>` citation (`tools/webfetch.py`; timeout/size-bounded, scripts stripped) | Network read |
| `FILE_PARSE` | Parse a local PDF/CSV/JSON/text file → citable text (`tools/fileparse.py`, sandboxed; PDF/DOCX degrade gracefully if the optional dep is absent) | Local read |
| `SHELL_RUN` | Run one shell command via `tools/shellrun.py` (timeout + output cap). **T2 — every command requires explicit user approval before it runs** (`tools.shell_run.enabled`) | Shell exec (gated) |
| `CLARIFY` | One clarifying question that becomes the reply verbatim (no generation) | None |
| `RESEARCH` | Queue a bounded background research job (2 searches + 1 synthesis); result arrives proactively | Local write, deferred network |

Unknown planner actions become validation warnings and are never executed. `MEMORY_WRITE` and `REMINDER` are permission-gated: they refuse to run unless the user's own words asked for them (`REMINDER` also accepts self-declared commitments like "I promised X by Friday" — the same phrasing commit-time extraction stores anyway, and both share one dedup key).

### Trust tiers and approvals (`brain/trust.py`, `brain/approvals.py`)

Every action has a trust tier keyed off its worst-case effect (`brain/trust.py`): **T0** read/compute (runs freely), **T1** reversible local write (runs, audited), **T2** outbound/irreversible (requires approval). An unlisted action defaults to T2 — fail closed. When `ActionRunner.run` sees a T2 action it does **not** execute it: it writes a `pending` row to the catalog `approvals` table and returns an `awaiting_approval` result. `ApprovalCoordinator.resolve(id, approved)` records the decision and, only on approval, re-invokes the runner with `approved=True` exactly once (settled approvals never flip; denials never run). Every T2 decision writes an `action_audit` row. `SHELL_RUN` is the first T2 action. The user approves/denies via the `/approvals`, `/approve <id>`, `/deny <id>` operator commands (they work from CLI and phone, since both share the command queue); `main.py` surfaces a pending approval after any turn that produced one. Remaining: phone-UI approve/deny buttons (currently text commands) and proactive delivery of long-running executed results.

Idle-produced messages (due reminders, finished research) reach the user through `MemoryController.set_proactive_hook` — `main.py` wires it to both the CLI and Socket.IO, buffering for the UI while the socket is down. Reminders and overdue research force-starts run on a dedicated ~2s tick (`_reminder_loop`) so they don't depend on the heavier idle pipeline; idle steps in `run_idle_jobs` are individually isolated so one failing step can't starve the rest. Proactive memory-check questions ("should I keep this insight?") travel as a separate `memory_check` message — rendered with yes/no buttons in the UI (`insight_feedback` event → `apply_insight_feedback`), answered by a short yes/no in the CLI — never appended to answer text.

### Source/provenance contract

Every fact or search row carries a stable ID (`memory:semantic:7`, `web:4e65...`). The generator sees these IDs beside supplied facts plus a source catalog, and every completed `Turn` records every source ID that was available to its answer. This is load-bearing for the "every answer knows what context it used" design goal — don't bypass it when adding new context sources.

### Memory (`brain/memory/`, SQLite in `brain/db/`)

- `episodic.db` — short-lived event memory; reinforced on recall (`strength`, `recall_count`, `expiry_at` extension); decays over idle cycles.
- `semantic.db` — indefinite fact store with confidence and provenance.
- `insight_ltm.db` — high-value synthesized discoveries only; supports proactive human confirmation states.
- `jobs.db` — background synthesis/maintenance bookkeeping.
- `ltm_vectors.db` — vector index for long-term insight retrieval (KNN via `sqlite-vec`, with scan fallback if the extension is unavailable).
- `memory_catalog.db` (`brain/memory/catalog.py`) — the **editable source of truth** for recall: unified facts, episodes, decisions, commitments, preferences, ideas, insights, source excerpts, entities (people/places/projects/documents/commitments), relations, tags (`tags`/`tag_assignments`/`tag_feedback`), `capture_events`, `context_snapshots`, and the style profile (in `catalog_meta`).

The pre-catalog databases (episodic/semantic/insight_ltm) remain raw evidence and compatibility stores; the catalog holds derived, editable interpretations. During recall, catalog state wins: corrected content replaces legacy text, and rejected/archived/deleted items are excluded from both unified and legacy retrieval. Automated sync may promote a `candidate` item but can never overwrite a `corrected` item, resurrect `deleted` memory, or downgrade `confirmed` memory. Review states: `candidate`, `confirmed`, `corrected`, `rejected`, `archived`, `deleted`.

`MemoryService` reconciles changed catalog rows into the vector index before the next recall, so dashboard edits (made without loading any models) stay searchable.

Generated `*.db` files, logs, and raw traces are gitignored — don't expect them to persist across a fresh checkout.

### Memory review UI

`server/phone_server.py` exposes `/api/memory` endpoints and the dashboard at `/memory` (search, filter, provenance inspection, correct, confirm, reject, archive, delete, merge, export). Exports land in a timestamped `exports/memory/<timestamp>/` directory as `memory_items.jsonl`, `entities.jsonl`, `relations.jsonl`, `memory.md`, plus a format README, then get zipped for download; deleted records stay in JSONL for audit history.

### Health

`brain/health.py` (`HealthReporter`) tracks model, search, and embedding provider state separately. Search and embedding start `unknown` until exercised — a failed search or a zero embedding vector becomes visibly `unavailable`. `/health` renders this in both CLI and phone UI even when the model itself starts degraded.

### Tagging, executive function, style (Phase 3-5 layer)

- `brain/memory/tagging.py` (`TagEngine`) — heuristic auto-tagging (project/person/location/time) with a feedback loop: user accept/reject/correct decisions in `tag_feedback` boost or suppress future suggestions. Catalog-only, no model calls — the phone server reuses it.
- `brain/memory/context_engine.py` (`ContextEngine`) — infers active project + recent people/locations from per-turn context snapshots and recent tags; feeds tagging and the daily brief.
- `brain/memory/commitments.py` — regex commitment extraction (promise/deadline/confidence) from normal conversation; stored as `commitment` items with `due_at`/`status` metadata. Reminders (overdue / due-soon / active-project) ride the proactive-prompt channel; focus sessions suppress all but overdue.
- Contradictions: same-property facts with conflicting values are flagged on both items (`metadata.contradicts`) — never silently collapsed; the generator prompt requires presenting both. Staleness (`metadata.stale`) is marked during idle jobs for overdue commitments and past-dated items.
- Answers get a readable `Sources:` footer via `render_source_summary` (config `routing.show_sources`).
- Style profile: conversational instructions ("be concise", "never ...") accumulate in `catalog_meta` and are injected into every generation; creative pathway uses `brain/prompts/skills/creative.md`.

### Operator commands (CLI / phone UI)

`/brain [status|ltm|full]`, `/health`, `/ask` (force one proactive insight ask), `/idle` (force an idle cycle now, incl. staleness pass), `/brief` (daily brief), `/loops` + `/done <id>` (open commitments), `/focus [min|off]`, `/review` (spaced drill), `/approvals` + `/approve <id>` + `/deny <id>` (T2 action approvals, e.g. shell commands), `/delete` (wipe raw memory, catalog, entities, jobs, vectors).

Natural-language mode triggers (routed by `IntentRouter`): brain-dump ("tell me what's in your brain"), remember ("remember this...", "don't forget..."), and recall ("where is my car...", "what do you know about...").

### Proactive insight behavior

Deliberately quality-gated and infrequent: turn/time cooldowns, max-ask limits per candidate, snooze windows, quality scoring, and explicit `confirmed`/`rejected`/`snoozed` feedback handling. Config lives in `brain/brain_config.json` under `memory.proactive_*`.

## Testing Notes

`tests/` injects fake model, embedding, search, router, and memory clients (see `_FakeRouter`, `_FakeMemory` patterns in `tests/test_orchestrator.py`) so the full turn loop is testable without llama.cpp/Ollama/network. When adding a new pathway or action type, extend these fakes rather than hitting real subsystems.

## Working Guideline

`WORK_GUIDELINE.md` in this repo defines the expected working style: understand the system before changing it, prefer the smallest clean change over new abstractions, follow existing patterns, and avoid unrelated refactors or style churn. Treat it as binding for any non-trivial change in this repo.
