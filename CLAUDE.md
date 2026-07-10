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

# Or both at once (also starts/stops llama-server)
./run

# Tests (fully offline — fakes model/embedding/search clients, no llama.cpp/Ollama/network needed)
venv/bin/python -m pytest tests
venv/bin/python -m pytest tests/test_orchestrator.py
venv/bin/python -m pytest tests/test_orchestrator.py::test_name -v

# Scripted end-to-end trace through the same TurnOrchestrator the live app uses
python3 raw_full_trace.py
# writes to logs/raw_runs/<timestamp>/{session_transcript,brain_timeline,memory_chronological,web_searches,idle_processes,thoughts}.txt
```

Isolated memory DB for a scratch dashboard run: `AMTAVLA_CATALOG_DB=/tmp/amtavla-memory.db venv/bin/python server/phone_server.py`.

External runtime deps (not installed via pip): a locally-built `llama.cpp` server with a `.gguf` model in `~/llama.cpp/models`, and Ollama with `nomic-embed-text` pulled for embeddings.

## Architecture

`main.py` owns transport (CLI / Socket.IO) and operator commands. `brain/orchestrator.py` (`TurnOrchestrator`) owns all cognitive control flow for a turn — this split is the main seam in the codebase.

### Turn loop (`brain/orchestrator.py`)

1. Create a typed `Turn` (`brain/contracts.py`) with a stable ID; sample subsystem health.
2. `IntentRouter` (`brain/intent_router.py`) returns a `RouteDecision` — hybrid rules/regex + embedding retrieval + LLM rerank, with low-confidence fallback to `planner_full`.
3. `MemoryController` (`brain/memory_controller.py`) returns a source-aware `ContextPack` (semantic + episodic + insight context) without doing implicit web work.
4. Pathway determines planning: `DIRECT_PATHWAYS` (`direct_reply`, `creative_reply`, `remember_reply`, `memory_recall_reply`) skip planning; `search_then_reply` gets one deterministic search action; everything else goes through `Planner` (`brain/planner.py`), bounded by `max_plan_steps`.
5. `ActionRunner` (`brain/action_runner.py`) executes each `Action` in order, always returning an `ActionResult` (even on failure — failures are structured, not exceptions).
6. `ResponseGenerator` (`generator.py`) builds a source-aware prompt from the plan, action results, and context, and generates the response.
7. The completed `Turn` + its `TraceEvent` list are queued for memory commit.
8. Idle workers run background synthesis, promote strong discoveries to insight LTM, and decay/expire episodic memory.

All contracts in `brain/contracts.py` are plain dataclasses that serialize to JSON-safe dicts for persistence/UI (`Turn`, `RouteDecision`, `ContextPack`, `Plan`, `Action`, `ActionResult`, `TraceEvent`, `SourceRef`, `SearchResult`).

### Action surface

| Action | Responsibility | External effect |
| --- | --- | --- |
| `THINK` | Bounded reasoning instruction to generation | None |
| `SEARCH` | Structured web rows via `tools/websearch.py` | Network read |
| `CALCULATE` | Arithmetic via restricted AST | None |
| `MEMORY_SEARCH` | Recall local semantic/episodic/insight context | Local read |
| `MEMORY_WRITE` | Store one explicit durable fact with provenance | Local write |

Unknown planner actions become validation warnings and are never executed.

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

`server/phone_server.py` exposes `/api/memory` endpoints and the dashboard at `/memory` (search, filter, provenance inspection, correct, confirm, reject, archive, delete, merge, export). Exports land in `exports/` as `memory_items.jsonl`, `entities.jsonl`, `relations.jsonl`, `memory.md`, plus a format README; deleted records stay in JSONL for audit history.

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

`/brain [status|ltm|full]`, `/health`, `/ask` (force one proactive insight ask), `/idle` (force an idle cycle now, incl. staleness pass), `/brief` (daily brief), `/loops` + `/done <id>` (open commitments), `/focus [min|off]`, `/review` (spaced drill), `/delete` (wipe raw memory, catalog, entities, jobs, vectors).

Natural-language mode triggers (routed by `IntentRouter`): brain-dump ("tell me what's in your brain"), remember ("remember this...", "don't forget..."), and recall ("where is my car...", "what do you know about...").

### Proactive insight behavior

Deliberately quality-gated and infrequent: turn/time cooldowns, max-ask limits per candidate, snooze windows, quality scoring, and explicit `confirmed`/`rejected`/`snoozed` feedback handling. Config lives in `brain/brain_config.json` under `memory.proactive_*`.

## Testing Notes

`tests/` injects fake model, embedding, search, router, and memory clients (see `_FakeRouter`, `_FakeMemory` patterns in `tests/test_orchestrator.py`) so the full turn loop is testable without llama.cpp/Ollama/network. When adding a new pathway or action type, extend these fakes rather than hitting real subsystems.

## Working Guideline

`WORK_GUIDELINE.md` in this repo defines the expected working style: understand the system before changing it, prefer the smallest clean change over new abstractions, follow existing patterns, and avoid unrelated refactors or style churn. Treat it as binding for any non-trivial change in this repo.
