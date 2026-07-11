# Amtavla Architecture

Amtavla is a local-first cognitive assistant. It accepts already-parsed text and runs every normal request through one typed, observable turn loop.

## Runtime Loop

1. `main.py` receives text from the CLI or phone bridge.
2. `TurnOrchestrator` creates a `Turn` with a stable ID and input metadata.
3. Health state is sampled for the model, search, and embedding providers.
4. `IntentRouter` returns a `RouteDecision`.
5. `MemoryController` returns a source-aware `ContextPack` without doing implicit web work.
6. Direct pathways skip planning; search pathways get one deterministic search action; other pathways use `Planner`.
7. `ActionRunner` executes each bounded `Action` in order and returns an `ActionResult` even when the action fails.
8. `GroundedReasoner` optionally runs one bounded evidence-synthesis pass (question-like input, non-direct pathway, sufficient sources) and hands the generator validated claims tied to real source IDs.
9. `ResponseGenerator` receives the plan, action results, context, reasoning pass, and source catalog.
10. The completed turn and its `TraceEvent` list are queued for memory commit.
11. Idle workers consolidate memory, synthesize insights, and apply episodic decay.

`main.py` owns transport and operator commands. `brain/orchestrator.py` owns cognitive control flow.

## Turn Contracts

The contracts in `brain/contracts.py` are plain dataclasses:

- `Turn`: one request, its state, response, source IDs, and trace
- `RouteDecision`: selected intent, pathway, confidence, and decision source
- `ContextPack`: semantic, episodic, insight, and web context plus source references
- `Plan`: bounded actions, planner thinking, and validation warnings
- `Action`: an allowed operation with an ID and detail
- `ActionResult`: structured success, output, error, timing, and sources
- `TraceEvent`: timed inputs, outputs, and source IDs for one turn stage
- `SourceRef` and `SearchResult`: stable provenance records used by generation and debug views

All contracts serialize to JSON-safe dictionaries before persistence or UI emission.

## Action Surface

| Action | Responsibility | External effect |
| --- | --- | --- |
| `THINK` | Pass a bounded reasoning instruction to generation | None |
| `SEARCH` | Retrieve structured web rows | Network read |
| `CALCULATE` | Evaluate arithmetic through a restricted AST | None |
| `MEMORY_SEARCH` | Recall local semantic, episodic, and insight context | Local read |
| `MEMORY_WRITE` | Store one explicit durable fact with provenance | Local write |
| `SUMMARIZE` | Collect recent catalog notes as cited material for generation | Local read |
| `REMINDER` | Store an explicit reminder as a commitment with a resolved due time | Local write |
| `NOTE_READ` | List, read, or find local files under a sandboxed read-only root | Local read |
| `CLARIFY` | Ask one clarifying question; the question is the reply verbatim | None |
| `RESEARCH` | Queue a bounded background research job executed by the idle worker | Local write, deferred network |

Unknown planner actions become validation warnings and are never executed. Action failures remain structured results so generation and traces can represent partial failure honestly. `MEMORY_WRITE` and `REMINDER` require the user's own words to contain an explicit request before they will run; `REMINDER` also accepts a self-declared commitment ("I promised X by Friday"), which the memory layer would capture at commit time regardless.

Due reminders and finished research reach the user through a proactive hook on `MemoryController`: jobs queue messages in a service outbox, and the hook (wired in `main.py`) delivers them to the CLI and the phone UI, buffering for the UI while the socket is down. A dedicated ~2s reminder tick (`_reminder_loop`) fires due reminders and force-starts research jobs that have waited over a minute, independent of the heavier idle pipeline; each `run_idle_jobs` step is isolated so one failing step cannot starve the others. Memory-check questions ("should I keep this insight?") are delivered as a separate `memory_check` message — yes/no buttons in the UI feed `apply_insight_feedback` — instead of being appended to answer text.

## Source Contract

Memory items use IDs such as `memory:semantic:7`. Web rows use stable content-derived IDs such as `web:4e65...`. The generator sees these IDs beside the supplied facts and receives a source catalog. Each completed `Turn` records every source ID made available to its answer.

Search remains structured until prompt assembly. `tool_websearch()` is retained only as a compatibility adapter for older scripts that require rendered text.

## Health Contract

`HealthReporter` exposes separate model, search, and embedding states. Search and embedding begin as `unknown` until exercised; a failed search or zero embedding vector becomes visibly unavailable. The `/health` command renders the current snapshot in the CLI and phone interface, even when the model starts in degraded mode.

## Main Modules

- `main.py`: CLI and Socket.IO transport, operator commands, output delivery
- `brain/orchestrator.py`: complete foreground turn ownership
- `brain/contracts.py`: typed turn and provenance contracts
- `brain/action_runner.py`: bounded action execution
- `brain/health.py`: subsystem health aggregation
- `brain/intent_router.py`: hybrid route selection
- `brain/planner.py`: validated structured planning
- `generator.py`: source-aware prompt assembly and response generation
- `brain/memory_controller.py`: foreground and idle memory coordination
- `brain/memory/service.py`: SQLite memory, recall, provenance, synthesis, and decay
- `tools/websearch.py`: structured search client and compatibility renderer
- `server/phone_server.py`: phone UI, debug UI, and Socket.IO relay

## Storage

Runtime memory lives under `brain/db/`:

- `episodic.db`: turn events and recall logs
- `semantic.db`: durable facts and provenance
- `insight_ltm.db`: candidate and promoted insights
- `jobs.db`: background job bookkeeping
- `ltm_vectors.db`: long-term vector index
- `memory_catalog.db`: editable derived memory, sources, history, entities, and relations

Generated databases, logs, and raw traces are ignored by Git.

## Trusted Memory Layer

`brain/memory/catalog.py` owns the editable Phase 2 memory model. Raw events remain in `episodic.db`; the catalog stores derived interpretations of those events.

Catalog tables:

- `memory_items`: typed content, confidence, importance, review state, version, and merge state
- `memory_sources`: links to events, turns, tool runs, synthesis, imports, and corrections
- `memory_history`: append-only records of edits and review transitions
- `entities`: people, places, projects, ideas, documents, commitments, and organizations
- `memory_item_entities`: typed item-to-entity links
- `relations`: directed entity relationships with optional supporting memory

Review states are `candidate`, `confirmed`, `corrected`, `rejected`, `archived`, and `deleted`. Automated synchronization may promote a candidate, but it cannot overwrite a correction, resurrect deleted memory, or downgrade confirmed memory.

Existing facts, episodes, and insights migrate once. Their original databases remain compatibility and raw-evidence stores. During recall, catalog state wins: corrected content replaces legacy text, while rejected, archived, or deleted items are excluded from unified and legacy retrieval.

The vector repository indexes active memory items and entities. Dashboard edits occur without loading models; `MemoryService` reconciles changed catalog rows into the vector index before the next recall.

## Phase 3-5 Layer: Tagging, Executive Function, Style

`brain/memory/tagging.py` (`TagEngine`) auto-tags items by project, person,
location, and time using regex heuristics, known entities, session context,
and accumulated `tag_feedback` — repeated rejections suppress a tag, accepts
boost it. It is catalog-only (no models), so the dashboard server uses the
same engine. `brain/memory/context_engine.py` (`ContextEngine`) infers the
active project and recent people/locations/topics from context snapshots and
recent tags; a snapshot is persisted per committed turn.

`brain/memory/commitments.py` extracts commitments (promise + deadline +
confidence) from normal conversation; they are stored as `commitment` items
with `due_at`/`status` metadata. `MemoryService` surfaces reminders (overdue,
due-soon, active-project match) through the same channel as proactive asks,
suppressed during focus sessions except when overdue. `daily_brief()` returns
one overdue commitment, one contradiction, and one pattern.

Contradiction detection compares same-property facts ("parked at X" vs
"parked at Y") and flags both sides via `metadata.contradicts` — recall
annotates them, and the generator prompt requires presenting both versions.
Staleness rules run during idle jobs: overdue commitments and past-dated
memories get `metadata.stale`. Answers end with a readable `Sources:` footer
(`render_source_summary`, gated by `routing.show_sources`).

Style instructions in conversation accumulate into a style profile stored in
`catalog_meta` and injected into every generation as a "User Style Profile"
block. Creative replies use `skills/creative.md` to revive related old ideas
and offer divergent directions.

New catalog tables: `tags`, `tag_assignments`, `tag_feedback`,
`capture_events`, `context_snapshots`. Tag review endpoints live under
`/api/memory/items/<id>/tags`, direct note ingestion at `/api/memory/capture`.

## Memory Review API

`server/phone_server.py` exposes local endpoints under `/api/memory` and serves the review dashboard at `/memory`. The dashboard supports search, filtering, provenance inspection, correction, confirmation, rejection, archive, deletion, merge, and export.

`/api/memory/export` writes a timestamped `exports/memory/<timestamp>/` directory containing `memory_items.jsonl`, `entities.jsonl`, `relations.jsonl`, `memory.md`, and a format README, then zips it for download. Deleted records remain in JSONL to preserve audit history.

## Phone Bridge

`server/phone_server.py` also relays operator commands over plain HTTP polling for clients that can't hold a Socket.IO connection: `/command` (POST to enqueue, GET to poll) and `/response` (POST to enqueue, GET to poll) each pair with a `/command/ack` or `/response/ack` endpoint that clears the entry once the other side has consumed it.

## Phase 1 Verification

The test harness injects fake model, embedding, search, router, memory, and response clients. It covers deterministic routes, plan validation, all actions, explicit memory writes, structured web rows, source-aware prompts, health state, action failure, and complete turn commit behavior without requiring local inference services.

Phase 3 can now add tagging, context snapshots, contradiction detection, and richer source presentation without changing the turn, action, or memory review contracts.
