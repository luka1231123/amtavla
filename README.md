# amtavla

amtavla is a local-first LLM cognitive system designed to cooperate symbiotically with your brain.

The goal is practical cognitive augmentation: better recall, better planning, grounded answers, and stronger long-term insight formation while you keep control of the loop.

## Symbiotic Design

amtavla treats the assistant as a cooperative cognitive layer, not just a chat bot:

- **Fast intent routing** for immediate response pathway selection
- **Semantic memory prefetch** for known facts/preferences/constraints
- **Recall engine** combining episodic memory, semantic memory, insight memory, and live web grounding
- **Todo/plan generation** from recalled context
- **Execution + response** in the foreground
- **Post-turn episodic write** so every completed turn is traceable
- **Idle background cognition** for synthesis, LTM promotion, and episodic decay

## Brain Loop (Runtime Order)

Each turn runs in this order:

1. Create a typed turn and sample subsystem health (`brain/orchestrator.py`)
2. Select an intent pathway (`brain/intent_router.py`)
3. Recall source-aware semantic, episodic, and insight context
4. Build a validated plan (`brain/planner.py`)
5. Execute bounded actions (`brain/action_runner.py`)
6. Generate against structured results and source IDs (`generator.py`)
7. Commit the response, action results, and complete trace

When idle:

- background synthesis jobs run
- strong discoveries are promoted to insight LTM
- episodic strength decays and expired low-strength events are deleted

## Memory Architecture

Memory is SQLite-backed in `brain/db/`:

- `episodic.db`
  - short-lived event memory
  - reinforced on recall (`strength`, `recall_count`, `expiry_at` extension)
  - decays over idle cycles
- `semantic.db`
  - indefinite fact store with confidence and provenance
- `insight_ltm.db`
  - high-value synthesized discoveries only
  - supports proactive human confirmation states
- `jobs.db`
  - internal synthesis/maintenance bookkeeping
- `ltm_vectors.db`
  - vector index for long-term insight retrieval
  - KNN recall via `sqlite-vec` (with scan fallback if extension is unavailable)
- `memory_catalog.db`
  - unified facts, episodes, decisions, commitments, preferences, ideas, insights, and source excerpts
  - provenance, review history, corrections, merges, archive/delete state
  - people, places, projects, documents, commitments, and entity relations
  - tags, tag assignments (suggested/accepted/corrected/rejected), and tag feedback
  - capture events and per-turn context snapshots
  - style profile (in `catalog_meta`)

The old databases remain raw evidence and compatibility stores. The catalog is the editable source of truth for recall.

## Proactive Insight Behavior

Proactive prompts are intentionally infrequent and quality-gated:

- turn/time cooldowns
- max ask count per candidate
- snooze windows
- quality scoring to avoid malformed insight prompts
- explicit user feedback handling (`confirmed`, `rejected`, `snoozed`)

## Tagging, Capture, and Context (Phase 3)

Every committed turn and captured note is auto-tagged within milliseconds by a
heuristic `TagEngine` (project, person, location, time — no model calls). Tags
are suggestions until reviewed: one-tap accept/reject/correct in the `/memory`
dashboard, and every decision feeds `tag_feedback` so repeated rejections
suppress a tag and accepts boost it. A `ContextEngine` infers the active
project from recent context snapshots and accepted tags, and feeds it back
into tagging and retrieval.

- capture pipeline: every turn logs a `capture_events` row; `POST /api/memory/capture` ingests pasted/voice notes directly
- retrieval filters: by tag, entity, and time window (`what did I say yesterday...` narrows recall to yesterday)
- contradiction detection: facts with the same property but conflicting values (e.g. two parking spots) are flagged on both sides, never silently collapsed — the generator is instructed to present both
- staleness rules: overdue commitments and past-dated memories are marked stale during idle cycles and disclosed in answers
- source-backed answers: replies end with a readable `Sources:` footer showing the memories/web pages actually used (config `routing.show_sources`)

## Executive Function (Phase 4)

- commitments are extracted from normal conversation ("remind me to...", "I promised...", "I need to... by friday") with deadlines resolved to dates
- `/loops` lists open commitments; `/done <id>` closes one
- reminders surface with responses when a commitment is overdue, due within a day, or related to the active project (6h gap between repeats)
- `/focus [minutes|off]` starts a focus session: proactive asks and non-urgent reminders are suppressed; overdue commitments still get through
- `/brief` prints the daily brief: one overdue commitment, one contradiction, one pattern, open-loop count

## Creativity and Learning (Phase 5)

- style instructions in conversation ("be concise", "use bullet points", "never use emoji", "always ...") accumulate into a persistent style profile applied to every generation
- creative replies revive related old `idea` items from memory and offer divergent, numbered directions (kill/merge/continue)
- `/review` runs a spaced review drill over the least-recently-reviewed confirmed memories

## Modes And Commands

Foreground commands:

- `/brain [status|ltm|full]` - memory debug summary
- `/health` - model, search, and embedding provider state
- `/ask` - force one proactive insight ask
- `/idle` - force idle cycle now (synthesis + decay + staleness)
- `/brief` - daily brief (overdue commitment, contradiction, pattern)
- `/loops` - open commitments; `/done <id>` marks one done
- `/focus [minutes|off]` - focus session (suppresses non-urgent prompts)
- `/review` - spaced review drill
- `/delete` - clear raw memory, the editable catalog, entities, jobs, and vectors

Natural-language modes:

- `tell me what's in your brain` / `brain dump` / `show all memory` -> `brain_dump_reply`
- `remember this ...` / `don't forget ...` -> `remember_reply`
- `where is my car ...` / `what did i say ...` / `remind me where ...` -> `memory_recall_reply`
- `what do you know about ...` / `what do you remember about ...` -> `memory_recall_reply`

Intent routing is hybrid: rules/regex + embedding retrieval + LLM rerank, with low-confidence fallback to `planner_full`.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build llama.cpp server

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build
cmake --build build --config Release -j 8
mkdir -p ~/llama.cpp/models
```

Place your `.gguf` model in `~/llama.cpp/models`.

### 3. Embedding model (Ollama)

```bash
ollama pull nomic-embed-text
```

## Run

```bash
# Terminal 1: phone UI, debug UI, and memory review API
venv/bin/python server/phone_server.py

# Terminal 2: assistant loop
venv/bin/python main.py
```

Open:

- assistant UI: `http://127.0.0.1:8081`
- memory review: `http://127.0.0.1:8081/memory`
- runtime debug: `http://127.0.0.1:8081/debug`

The memory dashboard works with only Terminal 1 running. Existing facts, episodes, and insights migrate automatically the first time `main.py` starts.

For an isolated dashboard database:

```bash
AMTAVLA_CATALOG_DB=/tmp/amtavla-memory.db venv/bin/python server/phone_server.py
```

## Raw Trace Run

Run the scripted trace through the same `TurnOrchestrator` used by the live app:

```bash
python3 raw_full_trace.py
```

Artifacts are written to:

- `logs/raw_runs/<timestamp>/session_transcript.txt`
- `logs/raw_runs/<timestamp>/brain_timeline.txt`
- `logs/raw_runs/<timestamp>/memory_chronological.txt`
- `logs/raw_runs/<timestamp>/web_searches.txt`
- `logs/raw_runs/<timestamp>/idle_processes.txt`
- `logs/raw_runs/<timestamp>/thoughts.txt`

## Operational Notes

- Web search uses direct web lookup via `ddgs`.
- `logs/amtavla.log` captures runtime behavior.
- Memory ingestion now filters assistant meta-prompts to reduce semantic pollution.

## Tests

Install development dependencies and run the complete offline suite:

```bash
pip install -r requirements-dev.txt
python3 -m pytest tests
```

The suite uses fake inference, embedding, and search clients. It does not require llama.cpp, Ollama, or network access.

## What Is Still In Progress

- Proactive insight quality still needs calibration in long sessions.
- Phase 3 (tagging, capture, snapshots, contradiction/staleness, source-backed answers) and the pragmatic cores of Phase 4 (commitments, reminders, focus, brief) and Phase 5 (style profile, idea revival, review drills) are implemented; next up is project cockpit views, richer reminder triggers, and learning-item generation.
