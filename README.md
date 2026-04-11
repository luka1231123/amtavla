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

1. Intent detection (`brain/intent_router.py`)
2. Semantic fact extraction and prefetch (`brain/memory/service.py`)
3. Context recall (episodic + semantic + insights + live web)
4. Todo/plan creation (`main.py` + planner)
5. Tool/search execution
6. Response generation (`generator.py`)
7. Episodic commit

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

## Proactive Insight Behavior

Proactive prompts are intentionally infrequent and quality-gated:

- turn/time cooldowns
- max ask count per candidate
- snooze windows
- quality scoring to avoid malformed insight prompts
- explicit user feedback handling (`confirmed`, `rejected`, `snoozed`)

## Modes And Commands

Foreground commands:

- `/brain [status|ltm|full]` - memory debug summary
- `/ask` - force one proactive insight ask
- `/idle` - force idle cycle now (synthesis + decay)
- `/delete` - clear all memory databases (episodic, semantic, insight, jobs, vectors)

Natural-language modes:

- `tell me what's in your brain` / `brain dump` / `show all memory` -> `brain_dump_reply`
- `remember this ...` / `don't forget ...` -> `remember_reply`
- `where is my car ...` / `what did i say ...` / `remind me where ...` -> `memory_recall_reply`

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
python server/phone_server.py
python main.py
```

Open the phone UI at `http://127.0.0.1:8081` and debug dashboard at `http://127.0.0.1:8081/debug`.

## Raw Trace Run

Run the full raw terminal trace script:

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

Run focused tests for the new JSON/vector/LTM components:

```bash
python3 -m pytest tests/test_json_utils.py tests/test_vector_store.py tests/test_memory_ltm_knn.py
```

`pytest` is not pinned in `requirements.txt`; install it separately if needed.

## What Is Still In Progress

- Proactive insight quality still needs calibration in long sessions.
- Intent routing is still config-heavy and can be simplified further.
