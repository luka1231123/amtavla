# amtavla

amtavla is a local-first LLM cognitive system designed to cooperate symbiotically with your brain.

The goal is practical cognitive augmentation: better recall, better planning, grounded answers, and stronger long-term insight formation while you keep control of the loop.

## Symbiotic Design

amtavla treats the assistant as a cooperative cognitive layer, not just a chat bot:

- **Fast intent routing** for immediate response pathway selection
- **Semantic memory prefetch** for known facts/preferences/constraints
- **Recall engine** combining episodic memory, semantic memory, insight memory, and curated web grounding
- **Todo/plan generation** from recalled context
- **Execution + response** in the foreground
- **Post-turn episodic write** so every completed turn is traceable
- **Idle background cognition** for synthesis, LTM promotion, and episodic decay

## Brain Loop (Runtime Order)

Each turn runs in this order:

1. Intent detection (`brain/intent_router.py`)
2. Semantic fact extraction and prefetch (`brain/memory/service.py`)
3. Context recall (episodic + semantic + insights + curated web)
4. Todo/plan creation (`main.py` + planner)
5. Tool/search execution
6. Response generation (`generator.py`)
7. Episodic commit

Deep research requests are handled by a dedicated `research_deep_crawl` intent and run as background jobs.

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
  - background job table (synthesis/maintenance bookkeeping)

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
- `/idle` - force idle cycle now (research jobs + synthesis + decay)
- `/research-status` - list deep research job queue/status
- `/delete` - clear all memory databases (episodic, semantic, insight, jobs)

Natural-language modes:

- `tell me what's in your brain` / `brain dump` / `show all memory` -> `brain_dump_reply`
- `remember this ...` / `don't forget ...` -> `remember_reply`
- `where is my car ...` / `what did i say ...` / `remind me where ...` -> `memory_recall_reply`
- `deep research ...` / `deep dive ...` / `investigate thoroughly ...` -> `research_deep_crawl` (background)

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

## Abilities Test

The test suite intentionally uses one end-to-end abilities run:

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

Artifacts are written to:

- `logs/abilities_runs/<timestamp>/turns.jsonl`
- `logs/abilities_runs/<timestamp>/brain_trace.jsonl`
- `logs/abilities_runs/<timestamp>/errors.jsonl`
- `logs/abilities_runs/<timestamp>/summary.json`

## Operational Notes

- Standard web grounding uses curated sources (Wikipedia + Wikidata adapters).
- Broad multi-source crawl is only used by the `research_deep_crawl` background pathway.
- `logs/amtavla.log` captures runtime web/search behavior and rate-limit patterns.
- Memory ingestion now filters assistant meta-prompts to reduce semantic pollution.

## What Is Still In Progress

- Query normalization for hard natural-language factual questions still needs tuning for edge cases.
- Wikipedia/Wikidata rate-limit handling is present (cooldown), but can still reduce result quality during bursts.
- Proactive insight quality is improved, but long-running sessions need more calibration for better novelty detection.
- Intent exemplars are currently config/keyword-heavy; a dedicated exemplar dataset would improve routing quality further.
