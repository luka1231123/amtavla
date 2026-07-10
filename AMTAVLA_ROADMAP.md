# Amtavla Roadmap: Second Brain Operating System

Research date: 2026-07-10.

## Mission

Amtavla is the software continuation of a sub-vocal recognition project. The input layer will eventually convert silent/sub-vocal human intent into parsed text. This roadmap concerns the software side only: once text arrives, Amtavla should understand the request, recall context, use tools, run agents, update memory, and return the most useful answer or action.

The end goal is a seamless second brain operating system: a private cognitive layer that works with the human to extend memory, reasoning, planning, creativity, execution, learning, and self-organization.

## Research Grounding

The strongest reference systems point in the same direction:

- Vannevar Bush's Memex described a private associative memory machine where a person stores records and retrieves them quickly through linked trails: [As We May Think](https://www.theatlantic.com/magazine/archive/1945/07/as-we-may-think/303881/).
- Douglas Engelbart framed augmentation as a whole human-tool-method system, not only a software feature: [Augmenting Human Intellect](https://www.dougengelbart.org/pubs/papers/scanned/Doug_Engelbart-AugmentingHumanIntellect.pdf).
- J. C. R. Licklider's man-computer symbiosis model keeps the human in charge of goals and evaluation while computers do routinizable preparation work: [Man-Computer Symbiosis](https://groups.csail.mit.edu/medg/people/psz/Licklider.html).
- Microsoft MyLifeBits explored a personal database for everything a person captures or creates: [MyLifeBits](https://www.microsoft.com/en-us/research/publication/mylifebits-a-personal-database-for-everything/).
- Modern PKM tools show the value of local files, links, graphs, and user control: [Obsidian](https://obsidian.md/) and [Logseq](https://github.com/logseq/logseq).
- Modern capture products show demand for ambient memory and meeting recall: [Limitless](https://www.limitless.ai/new).
- PARA/Second Brain methods emphasize actionability over abstract filing: [PARA Method](https://fortelabs.com/blog/para/).
- MemGPT/Letta argues for OS-like memory tiers and virtual context management for long-running LLM agents: [MemGPT paper](https://arxiv.org/abs/2310.08560).
- LangGraph and OpenAI Agents SDK show useful production patterns: persistent state, human-in-the-loop control, tracing, tool loops, handoffs, and resumable approval flows: [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview), [OpenAI Agents guide](https://developers.openai.com/api/docs/guides/agents).

The lesson: Amtavla should not become a chatbot with memory. It should become a user-owned cognitive operating system.

## Design Principles

1. Local-first and user-owned.
2. The system shouldn't replace human cognition, rather help you see blind spots and create friction where necessary.
3. Every answer should know what context it used.
4. Background cognition should help, and proactively interrupt if necessary.
5. The system should organize by active goals, projects, people, and commitments.
6. The core loop should be simple enough to debug.
7. Add intelligence in layers, not as one giant agent.

## Final Target State - Feature List

Input & capture

Accepts parsed text from sub-vocal input in real time, plus typed, voice, and pasted input
Every note auto-tagged with likely project, person, location, and time context within seconds
One-tap correction on wrong tags, with the correction remembered for next time
Continuous ambient capture option (meetings, calls) with transcript + speaker attribution

Memory

Long-term store of facts, conversations, decisions, commitments, people, places, preferences, deadlines, and ideas
Retrieval by meaning, time, entity, location, project, or emotional/contextual state
Every answer shows its sources: which notes, conversations, or tool calls it drew from
Detects stale, outdated, or contradictory memories and flags them instead of silently picking one
Full audit trail of what the system did, when, and why
Local-first storage with optional encrypted sync; full export to plain files at any time

Attention & executive function

Holds working context across the day as the user switches tasks, without needing to be re-briefed
Converts stated intent into concrete next actions automatically
Tracks open loops and commitments made in conversation, not just typed tasks
Reminds based on detected task/context state ("when I'm done with this"), not only fixed times
Daily proactive brief: one overdue commitment, one contradiction, one pattern from recent history
Reduces notification noise; protects declared focus sessions from interruption

Reasoning & planning

Breaks large or vague goals into concrete, ordered next steps
Gathers evidence and compares options with named tradeoffs, not just pros/cons text
Simulates likely outcomes of a decision using stated priorities and past similar decisions
Flags when a current choice resembles a past one the user regretted
Asks a clarifying question when a request is underspecified, instead of guessing silently

Creativity

Maintains a taste/style model (structure, tone, word choice, editing habits) applied to all drafts
Surfaces old, related fragments and ideas unprompted when relevant to new work
Generates multiple divergent directions on request, with a way to merge, kill, or combine them
Maintains long-running creative/idea threads across weeks, including memory of abandoned directions and why they were dropped

Learning

Explains concepts adapted to the user's preferred style and level
Generates summaries, drills, and spaced review from captured material
Tracks weak areas and progress over time
Links new material to relevant existing memory automatically

Execution & agents

Runs specialist agents on request (research, drafting, scheduling, etc.), each producing a concrete output
Multi-day autonomous tasks that continue in the background and report back at a set check-in point
Orchestrates across connected tools/services (calendar, email, messaging) to resolve conflicts or complete tasks
Explicit approval screen before any action that leaves the device (send, post, spend, message)
One-line summary of any autonomous action taken, available on check-in

Self-understanding

Recurring monthly reflection comparing stated goals/values against actual time and behavior, delivered without softening
Identifies recurring behavioral patterns across projects and decisions
Surfaces blind spots and tradeoffs without asserting authority over the user's choices

Presence & interface

Voice interface for spoken, real-time back-and-forth, not just text
Can interrupt proactively, mid-task, when something is time-sensitive or important
Single continuous timeline view combining notes, tasks, calendar, and system conversation
Optional graph view of people/projects/ideas, browsable but not required for normal use
Adjusts its own interaction style over time based on user's observed preference for being pushed vs. left alone

## Optimal Cognitive Loop

Every user turn should pass through one clear loop.

```text
1. Receive parsed text
2. Normalize input
3. Classify intent and urgency
4. Load current session state
5. Recall memory from multiple stores
6. Build a compact working context
7. Decide route:
   - direct answer
   - memory answer
   - tool action
   - research
   - planning
   - multi-agent task
   - clarification
8. Execute route with bounded tools and agents
9. Verify result against sources, memory, and user intent
10. Respond in the right style
11. Write turn trace, facts, tasks, and learnings
12. Queue background synthesis
```

Idle/background loop:

```text
1. Consolidate recent events
2. Extract facts, commitments, decisions, and open loops
3. Link entities across memory
4. Detect contradictions and stale facts
5. Promote durable knowledge
6. Generate useful reminders or questions
7. Prepare project digests
8. Decay or archive low-value memories
9. Update indexes and summaries
```

## Core Architecture

### 1. Input Layer

Responsibility: receive already-parsed text from CLI, phone UI, or future sub-vocal recognition.

Should provide:

- raw text
- timestamp
- device/source
- confidence from upstream recognizer when available
- optional context tags: location, active app, project, conversation mode

Sub-vocal recognition should remain separate. Amtavla should treat it as one input provider.

### 2. Orchestrator

Responsibility: own the turn loop.

Modules:

- intent router
- context builder
- policy/permission engine
- planner
- action runner
- response generator
- memory commit coordinator
- trace emitter

The orchestrator should be deterministic around control flow. LLMs can choose plans, but the application should own execution rules.

### 3. Memory System

Memory should be tiered.

Working memory:

- current conversation
- current task
- active project
- recent user state
- temporary scratchpad

Episodic memory:

- conversations
- events
- decisions
- actions taken
- tool traces
- user feedback

Semantic memory:

- stable facts
- preferences
- constraints
- concepts
- definitions
- user profile facts

Entity graph:

- people
- places
- projects
- documents
- tools
- goals
- commitments
- recurring themes

Project memory:

- active projects
- goals
- next actions
- deadlines
- decisions
- blockers
- related notes
- artifacts

Insight memory:

- patterns detected over time
- recurring behaviors
- synthesis across projects
- strategic observations
- user-approved durable insights

External knowledge cache:

- web results
- documents
- research notes
- citations
- source metadata

Memory must support:

- create
- recall
- update
- correct
- merge
- archive
- forget
- explain provenance
- export

### 4. Retrieval Layer

Recall should combine:

- keyword search
- vector search
- temporal search
- entity graph traversal
- project/task search
- recency and importance ranking
- user feedback signals
- source trust

The output should not be raw memory dumps. It should be a compact context pack:

```text
Relevant facts
Relevant episodes
Relevant entities
Relevant project state
Relevant tasks
Relevant source excerpts
Uncertainties
Potential contradictions
```

### 5. Tool and Agent System

Tools should be typed, permissioned, and auditable.

Core tools:

- web search
- local file search
- calculator
- code runner where safe
- calendar/task connectors
- note/document reader
- memory editor
- reminder scheduler
- browser/research tool
- summarizer

Core agents:

- Memory Librarian: organizes, deduplicates, and corrects memory
- Researcher: searches, reads, cites, and synthesizes
- Planner: breaks goals into projects and next actions
- Executor: performs bounded tool work
- Verifier: checks claims, sources, and outputs
- Coach: helps with habits, reflection, and learning
- Privacy Guardian: enforces permissions and sensitive-data policy
- Project Manager: tracks commitments, deadlines, blockers, and next steps

Agents should be specialists called by the orchestrator. Avoid one huge autonomous agent.

### 6. Permission and Trust Layer

Amtavla should ask before:

- sending messages
- deleting data
- changing files
- making purchases
- publishing content
- sharing private memory
- contacting people
- scheduling externally visible events

Each action should record:

- user request
- selected route
- memory used
- tools called
- outputs
- approval state
- final result

### 7. User Interfaces

Minimum surfaces:

- CLI for development
- phone UI for fast text/sub-vocal workflow
- debug dashboard
- memory review dashboard
- project dashboard
- timeline/replay view
- task/reminder view

Long-term surfaces:

- always-available minimal capture view
- project cockpit
- daily review
- weekly synthesis
- memory graph
- "why do you know this?" inspector
- "forget/correct this" controls everywhere

## Implementation Roadmap

This roadmap is written from the final goal state backward, then converted into the exact build order. Each phase must ship three things together:

- capability: what the user can newly do
- substrate: the data model, services, and APIs required
- trust: inspection, correction, approval, tracing, and tests

A second brain cannot be built as one big agent. It needs a stable event substrate, trustworthy memory, context inference, retrieval, planning, tools, agents, and interfaces layered in that order.

## Final Architecture to Build Toward

The final system should have these major services:

```text
Input Gateway
Capture Pipeline
Context Engine
Memory OS
Retrieval Engine
Task and Commitment Engine
Planner
Action Runner
Agent Runtime
Permission Engine
Proactive Engine
Reflection Engine
Interface Layer
Trace and Eval Layer
Export and Sync Layer
```

The core internal objects should be:

```text
Turn
CaptureEvent
ContextSnapshot
MemoryItem
Entity
Relation
Project
Commitment
Task
Reminder
Source
ContextPack
Plan
ActionRun
AgentRun
Approval
TraceEvent
Reflection
StyleProfile
LearningItem
CreativeThread
```

## Phase 8: Final Cognitive OS

Goal: the complete final state described above.

User-facing result:

- parsed sub-vocal, typed, voice, pasted, and ambient text enter one continuous system
- notes, meetings, decisions, projects, people, tasks, and system conversations live in one timeline
- every answer can show the notes, conversations, memories, and tool calls it used
- the system holds working context across task switches
- it can interrupt when something is genuinely time-sensitive
- it runs multi-day agents with check-ins and approval gates
- it compares goals against behavior and surfaces blind spots
- the user can inspect, correct, delete, and export memory at any time

Implementation requirements:

- local-first storage with optional encrypted sync
- append-only trace log plus editable derived memory
- graph plus vector plus temporal retrieval
- typed action registry and permission engine
- durable background jobs
- continuous eval suite
- timeline, memory, project, task, approval, graph, and reflection interfaces

Do not start here. This is the target that constrains all earlier phases.

## Phase 7: Presence, Reflection, and Adaptive Interaction

Goal: make Amtavla feel like an always-near cognitive layer, not a command box.

Build:

- voice interface for real-time back-and-forth
- interaction-style model tracking whether the user prefers challenge, brevity, detail, interruption, or quiet
- proactive interruption policy with severity levels: silent, digest, prompt, interrupt
- monthly reflection engine comparing stated goals/values against actual timeline, tasks, and behavior
- self-understanding reports that identify patterns without claiming authority over the user's choices
- unified timeline view combining notes, tasks, calendar-like events, tool runs, agent runs, and conversations
- optional graph view for people, projects, ideas, and commitments

Implementation:

- create `interaction_preferences` or `style_profile`
- create `reflections` table with source links and confidence
- create `proactive_events` table with reason, trigger, urgency, status, and feedback
- add a policy scorer: value, urgency, confidence, interrupt cost
- add UI controls: interrupt less, push harder, snooze, wrong, useful
- make every proactive item explainable from its source events

Exit criteria:

- the system can deliver one monthly reflection with cited evidence
- the user can tune how strongly Amtavla pushes
- every interruption has a visible reason and dismissal path

## Phase 6: Long-Running Agents and Cross-Tool Execution

Goal: let Amtavla complete real work over hours or days while remaining safe.

Build:

- specialist agents: Researcher, Drafter, Scheduler, Memory Librarian, Project Manager, Verifier, Privacy Guardian
- durable multi-day jobs with check-in times
- approval screen before anything leaves the device: send, post, message, spend, publish, schedule externally
- one-line summaries of autonomous actions at check-in
- cross-tool orchestration for calendar, email, messaging, documents, browser, and local files
- conflict resolution flows, such as schedule conflicts or duplicated commitments

Implementation:

- create `agent_runs`, `tool_runs`, `approvals`, and `job_checkpoints`
- implement job states: queued, running, waiting_for_approval, waiting_for_user, waiting_for_time, failed, completed, canceled
- define an agent contract: input, context pack, allowed tools, budget, output schema, trace
- define permission manifests per tool
- add resumable execution: every job writes state after each step
- add verifier pass for research, scheduling, and external actions
- add cancellation and rollback hooks where possible

Exit criteria:

- Amtavla can run a background research task and report back at a chosen time
- Amtavla can draft an outbound message but cannot send without approval
- every agent run is replayable from traces

## Phase 5: Creativity, Learning, and Personal Style

Goal: make Amtavla useful for long-running creative and learning work, not just recall.

Status: pragmatic core implemented on 2026-07-11 (persistent style profile learned from conversational instructions and applied to all generation, creative-pathway prompt reviving related idea items with divergent directions, /review spaced drill). Remaining: learning-item generation, weak-area tracking, draft-level keep/revise/combine UI.

Build:

- taste/style model for structure, tone, word choice, editing habits, formatting, and rejection patterns
- creative threads that remember old directions, abandoned ideas, and why they were dropped
- divergent idea generation with merge, kill, combine, and continue controls
- learning items generated from captured material
- summaries, drills, and spaced review
- weak-area and progress tracking
- automatic linking from new material to existing memories

Implementation:

- create `style_profile`, `creative_threads`, `idea_fragments`, `learning_items`, and `review_events`
- extract style signals from accepted edits, rejected drafts, and repeated user instructions
- keep creative decisions as memory items: kept, rejected, merged, abandoned
- implement spaced review scheduler for learning items
- add "connect this to what I know" retrieval mode
- add UI actions on drafts and ideas: keep, revise, combine, discard, remember why

Exit criteria:

- Amtavla can draft in a recognizable user-preferred style
- it can revive a relevant old idea during new work
- it can produce review drills from notes and track weak areas

## Phase 4: Executive Function and Project OS

Goal: convert intent into concrete action and maintain project state across time.

Status: pragmatic core implemented on 2026-07-11 (conversation commitment extraction with deadlines, open loops via /loops and /done, overdue/due-soon/active-project reminders, /focus sessions, /brief daily brief). Remaining: project cockpit views, task states beyond open/done, richer reminder triggers.

Build:

- commitment extraction from conversation, not only explicit tasks
- open-loop detection
- project-aware working context
- next-action generation
- reminders based on context state, not only clock time
- focus-session protection
- daily brief with one overdue commitment, one contradiction, and one pattern
- project cockpit showing goals, next actions, blockers, decisions, and related notes

Implementation:

- create `projects`, `tasks`, `commitments`, `reminders`, `focus_sessions`, and `daily_briefs`
- implement commitment extractor: promise, deadline, owner, project, source, confidence
- implement task states: candidate, accepted, active, waiting, blocked, done, dismissed
- implement reminder triggers: time, project switch, location, active context, "when done with current task"
- implement focus policy: suppress low-value prompts, allow urgent prompts
- add project linker using tags, entity mentions, recency, and correction feedback
- add daily brief generator with strict caps to avoid noise

Exit criteria:

- Amtavla can detect a commitment made in normal conversation
- it can show open loops by project
- it can remind based on context, not only fixed time
- it can protect declared focus sessions

## Phase 3: Context, Tagging, and Source-Aware Retrieval

Goal: make capture and recall accurate enough for the final memory features.

Status: implemented on 2026-07-11 (heuristic TagEngine/ContextEngine, capture_events, context_snapshots, tag feedback loop, contradiction/staleness flags, source-backed answer footers, tag/entity/time retrieval filters).

Build:

- capture pipeline for parsed text, typed input, pasted input, voice transcripts, and ambient transcripts
- auto-tagging by likely project, person, location, and time context within seconds
- one-tap correction for wrong tags, remembered for future inference
- retrieval by meaning, time, entity, location, project, and emotional/contextual state
- source-backed answers showing notes, conversations, and tool calls used
- stale and contradictory memory detection

Implementation:

- create `capture_events`, `tags`, `tag_assignments`, `tag_feedback`, `sources`, and `context_snapshots`
- add tag candidates with confidence and status: suggested, accepted, corrected, rejected
- create a `ContextEngine` that combines recency, active project, user correction history, entities, location, and current session
- create a `RetrievalEngine` that returns structured `ContextPack` objects
- index every memory item by keyword, vector, entity, time, project, source, and importance
- implement contradiction checks for facts with same entity/property but conflicting values
- implement staleness rules for time-sensitive facts and commitments
- make response generation cite source IDs internally and render readable source summaries

Exit criteria:

- every captured note gets suggested tags quickly
- user corrections improve later tag choices
- answers can show which memories or tool calls were used
- contradictory memories are flagged instead of silently collapsed

## Phase 2: Trusted Memory OS

Goal: replace fragile memory accumulation with inspectable, editable, user-owned memory.

Status: implemented and verified on 2026-07-10.

Build:

- unified memory item model for facts, episodes, decisions, commitments, preferences, ideas, insights, and source excerpts
- memory review dashboard
- correction, merge, archive, delete, and export
- provenance on every memory
- entity graph for people, places, projects, ideas, documents, commitments
- plain-file export at any time
- optional encrypted sync later, not before local export works

Implementation:

- add `memory_items` as the primary derived-memory table
- add `entities` and `relations`
- keep raw turns/events append-only; make memory items derived and editable
- migrate existing semantic facts and insights into memory item views or adapters
- add source links from memory item to event, turn, note, transcript, or tool run
- add review states: candidate, confirmed, corrected, rejected, archived, deleted
- add memory APIs: list, search, inspect, correct, merge, archive, delete, export
- add vector nodes for memory items, episodes, entities, and projects
- add tests for explicit memory, correction, forgetting, provenance, and recall

Exit criteria:

- user can ask "what do you know about X?"
- user can correct or delete the answer's memory source
- every memory item can explain where it came from
- memory can be exported without proprietary lock-in

Delivered:

- unified catalog for facts, episodes, decisions, commitments, preferences, ideas, insights, and source excerpts
- source links and immutable history for creation, correction, review changes, and merges
- automatic migration adapters for existing semantic facts, episodes, and insights
- review rules that automated reinforcement cannot downgrade or resurrect
- correction, confirmation, rejection, archive, soft delete, merge, inspect, search, and export APIs
- entities, item links, typed relations, and vector nodes for memory, episodes, entities, and projects
- catalog-first recall that suppresses stale legacy records after review changes
- local `/memory` dashboard with filters, provenance, history, editing, merge, and ZIP export
- lossless JSONL plus readable Markdown export

## Phase 1: Reliable Turn Loop and Action Substrate

Goal: make the current assistant loop clean enough to support all later phases.

Status: implemented and verified on 2026-07-10.

Build:

- deterministic turn object
- structured context pack
- aligned planner/action schema
- action runner
- structured web/search results
- source-aware response contract
- model/search/embedding health reporting
- focused full-turn eval harness

Implementation:

- introduce dataclasses or typed dicts: `Turn`, `RouteDecision`, `ContextPack`, `Plan`, `Action`, `ActionResult`, `TraceEvent`
- align planner prompt, schema, parser, and executor so unsupported actions cannot be silently dropped
- keep initial actions simple: `THINK`, `SEARCH`, `CALCULATE`, `MEMORY_SEARCH`, `MEMORY_WRITE`
- make `tool_websearch` return structured rows internally, then render text only at the response boundary
- make embedding failure visible in health state; do not silently treat zero vectors as good retrieval
- create fake LLM, fake embedder, and fake search clients for tests
- add route, plan, action, memory-write, and full-turn eval tests
- update debug events to include timing, inputs, outputs, and source IDs

Exit criteria:

- tests run without local llama.cpp or Ollama
- every turn has a trace
- every action result is structured
- every generated answer can state what context it used

Delivered:

- typed turn, route, context, plan, action, result, source, search, and trace contracts
- dependency-injected turn orchestrator with one deterministic control-flow owner
- bounded `THINK`, `SEARCH`, `CALCULATE`, `MEMORY_SEARCH`, and `MEMORY_WRITE` actions
- structured search rows with stable source IDs and a compatibility text renderer
- source-aware response prompts and per-turn response source IDs
- model, search, and embedding health snapshots plus `/health`
- fake-client route, planner, action, memory, generator, and full-turn evaluations

## Phase 0: Current Stabilization

Goal: protect the current project while the architecture is being prepared.

Current strengths:

- local llama.cpp response path
- Ollama embeddings
- hybrid intent routing
- SQLite episodic, semantic, insight, jobs, and vector stores
- web search through DDGS
- phone UI and debug dashboard
- idle memory synthesis and decay
- proactive insight confirmation
- raw trace script
- focused vector/JSON/LTM tests

Immediate implementation tasks:

- keep this roadmap and `WORK_GUIDELINE.md`
- ignore generated raw traces and rotated logs
- document current architecture briefly
- add tests around current memory behavior before refactoring
- fix explicit `remember/save/don't forget` storage for non-first-person facts
- add planner parsing tests
- add proactive feedback transition tests
- add memory pollution filter tests

Exit criteria:

- generated files do not pollute git status
- current behavior is documented
- the first refactors have test coverage

## Goal-State Coverage Map

Input and capture:

- Phase 1 defines turn and trace objects
- Phase 3 implements capture events, auto-tagging, corrections, and context snapshots
- Phase 7 adds real-time voice/presence interfaces

Memory:

- Phase 2 implements user-owned memory, provenance, correction, deletion, export
- Phase 3 implements source-aware retrieval, contradictions, staleness, and context packs

Attention and executive function:

- Phase 4 implements commitments, open loops, reminders, focus protection, and daily briefs
- Phase 7 implements proactive interruption policy

Reasoning and planning:

- Phase 1 implements structured plans and actions
- Phase 6 implements specialist agents, verifier, and cross-tool work

Creativity:

- Phase 5 implements style profile, creative threads, divergent directions, and idea history

Learning:

- Phase 5 implements summaries, drills, spaced review, weak-area tracking, and memory linking

Execution and agents:

- Phase 6 implements durable jobs, permissions, approvals, check-ins, and tool orchestration

Self-understanding:

- Phase 7 implements monthly reflection and behavior-vs-goals comparison

Presence and interface:

- Phase 3 starts source display and memory inspection
- Phase 4 adds project/task dashboards
- Phase 6 adds approval and agent-run views
- Phase 7 adds timeline, graph, voice, interruption, and reflection views

## Data Model Direction

Long term, memory should move toward these tables or collections:

```text
events
memory_items
entities
relations
projects
tasks
sources
documents
tool_runs
agent_runs
approvals
summaries
embeddings
```

Important fields:

```text
id
type
content
status
confidence
importance
source_id
provenance
created_at
updated_at
last_used_at
expires_at
user_feedback
embedding_id
entity_links
project_links
privacy_level
```

## The Ideal Answer Contract

Amtavla responses should be:

- context-aware
- concise unless depth is requested
- clear about uncertainty
- grounded in memory or sources when needed
- action-oriented when the user wants movement
- respectful of privacy and permissions

For non-trivial tasks, the response should internally answer:

```text
What did the user ask?
What context matters?
What memories were used?
What tools were used?
What is the best answer or action?
What should be remembered?
What should happen later?
```

The user should see only what helps.

## Non-Negotiables

- User can inspect memory.
- User can correct memory.
- User can delete memory.
- User can export memory.
- Tool actions are traceable.
- Sensitive actions require approval.
- Background agents are interruptible.
- The system fails visibly, not silently.
- The assistant stays useful even without cloud services.

## Product North Star

Amtavla should feel like a quiet cognitive exoskeleton:

- it remembers without becoming creepy
- it acts without stealing control
- it thinks in the background without making noise
- it organizes around the user's goals
- it helps the user become more capable, not more dependent
