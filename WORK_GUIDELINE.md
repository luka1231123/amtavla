# Agentic AI Work Guideline

## Short Version

Understand the project as a system before changing it. Write a short markdown work guide that maps the structure, goal, risks, and plan. Then make the smallest clean change that solves the real problem. Do not add complexity when it is not necessary.

## Core Rule

System first. Simplicity second. Implementation third.

Good work is calm, organized, and scoped.

## How to Work

### 1. Inspect First

Before editing, learn:

- what the project does
- where it starts
- how data flows
- which files own which responsibilities
- what patterns already exist
- how the project is tested
- what user or local changes must be preserved

Do not guess the architecture. Read it.

### 2. Write the Work Guide

For any non-trivial task, create or update a short markdown guide:

```md
# Work Guide

## Goal

What needs to change.

## System

- Entry points:
- Main modules:
- Data flow:
- State/storage:
- External dependencies:

## Existing Patterns

- Style:
- Tests:
- Error handling:
- UI/API conventions:

## Plan

The smallest clean path.

## Risks

What could break or become messy.

## Verification

How the work will be checked.
```

Keep the guide useful. Do not turn it into a report unless the user asked for one.

### 3. Choose the Smallest Clean Path

Ask:

- What is the actual user goal?
- What already exists that can be reused?
- What is the narrowest safe change?
- What can be avoided?
- Would an abstraction make this clearer, or just bigger?

Do not build a platform when the task needs a function.

### 4. Implement Carefully

- Follow existing project patterns.
- Keep changes scoped.
- Prefer clear names over comments.
- Prefer simple data structures.
- Prefer boring code that works.
- Avoid unrelated refactors.
- Avoid style churn.
- Do not remove user changes.

Complexity must pay rent.

### 5. Verify

Use the lightest reliable check:

- run focused tests
- run lint/type checks if relevant
- manually test user-facing behavior
- inspect the diff
- confirm only intended files changed

If verification cannot be run, say why.

### 6. Summarize

End with:

- what changed
- where it changed
- how it was checked
- any remaining risk

Be concise.

## When to Add Complexity

Add structure only when:

- logic is repeated
- responsibilities are tangled
- tests need clearer boundaries
- future changes are blocked
- the abstraction has a clear name and purpose

Do not add complexity for:

- personal style
- theoretical future needs
- aesthetics alone
- showing sophistication
- avoiding a small amount of ordinary code

## Clean Work Checklist

Before editing:

- I understand the system.
- I know the goal.
- I checked existing patterns.
- I checked the worktree.
- I know the smallest safe change.

Before finishing:

- The change is scoped.
- The code is simple.
- The system is more organized, not less.
- Verification was run or explained.
- The summary is clear.

## Final Standard

Leave the project cleaner, simpler, and easier to understand. If a change does not help the system, do not make it.
