<schema_name>plan_schema</schema_name>

```json
{
  "steps": [
    {"action": "SEARCH", "detail": "short query"},
    {"action": "CALCULATE", "detail": "arithmetic expression"},
    {"action": "MEMORY_SEARCH", "detail": "memory query"},
    {"action": "MEMORY_WRITE", "detail": "durable fact"},
    {"action": "SUMMARIZE", "detail": "what to summarize (notes scope)"},
    {"action": "REMINDER", "detail": "full reminder request incl. time"},
    {"action": "NOTE_READ", "detail": "list files | read <path> | find <term>"},
    {"action": "CLARIFY", "detail": "one clarifying question"},
    {"action": "RESEARCH", "detail": "research topic"},
    {"action": "THINK", "detail": "reasoning instruction"}
  ],
  "thinking": "brief reasoning about approach"
}
```
