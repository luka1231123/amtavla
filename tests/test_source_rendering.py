from brain.contracts import ContextPack, Plan, SourceRef
from generator import build_response_context, render_source_summary


def _pack_with_conflict():
    return ContextPack.from_memory(
        {
            "memory_items": [
                {
                    "id": 1,
                    "item_type": "fact",
                    "content": "car parked at level 3",
                    "review_state": "confirmed",
                    "confidence": 0.9,
                    "updated_at": 200.0,
                    "metadata": {"contradicts": [2]},
                    "tags": [{"tag_type": "location", "name": "garage"}],
                },
                {
                    "id": 2,
                    "item_type": "fact",
                    "content": "car parked at street level",
                    "review_state": "confirmed",
                    "confidence": 0.6,
                    "updated_at": 100.0,
                    "metadata": {"contradicts": [1], "stale": True, "stale_reason": "old"},
                },
            ]
        }
    )


def test_context_renders_conflict_block_and_annotations():
    rendered = build_response_context(Plan(), [], _pack_with_conflict())
    assert "Conflicting Memory" in rendered
    assert "CONFLICTS with item #2" in rendered
    assert "STALE: old" in rendered
    assert "tags: location:garage" in rendered


def test_source_ref_metadata_carries_flags():
    pack = _pack_with_conflict()
    first, second = pack.sources[:2]
    assert first.metadata["contradicts"] == [2]
    assert first.metadata["tags"] == ["location:garage"]
    assert second.metadata["stale"] is True
    assert first.metadata["confidence"] == 0.9


def test_render_source_summary_prefers_cited_sources():
    sources = [
        SourceRef("memory:item:1", "memory_item", "fact", "car at level 3"),
        SourceRef("web:abc123def456", "web", "Parking info", url="https://example.com/p"),
    ]
    cited = render_source_summary("It is at level 3 [memory:item:1].", sources)
    assert "car at level 3" in cited
    assert "example.com" not in cited

    fallback = render_source_summary("No citations here.", sources)
    assert "car at level 3" in fallback
    assert "example.com" in fallback

    assert render_source_summary("anything", []) == ""
