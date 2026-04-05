from brain.ltm_tree import (
    LtmTree,
    _safe_embed,
    _cosine_similarity,
    _safe_chat,
    MAX_DEPTH,
)

MODEL = "qwen2.5-coder:1.5b"


def summarize_for_stm(user_input, response):
    prompt = f"""
Summarize this conversation turn into ONE line (under 20 words) for short-term memory.
Capture key facts, tasks, technical details, or user preferences.

User: {user_input}
Assistant: {response}

Output ONLY the summary line. No labels, no prefixes.
"""
    snippet = _safe_chat([{"role": "user", "content": prompt}], model=MODEL)
    print(f"   [DEBUG-CONSOLIDATOR] -> STM snippet: {snippet}")
    return snippet


def detect_topic_shift(user_input, current_branch: dict | None):
    if current_branch is None:
        return True, ""

    input_embedding = _safe_embed(user_input)
    branch_embedding = current_branch["embedding"]
    sim = _cosine_similarity(input_embedding, branch_embedding)

    if sim > 0.56:
        return False, current_branch["topic"]

    STOP_WORDS = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "used",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "don",
        "now",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "it",
        "its",
        "they",
        "them",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "about",
        "if",
        "but",
        "and",
        "or",
        "up",
    }

    input_words = set(w.strip(".,!?;:\"'()[]{}") for w in user_input.lower().split())
    input_words -= STOP_WORDS
    branch_text = current_branch["topic"] + " " + " ".join(current_branch["content"])
    branch_words = set(branch_text.lower().split())
    keyword_overlap = len(input_words & branch_words)
    if keyword_overlap >= 1 and sim > 0.48:
        return False, current_branch["topic"]

    if sim < 0.50:
        prompt = f"""
Generate a concise topic name for this new subject.

New Input: {user_input}

Output ONLY the topic name, under 5 words. No quotes, no punctuation.
"""
        new_topic = (
            _safe_chat([{"role": "user", "content": prompt}], model=MODEL)
            .strip()
            .strip('"')
            .strip("'")
            .rstrip(".")
        )
        if not new_topic:
            new_topic = user_input[:50]
        print(f"   [DEBUG-CONSOLIDATOR] -> Topic shift (sim={sim:.2f}): '{new_topic}'")
        return True, new_topic

    prompt = f"""
Determine if the user changed the subject.

Current Topic: {current_branch["topic"]}
Context: {" ".join(current_branch["content"][:3])}
New Input: {user_input}

Examples of SAME topic (continuations):
- "How do decorators work?" (when topic is Python)
- "What about classes?" (when topic is Python)

Examples of DIFFERENT topics:
- "What's the weather in Tokyo?"
- "How to bake a cake?"

Answer SAME or DIFFERENT. If DIFFERENT, also give a one-line topic name.
"""
    answer = _safe_chat([{"role": "user", "content": prompt}], model=MODEL)
    answer_upper = answer.upper()
    if "DIFFERENT" in answer_upper or answer_upper.startswith("YES"):
        new_topic = ""
        if "DIFFERENT" in answer_upper:
            parts = answer.split("DIFFERENT")
            if len(parts) > 1:
                new_topic = (
                    parts[1]
                    .strip()
                    .rstrip(":")
                    .strip()
                    .strip('"')
                    .strip("'")
                    .rstrip(".")
                )
        if not new_topic:
            topic_prompt = f"Generate a concise topic name for: {user_input}. Output ONLY the topic name, under 5 words."
            new_topic = (
                _safe_chat([{"role": "user", "content": topic_prompt}], model=MODEL)
                .strip()
                .strip('"')
                .strip("'")
                .rstrip(".")
            )
        if not new_topic:
            new_topic = user_input[:50]
        print(f"   [DEBUG-CONSOLIDATOR] -> Topic shift (sim={sim:.2f}): '{new_topic}'")
        return True, new_topic
    return False, current_branch["topic"]


def consolidate_to_tree(stm_lines: str, tree: LtmTree, current_branch_id: str | None):
    print(
        f"   [DEBUG-CONSOLIDATOR] -> Consolidating STM to tree (branch: {current_branch_id})"
    )

    prompt = f"""
Review these short-term memory lines and distill them into concise content statements for long-term storage.

STM Lines:
{stm_lines}

Output each important point as a separate line.
Skip trivial or already-known information.
Return ONLY the distilled lines, one per line.
Return nothing if nothing is worth keeping.
"""
    content = _safe_chat([{"role": "user", "content": prompt}], model=MODEL)

    if not content:
        print("   [DEBUG-CONSOLIDATOR] -> Nothing worth consolidating")
        return

    distilled = [l.strip() for l in content.splitlines() if l.strip()]
    print(f"   [DEBUG-CONSOLIDATOR] -> Distilled {len(distilled)} lines")

    distilled_text = " ".join(distilled)
    distilled_embedding = _safe_embed(distilled_text)

    best_match = tree.find_best_branch(
        distilled_embedding, exclude_id=current_branch_id
    )

    if best_match is not None:
        subtopic_prompt = f"""
Existing Topic: {best_match["topic"]}
New Content: {distilled_text}

Is the new content a distinct subtopic of the existing topic, or does it belong directly in the existing topic?
Answer with ONLY 'SUBTOPIC' or 'SAME'.
"""
        answer = _safe_chat(
            [{"role": "user", "content": subtopic_prompt}], model=MODEL
        ).upper()

        if "SUBTOPIC" in answer:
            topic_prompt = f"""
Given this existing topic and new content, generate a concise topic name for the new content.
Existing Topic: {best_match["topic"]}
New Content: {distilled_text}
Output ONLY the topic name, under 5 words.
"""
            new_topic = _safe_chat(
                [{"role": "user", "content": topic_prompt}], model=MODEL
            ).rstrip(".")

            current_depth = tree._depth_of(best_match["id"])
            if current_depth < MAX_DEPTH - 1:
                new_branch = tree.add_branch(
                    new_topic, distilled, parent_id=best_match["id"]
                )
                if new_branch:
                    print(
                        f"   [DEBUG-CONSOLIDATOR] -> Created subbranch '{new_topic}' under '{best_match['topic']}'"
                    )
                    tree.check_and_merge_siblings(new_branch["id"])
                else:
                    tree.append_to_branch(best_match["id"], distilled)
                    print(
                        f"   [DEBUG-CONSOLIDATOR] -> Max depth, appended to '{best_match['topic']}'"
                    )
            else:
                tree.append_to_branch(best_match["id"], distilled)
                print(
                    f"   [DEBUG-CONSOLIDATOR] -> Max depth, appended to '{best_match['topic']}'"
                )
        else:
            tree.append_to_branch(best_match["id"], distilled)
            print(
                f"   [DEBUG-CONSOLIDATOR] -> Appended to existing branch '{best_match['topic']}'"
            )
            tree.check_and_merge_siblings(best_match["id"])
    else:
        topic_prompt = f"""
Given this content, generate a concise topic name.
Content: {distilled_text}
Output ONLY the topic name, under 5 words.
"""
        new_topic = _safe_chat(
            [{"role": "user", "content": topic_prompt}], model=MODEL
        ).rstrip(".")

        new_branch = tree.add_branch(new_topic, distilled)
        if new_branch:
            print(f"   [DEBUG-CONSOLIDATOR] -> Created new root branch '{new_topic}'")
            tree.check_and_merge_siblings(new_branch["id"])
