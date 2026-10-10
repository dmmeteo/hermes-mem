from hermes_mem.select import HostMessage, ProcessingPolicy, build_episodes


def m(i, role, content, tool_calls=False, summary=False):
    return HostMessage(id=i, ref=f"{role}@{i}", role=role, content=content, timestamp=float(i),
                       has_tool_calls=tool_calls, compacted=False, summary=summary)


def test_episode_keeps_request_and_final_answer_but_not_tool_traffic():
    episodes = build_episodes("s", [
        m(1, "user", "Чому ми взяли Dragonfly замість Redis?"),
        m(2, "assistant", "Let me check the notes.", tool_calls=True),
        m(3, "assistant", "Бо він багатопотоковий і економніший по пам'яті."),
        m(4, "user", "[CONTEXT COMPACTION — REFERENCE ONLY] summary", summary=True),
    ])
    assert [(e.user_text, e.assistant_text) for e in episodes] == [
        ("Чому ми взяли Dragonfly замість Redis?", "Бо він багатопотоковий і економніший по пам'яті.")
    ]
    assert episodes[0].key == "s/user@1"
    assert episodes[0].assistant_refs == ["assistant@3"]


def test_bare_acknowledgement_is_dropped_but_short_correction_is_kept():
    episodes = build_episodes("s", [
        m(1, "user", "дякую!"), m(2, "assistant", "Будь ласка."),
        m(3, "user", "Ні, тільки Postgres."), m(4, "assistant", "Ок."),
    ])
    assert [e.user_text for e in episodes] == ["Ні, тільки Postgres."]


def test_pasted_log_is_not_remembered_but_the_answer_is():
    log = "\n".join(f"2026-10-10 09:0{i}:00 ERROR worker {i} crashed" for i in range(6))
    episodes = build_episodes("s", [m(1, "user", log), m(2, "assistant", "The worker OOMs; raise the memory limit.")])
    assert [(e.user_text, e.assistant_text) for e in episodes] == [("", "The worker OOMs; raise the memory limit.")]


def test_long_answer_is_chunked_with_the_question_in_every_chunk():
    answer = "\n\n".join(f"Paragraph {i}: " + "detail " * 60 for i in range(8))
    (episode,) = ProcessingPolicy().apply(build_episodes("s", [m(1, "user", "Explain the rollout plan"), m(2, "assistant", answer)]))
    assert len(episode.chunks) > 2
    assert all(chunk.startswith("User: Explain the rollout plan") for chunk in episode.chunks)
    assert all(len(chunk) <= 1400 for chunk in episode.chunks)
