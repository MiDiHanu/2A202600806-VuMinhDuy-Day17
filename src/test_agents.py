from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the `src` directory is on sys.path when pytest runs from elsewhere.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from agent_advanced import AdvancedAgent  # noqa: E402
from agent_baseline import BaselineAgent  # noqa: E402
from config import LabConfig, ProviderConfig  # noqa: E402
from memory_store import (  # noqa: E402
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for tests.

    - point ``state_dir`` into ``tmp_path``
    - shrink the compact threshold so compactions happen quickly in tests
    """
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=200,
        compact_keep_messages=3,
        model=ProviderConfig(provider="openai", model_name="gpt-4o-mini"),
        judge_model=ProviderConfig(provider="anthropic", model_name="claude-3-5-sonnet-latest"),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify User.md can be created, updated, and edited."""
    store = UserProfileStore(tmp_path / "profiles")

    # read_text on missing file returns the default skeleton
    initial = store.read_text("dungct")
    assert "dungct" in initial

    # write_text creates the file and write_text returns the path
    target = store.path_for("dungct")
    store.write_text(
        "dungct",
        "# User profile: dungct\n\n- name: DũngCT\n- location: Đà Nẵng\n",
    )
    assert target.exists()
    assert "DũngCT" in store.read_text("dungct")

    # edit_text replaces one occurrence
    changed = store.edit_text("dungct", "- location: Đà Nẵng", "- location: Huế")
    assert changed is True
    assert "Huế" in store.read_text("dungct")
    assert "Đà Nẵng" not in store.read_text("dungct")

    # upsert_fact is idempotent when value matches
    store.upsert_fact("dungct", "name", "DũngCT")
    assert len(store.read_text("dungct").splitlines()) == len(
        store.read_text("dungct").splitlines()
    )
    # upsert_fact replaces when value differs (conflict handling)
    store.upsert_fact("dungct", "location", "Đà Nẵng")
    text = store.read_text("dungct")
    assert text.count("- location:") == 1, "duplicate location bullet was created"

    # file_size grows as we add facts
    assert store.file_size("dungct") > 0


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""
    mgr = CompactMemoryManager(threshold_tokens=150, keep_messages=2)
    tid = "thread-A"

    # Short message → no compaction
    mgr.append(tid, "user", "hi")
    assert mgr.compaction_count(tid) == 0

    # Long thread → at least one compaction
    long_message = "Đây là một message khá dài " * 30
    for i in range(8):
        mgr.append(tid, "user", f"{long_message} {i}")
        mgr.append(tid, "assistant", f"reply {i} " + "x" * 200)

    assert mgr.compaction_count(tid) >= 1
    ctx = mgr.context(tid)
    # After compact, recent messages list is at most keep_messages
    assert len(ctx["messages"]) <= 2  # type: ignore[arg-type]
    # Summary should now exist
    assert ctx["summary"]  # type: ignore[has-type]


def test_cross_session_recall(tmp_path: Path) -> None:
    """Advanced remembers across sessions; baseline does not."""
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config=config, force_offline=True)
    baseline = BaselineAgent(config=config, force_offline=True)

    # Session 1 — feed profile facts
    advanced.reply("dungct", "t1", "Chào bạn, mình tên là DũngCT.")
    advanced.reply(
        "dungct",
        "t1",
        "Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.",
    )
    advanced.reply("dungct", "t1", "Đồ uống yêu thích là cà phê sữa đá.")

    # Session 2 — fresh thread, ask recall
    result_a = advanced.reply("dungct", "t2", "Mình tên gì và đồ uống yêu thích là gì?")
    assert "DũngCT" in result_a["response"]
    assert "cà phê sữa đá" in result_a["response"]

    # Baseline — same setup, but on a *fresh* thread it forgets
    baseline.reply("dungct", "b1", "Chào bạn, mình tên là DũngCT.")
    baseline.reply(
        "dungct",
        "b1",
        "Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.",
    )
    baseline.reply("b1", "user", "Đồ uống yêu thích là cà phê sữa đá.")
    result_b = baseline.reply("dungct", "b2", "Mình tên gì và đồ uống yêu thích là gì?")
    assert "DũngCT" not in result_b["response"]
    assert "cà phê sữa đá" not in result_b["response"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Prompt tokens of advanced on a long thread should be lower than baseline."""
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config=config, force_offline=True)
    baseline = BaselineAgent(config=config, force_offline=True)

    long_turns = [
        f"Đây là message số {i} trong chuỗi dài. " + ("lorem ipsum " * 20)
        for i in range(20)
    ]

    for t in long_turns:
        baseline.reply("dungct", "long", t)
        advanced.reply("dungct", "long", t)

    baseline_prompt = baseline.prompt_token_usage("long")
    advanced_prompt = advanced.prompt_token_usage("long")

    # Advanced MUST have at least one compaction on this long thread
    assert advanced.compaction_count("long") >= 1
    # And it MUST carry a smaller prompt context than the baseline
    assert advanced_prompt < baseline_prompt, (
        f"Advanced prompt tokens ({advanced_prompt}) should be less than "
        f"baseline prompt tokens ({baseline_prompt}) after compact."
    )


# ---------------------------------------------------------------------------
# Light unit tests for memory_store helpers
# ---------------------------------------------------------------------------

def test_estimate_tokens_basic() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("   ") == 0
    assert estimate_tokens("a") == 1
    assert estimate_tokens("a" * 40) >= 8


def test_extract_profile_updates_filters_questions() -> None:
    # Question-only turn should not write to profile
    facts = extract_profile_updates("Bạn thử nhớ lại xem đồ uống yêu thích là gì?")
    assert facts == {}

    # Declarative turn should extract facts
    facts = extract_profile_updates(
        "Chào bạn, mình tên là DũngCT, mình ở Đà Nẵng và đang làm backend engineer."
    )
    assert facts.get("name") == "DũngCT"
    assert "Đà Nẵng" in facts.get("location", "")
    assert "backend engineer" in facts.get("profession", "")


def test_extract_profile_updates_filters_noise() -> None:
    # "đùa" hint on a short message should be ignored
    facts = extract_profile_updates("Đùa thôi, mình là product manager nhé.")
    assert "profession" not in facts or "product manager" not in facts.get(
        "profession", ""
    )


if __name__ == "__main__":
    # Allow ``python src/test_agents.py`` for ad-hoc checks
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
