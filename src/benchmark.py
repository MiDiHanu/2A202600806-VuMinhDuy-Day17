from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import estimate_tokens


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read a JSON file containing a list of conversations."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0 / 0.5 / 1 depending on how many expected facts are present.

    - 1.0 if all expected substrings appear
    - 0.5 if at least one appears
    - 0.0 otherwise
    """
    if not expected:
        return 1.0
    if not answer:
        return 0.0
    a = answer.lower()
    hits = sum(1 for e in expected if e.lower() in a)
    if hits == len(expected):
        return 1.0
    if hits > 0:
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality score for offline mode.

    Combines:
    - recall coverage (60%)
    - length penalty for overly long answers (40%)

    Returns a value in ``[0, 1]``.
    """
    if not answer:
        return 0.0
    if not expected:
        # No expected = pure length / coherence heuristic
        return 0.5

    a = answer.lower()
    hits = sum(1 for e in expected if e.lower() in a)
    coverage = hits / len(expected)

    # Penalise very long answers (likely noisy / off-topic)
    tokens = estimate_tokens(answer)
    if tokens <= 80:
        length_score = 1.0
    elif tokens <= 200:
        length_score = 0.7
    else:
        length_score = 0.4

    return round(0.6 * coverage + 0.4 * length_score, 3)


# ---------------------------------------------------------------------------
# Per-agent benchmark
# ---------------------------------------------------------------------------

def run_agent_benchmark(
    agent_name: str,
    agent: Any,
    conversations: list[dict[str, Any]],
    config: LabConfig,
) -> BenchmarkRow:
    """Evaluate one agent over many conversations.

    1. Feed every turn of every conversation to the agent.
    2. Track ``agent tokens only`` and ``prompt tokens processed``.
    3. Ask recall questions in a *fresh* thread to measure cross-session recall.
    4. Average recall and quality across questions.
    5. Record memory file growth and compaction count.
    """
    user_id_seen: set[str] = set()
    total_agent_tokens = 0
    total_prompt_tokens = 0
    total_compactions = 0
    memory_growth_bytes = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    baseline_size: dict[str, int] = {}
    if isinstance(agent, AdvancedAgent):
        for conv in conversations:
            uid = conv.get("user_id", "default")
            user_id_seen.add(uid)
            baseline_size[uid] = agent.memory_file_size(uid)

    for conv in conversations:
        cid = conv.get("id", "conv")
        user_id = conv.get("user_id", "default")
        turns = conv.get("turns", [])
        # Use the conversation id as the thread id so the same conversation
        # is one continuous memory window.
        thread_id = f"thread-{cid}"

        for turn in turns:
            result = agent.reply(user_id, thread_id, turn)
            total_agent_tokens += int(result.get("agent_tokens", 0))
            total_prompt_tokens += int(result.get("prompt_tokens_processed", 0))

        # --- recall evaluation in a *fresh* thread -------------------------
        for q in conv.get("recall_questions", []):
            question = q.get("question", "")
            expected = q.get("expected_contains", [])
            fresh_thread = f"recall-{cid}-{len(recall_scores)}"
            result = agent.reply(user_id, fresh_thread, question)
            answer = result.get("response", "")
            total_agent_tokens += int(result.get("agent_tokens", 0))
            total_prompt_tokens += int(result.get("prompt_tokens_processed", 0))
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    if isinstance(agent, AdvancedAgent):
        for uid in user_id_seen:
            memory_growth_bytes += agent.memory_file_size(uid) - baseline_size.get(uid, 0)
        for conv in conversations:
            cid = conv.get("id", "conv")
            total_compactions += agent.compaction_count(f"thread-{cid}")
            for i in range(len(conv.get("recall_questions", []))):
                total_compactions += agent.compaction_count(
                    f"recall-{cid}-{i}"
                )
    elif isinstance(agent, BaselineAgent):
        for conv in conversations:
            cid = conv.get("id", "conv")
            for tid in (f"thread-{cid}",) + tuple(
                f"recall-{cid}-{i}"
                for i in range(len(conv.get("recall_questions", [])))
            ):
                total_prompt_tokens += 0  # already accumulated
        # Baseline has no memory file
        memory_growth_bytes = 0

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(avg_recall, 3),
        response_quality=round(avg_quality, 3),
        memory_growth_bytes=memory_growth_bytes,
        compactions=total_compactions,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_rows(rows: list[BenchmarkRow], title: str) -> str:
    """Pretty-print a benchmark comparison table."""
    if not rows:
        return f"### {title}\n_(no data)_\n"

    header = (
        "| Agent | Agent tokens only | Prompt tokens processed | "
        "Cross-session recall | Response quality | "
        "Memory growth (bytes) | Compactions |\n"
        "|---|---|---|---|---|---|---|"
    )
    lines: list[str] = [f"### {title}", "", header]
    for row in rows:
        lines.append(
            f"| {row.agent_name} | {row.agent_tokens_only} | "
            f"{row.prompt_tokens_processed} | {row.recall_score:.2f} | "
            f"{row.response_quality:.2f} | {row.memory_growth_bytes} | "
            f"{row.compactions} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _run_suite(
    label: str,
    path: Path,
    config: LabConfig,
    state_dir: Path,
) -> list[BenchmarkRow]:
    """Run one benchmark suite (standard or long-context stress)."""
    conversations = load_conversations(path)
    if not conversations:
        return []

    # Use *fresh* state directories per suite so growth numbers are clean.
    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    rows = [
        run_agent_benchmark("Baseline", baseline, conversations, config),
        run_agent_benchmark("Advanced", advanced, conversations, config),
    ]
    print(format_rows(rows, label))
    return rows


def main() -> None:
    """Run both benchmark suites and print the comparison."""
    config = load_config(Path(__file__).resolve().parent.parent)
    state_dir = config.state_dir / "benchmark"
    state_dir.mkdir(parents=True, exist_ok=True)

    print("# Memory Systems Benchmark — Day 17\n")
    print(
        "Mục tiêu: so sánh **Baseline** (chỉ short-term) với **Advanced** "
        "(short-term + persistent + compact) trên cùng dữ liệu tiếng Việt.\n"
    )

    standard_path = config.data_dir / "conversations.json"
    long_path = config.data_dir / "advanced_long_context.json"

    standard_rows = _run_suite(
        "Standard benchmark (conversations.json)",
        standard_path,
        config,
        state_dir,
    )
    long_rows = _run_suite(
        "Long-context stress benchmark (advanced_long_context.json)",
        long_path,
        config,
        state_dir,
    )

    # Quick comparison note
    if standard_rows and long_rows:
        print("## Phân tích nhanh\n")
        b_std = next(r for r in standard_rows if r.agent_name == "Baseline")
        a_std = next(r for r in standard_rows if r.agent_name == "Advanced")
        b_long = next(r for r in long_rows if r.agent_name == "Baseline")
        a_long = next(r for r in long_rows if r.agent_name == "Advanced")

        print(
            f"- **Cross-session recall** (standard): "
            f"baseline={b_std.recall_score:.2f} → advanced={a_std.recall_score:.2f}"
        )
        print(
            f"- **Cross-session recall** (long-context): "
            f"baseline={b_long.recall_score:.2f} → advanced={a_long.recall_score:.2f}"
        )
        print(
            f"- **Prompt tokens processed** (standard): "
            f"baseline={b_std.prompt_tokens_processed} → advanced={a_std.prompt_tokens_processed}"
        )
        print(
            f"- **Prompt tokens processed** (long-context): "
            f"baseline={b_long.prompt_tokens_processed} → advanced={a_long.prompt_tokens_processed}"
        )
        print(
            f"- **Compactions** (long-context): advanced={a_long.compactions} "
            f"(compact memory đã chủ động nén thread dài)"
        )


if __name__ == "__main__":
    main()
