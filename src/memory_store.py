from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Cheap heuristic token estimator.

    - 0 for empty / whitespace-only text
    - roughly ``len(text) / 4`` characters per token, which is a stable
      approximation for both English and Vietnamese.
    """
    if not text:
        return 0
    stripped = text.strip()
    if not stripped:
        return 0
    # Vietnamese diacritics push the average char/token up a bit, but 4 is
    # a fair middle ground. We add a small floor for very short messages.
    return max(1, len(stripped) // 4)


# ---------------------------------------------------------------------------
# User.md persistent storage
# ---------------------------------------------------------------------------

_SLUG_PATTERN = re.compile(r"[^a-z0-9_-]+")


def _slugify(user_id: str) -> str:
    if not user_id:
        return "default"
    cleaned = _SLUG_PATTERN.sub("-", user_id.strip().lower()).strip("-")
    return cleaned or "default"


@dataclass
class UserProfileStore:
    """Persistent storage for ``User.md`` (one file per ``user_id``)."""

    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)

    # --- path helpers -------------------------------------------------------

    def path_for(self, user_id: str) -> Path:
        return self.root_dir / f"{_slugify(user_id)}.md"

    # --- I/O ----------------------------------------------------------------

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return self._default_profile(user_id)
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return ""

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace the first occurrence of ``search_text`` inside User.md."""
        path = self.path_for(user_id)
        if not path.exists():
            return False
        text = path.read_text(encoding="utf-8")
        if search_text not in text:
            return False
        path.write_text(text.replace(search_text, replacement, 1), encoding="utf-8")
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        if not path.exists():
            return 0
        try:
            return path.stat().st_size
        except Exception:
            return 0

    # --- fact helpers (used by the advanced agent) -------------------------

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse the markdown file back into a ``{field: value}`` dict."""
        text = self.read_text(user_id)
        out: dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("- "):
                continue
            body = line[2:]
            if ":" not in body:
                continue
            key, _, value = body.partition(":")
            out[key.strip().lower()] = value.strip()
        return out

    def upsert_fact(self, user_id: str, field_name: str, value: str) -> bool:
        """Insert or replace one fact in User.md. Returns True if changed."""
        if not field_name or not value:
            return False
        path = self.path_for(user_id)
        if not path.exists():
            self.write_text(user_id, self._default_profile(user_id))

        text = path.read_text(encoding="utf-8")
        bullet = f"- {field_name.lower()}: {value.strip()}"

        pattern = re.compile(
            rf"^- {re.escape(field_name.lower())}:.*$",
            re.MULTILINE,
        )
        if pattern.search(text):
            new_text = pattern.sub(bullet, text, count=1)
            if new_text != text:
                path.write_text(new_text, encoding="utf-8")
                return True
            return False

        # Insert before the trailing "" line if present, else append.
        if text.endswith("\n\n"):
            new_text = text + bullet + "\n"
        elif text.endswith("\n"):
            new_text = text + bullet + "\n"
        else:
            new_text = text + "\n" + bullet + "\n"
        path.write_text(new_text, encoding="utf-8")
        return True

    # --- default skeleton --------------------------------------------------

    @staticmethod
    def _default_profile(user_id: str) -> str:
        return (
            f"# User profile: {user_id}\n\n"
            "_Persistent facts learned across sessions. Updated by the agent._\n\n"
        )


# ---------------------------------------------------------------------------
# Profile fact extraction (Vietnamese)
# ---------------------------------------------------------------------------

# Heuristics: each entry is (field, regex, value-capture group)
# Only declarative patterns are accepted; question marks and second-person
# pronouns are filtered out separately.
_PROFILE_RULES: list[tuple[str, re.Pattern[str]]] = [
    # "tên mình là X", "tên tôi là X", "tên là X" (cả dạng rút gọn).
    # Hỗ trợ tên ghép 1-3 từ (vd "DũngCT", "DũngCT Stress").
    ("name", re.compile(
        r"tên\s+(?:mình\s+|tôi\s+|em\s+)?là\s+"
        r"((?:[A-ZÀ-Ỹ][A-Za-zÀ-ỹ0-9_]*\s*){1,3})",
        re.IGNORECASE,
    )),
    ("location", re.compile(
        r"(?:mình\s+(?:đang\s+)?ở|mình\s+sống\s+ở|hiện\s+ở)\s+([^,.;\n?]+?)(?=\s+(?:và\s+(?:đang\s+)?(?:làm|là|ở)|,\s*(?:mình|tôi))|$)",
        re.IGNORECASE,
    )),
    ("profession", re.compile(
        r"(?:mình\s+(?:đang\s+)?(?:làm\s+)?(?:là|working\s+as)\s+(?:một\s+)?|nghề\s+(?:là|của\s+mình\s+là)\s+|làm\s+(?:một\s+)?)(MLOps|backend\s+engineer|frontend\s+engineer|fullstack\s+engineer|[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ\-]*(?:\s+[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ\-]*)?\s*(?:engineer|developer|designer|manager|analyst|scientist|writer|founder|lead|architect|operator))",
        re.IGNORECASE,
    )),
    ("response_style", re.compile(
        r"(?:mình\s+muốn\s+bạn\s+trả\s+lời|mình\s+thích\s+(?:kiểu\s+)?(?:trả\s+lời|câu\s+trả\s+lời)|style\s+trả\s+lời\s+(?:mình\s+)?(?:là|thích)|trả\s+lời\s+(?:thành\s+)?(?:bullet|ngắn\s+gọn|gọn))\s*[:]?\s*([^,.;\n?]+?)(?=\s+(?:khi|có|và|,|\.|$))",
        re.IGNORECASE,
    )),
    ("favorite_drink", re.compile(
        r"(?:đồ\s+uống\s+yêu\s+thích\s+là|mình\s+thích\s+uống|mình\s+uống|mình\s+vẫn\s+uống)\s+([^,.;\n?]+)",
        re.IGNORECASE,
    )),
    ("favorite_food", re.compile(
        r"(?:món\s+ăn\s+yêu\s+thích\s+(?:là|của\s+mình)|mình\s+thích\s+ăn|thích\s+ăn)\s+([^,.;\n?]+?)(?=\s+(?:là|và|,|\.|$))",
        re.IGNORECASE,
    )),
    ("pet", re.compile(
        r"(?:mình\s+nuôi\s+(?:một\s+)?(?:bé\s+)?|nuôi\s+(?:một\s+)?(?:bé\s+)?)(?:một\s+bé\s+|bé\s+|con\s+)?([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ\s]+?)(?=\s+(?:tên\b|,|\.|$))",
        re.IGNORECASE,
    )),
    ("interests", re.compile(
        r"(?:mình\s+thích|mình\s+quan\s+tâm\s+đến|mình\s+đang\s+(?:học\s+)?(?:thêm\s+)?về)\s+([^,.;\n?]+)",
        re.IGNORECASE,
    )),
]

# Phrases that often signal a question instead of a fact. Conservative: only
# filter when the *whole* turn looks like a question rather than a statement.
_QUESTION_HINTS = re.compile(
    r"(^\s*(bạn|mình|có\s+thể|có\s+ai|thử|làm\s+sao|như\s+thế\s+nào|tại\s+sao)\b.*\?)",
    re.IGNORECASE,
)
# Wh-style questions like "Mình tên gì?", "Nghề gì?", "Ở đâu?" — usually
# a recall prompt rather than a fresh fact declaration.
_RECALL_QUESTION = re.compile(
    r"\b(mình\s+(?:tên|đang\s+(?:làm|làm\s+nghề\s+gì|ở\s+đâu|là\s+nghề\s+gì|nghề\s+gì)|đang\s+làm\s+nghề\s+gì|làm\s+nghề\s+gì|nghề\s+gì|tên\s+gì|ở\s+đâu|nuôi\s+con\s+gì|có\s+thể\s+làm\s+gì|thích\s+gì|là\s+gì)|tên\s+mình\s+là\s+gì|tên\s+gì|nghề\s+gì|ở\s+đâu)\b.*\??$",
    re.IGNORECASE,
)
# Tên mình là gì / Tên tôi là gì — these are obviously questions
_QUESTION_PHRASE = re.compile(
    r"\b(tên\s+(?:mình|tôi|em)\s+là\s+gì|nghề\s+(?:mình|tôi)\s+là\s+gì|mình\s+đang\s+làm\s+nghề\s+gì)\b",
    re.IGNORECASE,
)
# Words that often appear in queries rather than fact declarations.
_QUESTION_WORDS = re.compile(
    r"\b(nhắc\s+lại|thử\s+ghi\s+nhớ|bạn\s+thử|bạn\s+nhớ|bạn\s+có\s+biết|nhớ\s+giúp|tóm\s+tắt\s+ngắn|hãy\s+nhắc|hãy\s+mô\s+tả)\b",
    re.IGNORECASE,
)

# Words that signal "I was joking" / noise — should NOT overwrite a fact.
_NOISE_HINTS = re.compile(
    r"\b(đùa|joke|không\s+phải|chỉ\s+là|chứ\s+không\s+phải)\b",
    re.IGNORECASE,
)

# Max length of an extracted value — protects against runaway captures.
_MAX_VALUE_LEN = 80

# Question words that should never become a stored fact value.
_BAD_VALUES = {
    "gì", "gì?", "đâu", "đâu?", "nào", "nào?", "ai", "ai?", "sao", "sao?",
    "thế nào", "thế nào?", "là gì", "là gì?",
}


def _clean_value(raw: str | None) -> str:
    if not raw:
        return ""
    raw = raw.strip()
    # drop trailing particles
    for tail in (" nhé", " nha", " đó", " đấy", " nữa", " nhe", " nhaa"):
        if raw.endswith(tail):
            raw = raw[: -len(tail)]
    # collapse spaces
    raw = re.sub(r"\s+", " ", raw).strip()
    if len(raw) > _MAX_VALUE_LEN:
        raw = raw[:_MAX_VALUE_LEN].rstrip()
    # Drop values that are themselves question words ("gì", "đâu", "nào"...)
    if raw.lower() in _BAD_VALUES:
        return ""
    return raw


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract stable profile facts from one user message.

    Returns only the facts that look confidently present. The function
    intentionally avoids writing to ``User.md`` when a turn is mostly a
    question or contains self-correcting noise.
    """
    if not message:
        return {}

    # Skip pure question turns.
    if _QUESTION_HINTS.match(message):
        return {}
    # The whole message is one big question to the agent.
    if message.strip().endswith("?") and _QUESTION_WORDS.search(message):
        return {}
    # Recall-style questions like "Mình tên gì?", "Nghề gì?"
    if message.strip().endswith("?") and _RECALL_QUESTION.search(message):
        return {}
    # "Tên mình là gì" / "Nghề mình là gì" — even without trailing ?
    if _QUESTION_PHRASE.search(message):
        return {}

    # If the message is dominated by noise cues, skip to avoid overwriting.
    if _NOISE_HINTS.search(message) and len(message) < 120:
        return {}

    facts: dict[str, str] = {}
    for field, pattern in _PROFILE_RULES:
        m = pattern.search(message)
        if not m:
            continue
        value = _clean_value(m.group(1))
        if not value or len(value) < 2:
            continue
        facts[field] = value

    return facts


# ---------------------------------------------------------------------------
# Compact summary helpers
# ---------------------------------------------------------------------------

def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Heuristic compact summary of older messages.

    For an offline / deterministic lab we just stitch a short digest of the
    most informative lines. In a real deployment the LLM would do this.
    """
    if not messages:
        return ""

    head: list[str] = []
    for m in messages[-max_items:]:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        snippet = content if len(content) <= 120 else content[:117] + "..."
        head.append(f"- {role}: {snippet}")

    if not head:
        return ""

    return "Tóm tắt hội thoại trước:\n" + "\n".join(head)


# ---------------------------------------------------------------------------
# Compact memory manager
# ---------------------------------------------------------------------------

@dataclass
class CompactMemoryManager:
    """Compact memory for long threads.

    - Keeps recent messages in full.
    - When cumulative tokens exceed ``threshold_tokens``, summarises the
      older tail into a ``summary`` string and resets the message list to
      only the last ``keep_messages`` items.
    - Tracks the number of compactions per thread for benchmarking.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _ensure(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
                "total_tokens": 0,
            }
        return self.state[thread_id]

    def append(self, thread_id: str, role: str, content: str) -> None:
        st = self._ensure(thread_id)
        messages: list[dict[str, str]] = st["messages"]  # type: ignore[assignment]
        summary: str = st.get("summary", "")  # type: ignore[assignment]
        compactions: int = st.get("compactions", 0)  # type: ignore[assignment]
        total_tokens: int = st.get("total_tokens", 0)  # type: ignore[assignment]

        msg = {"role": role, "content": content}
        messages.append(msg)
        msg_tokens = estimate_tokens(content)
        total_tokens += msg_tokens

        if total_tokens >= self.threshold_tokens and len(messages) > self.keep_messages:
            old_tail = messages[: -self.keep_messages]
            new_summary = summarize_messages(old_tail)
            if summary:
                combined = summary + "\n" + new_summary
            else:
                combined = new_summary
            messages = messages[-self.keep_messages :]
            compactions += 1
            # Recompute total_tokens to reflect new shape: summary + recent
            total_tokens = estimate_tokens(combined) + sum(
                estimate_tokens(m["content"]) for m in messages
            )
            st["summary"] = combined

        st["messages"] = messages
        st["compactions"] = compactions
        st["total_tokens"] = total_tokens

    def context(self, thread_id: str) -> dict[str, object]:
        return self._ensure(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        return int(self._ensure(thread_id).get("compactions", 0))

    def prompt_tokens(self, thread_id: str) -> int:
        """Tokens carried into one turn: summary + recent messages."""
        st = self._ensure(thread_id)
        summary = str(st.get("summary", ""))
        messages: list[dict[str, str]] = st.get("messages", [])  # type: ignore[assignment]
        return estimate_tokens(summary) + sum(
            estimate_tokens(m.get("content", "")) for m in messages
        )
