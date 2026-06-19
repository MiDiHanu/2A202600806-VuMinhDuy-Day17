from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Advanced agent with three memory layers.

    1. Short-term thread memory (``CompactMemoryManager``)
    2. Persistent ``User.md`` (one file per ``user_id``)
    3. Compact memory: summary + last ``keep_messages`` items

    Bonus features:
    - Conflict handling: when a new fact overrides an old one, the new
      value replaces the old bullet (no duplicates).
    - Confidence threshold: question-only or noise-laden turns are
      filtered out before being written to ``User.md``.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        # Per-thread cumulative counters
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        # Tracks the most recent "stated" value per fact so we can detect
        # corrections and avoid storing obvious contradictions.
        self._last_stated: dict[tuple[str, str], str] = {}

        # Optional live agent; offline is the benchmark default.
        self.langchain_agent: Any = None
        if not force_offline:
            self.langchain_agent = self._maybe_build_langchain_agent()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None and not self.force_offline:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return int(self.thread_tokens.get(thread_id, 0))

    def prompt_token_usage(self, thread_id: str) -> int:
        """Sum of every prompt the agent has carried into this thread."""
        return int(self.thread_prompt_tokens.get(thread_id, 0))

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    # ------------------------------------------------------------------
    # Offline behaviour
    # ------------------------------------------------------------------

    def _reply_offline(
        self, user_id: str, thread_id: str, message: str
    ) -> dict[str, Any]:
        """Deterministic advanced path with the three memory layers.

        Flow:
        1. Extract stable profile facts from the incoming message.
        2. Persist them into ``User.md`` (conflict-aware).
        3. Append into compact memory.
        4. Build the prompt context = User.md + summary + recent messages.
        5. Generate a deterministic answer using the persisted memory.
        6. Update token counters.
        """
        # 1) extract + 2) persist
        facts = extract_profile_updates(message)
        for field, value in facts.items():
            self._store_fact(user_id, field, value)

        # 3) compact memory
        self.compact_memory.append(thread_id, "user", message)

        # 4) estimate prompt context for this turn
        prompt_tokens_now = self._estimate_prompt_context_tokens(user_id, thread_id)

        # 5) deterministic response using persistent + recent memory
        response = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", response)

        # 6) token accounting
        agent_tokens = estimate_tokens(response)
        self.thread_tokens[thread_id] = (
            self.thread_tokens.get(thread_id, 0) + agent_tokens
        )
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens_now
        )

        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens_processed": prompt_tokens_now,
        }

    def _store_fact(self, user_id: str, field: str, value: str) -> None:
        """Conflict-aware write to User.md.

        - If the new value equals the last stated value, skip (idempotent).
        - If it overrides the old value, replace.
        - Noise hints ("đùa", "chỉ là") are dropped in ``extract_profile_updates``
          so they never reach this path.
        """
        key = (user_id, field)
        previous = self.profile_store.facts(user_id).get(field, "")
        if previous and previous.lower() == value.lower():
            self._last_stated[key] = value
            return  # no change
        # Conflict handling: this is an *update*, not a duplicate.
        changed = self.profile_store.upsert_fact(user_id, field, value)
        if changed:
            self._last_stated[key] = value

    def _estimate_prompt_context_tokens(
        self, user_id: str, thread_id: str
    ) -> int:
        """Tokens carried into one turn.

        Includes:
        - ``User.md`` (persistent profile)
        - compact summary
        - recent kept messages
        """
        profile_text = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary = str(ctx.get("summary", ""))
        messages: list[dict[str, str]] = ctx.get("messages", [])  # type: ignore[assignment]

        tokens = estimate_tokens(profile_text) + estimate_tokens(summary)
        tokens += sum(estimate_tokens(m.get("content", "")) for m in messages)
        return tokens

    def _offline_response(
        self, user_id: str, thread_id: str, message: str
    ) -> str:
        """Deterministic answer that uses the persisted profile.

        Recall questions (containing the right keywords) are answered
        straight from ``User.md``. Compound questions collect all matching
        facts into a single response.
        """
        facts = self.profile_store.facts(user_id)
        msg = (message or "").strip().lower()
        if not msg:
            return "Bạn muốn mình hỗ trợ gì tiếp?"

        is_question = msg.endswith("?") or any(
            q in msg
            for q in ("nhắc lại", "thử ghi nhớ", "bạn nhớ", "bạn thử", "bạn có biết")
        )

        if is_question:
            parts: list[str] = []
            if any(k in msg for k in ("tên", "tên mình", "tên tôi", "mô tả")):
                name = facts.get("name")
                if name:
                    parts.append(f"tên là {name}")
            if any(k in msg for k in ("nghề", "làm gì", "làm nghề", "làm việc")):
                prof = facts.get("profession")
                if prof:
                    parts.append(f"đang làm {prof}")
            if any(k in msg for k in ("ở đâu", "nơi ở", "ở hiện")):
                loc = facts.get("location")
                if loc:
                    parts.append(f"đang ở {loc}")
            if "đồ uống" in msg or "uống gì" in msg:
                drink = facts.get("favorite_drink")
                if drink:
                    parts.append(f"đồ uống yêu thích là {drink}")
            if "món ăn" in msg or "ăn gì" in msg:
                food = facts.get("favorite_food")
                if food:
                    parts.append(f"món ăn yêu thích là {food}")
            if "thú cưng" in msg or "nuôi" in msg:
                pet = facts.get("pet")
                if pet:
                    parts.append(f"nuôi {pet}")
            if "style" in msg or ("trả lời" in msg and "thích" in msg):
                style = facts.get("response_style")
                if style:
                    parts.append(f"style trả lời thích: {style}")
            if any(k in msg for k in ("sở thích", "thích gì", "quan tâm")):
                intr = facts.get("interests")
                if intr:
                    parts.append(f"thích {intr}")

            if parts:
                if len(parts) == 1:
                    return f"Mình nhớ: {parts[0]}."
                return "Mình nhớ: " + ", ".join(parts) + "."

            if not facts:
                return (
                    "Mình chưa ghi nhận nhiều thông tin dài hạn của bạn. "
                    "Bạn có thể chia sẻ thêm để mình nhớ cho các lần sau."
                )
            keys = ", ".join(sorted(facts.keys()))
            return f"Hiện tại mình đã lưu các thông tin: {keys}."

        return "Đã ghi nhận thông tin vào bộ nhớ dài hạn."

    # ------------------------------------------------------------------
    # Optional live path
    # ------------------------------------------------------------------

    def _maybe_build_langchain_agent(self):
        """Optionally wire a live LangChain agent. Returns ``None`` if unavailable."""
        model = build_chat_model(self.config.model)
        if model is None:
            return None
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Bạn là một agent có persistent memory. "
                        "Dưới đây là profile người dùng đã lưu và tóm tắt "
                        "hội thoại trước. Hãy dùng chúng khi trả lời và ghi "
                        "nhận thêm fact mới một cách thận trọng.",
                    ),
                    ("system", "Profile người dùng (User.md):\n{profile}"),
                    ("system", "Tóm tắt thread trước:\n{summary}"),
                    ("placeholder", "{history}"),
                    ("human", "{input}"),
                ]
            )
            return prompt | model | StrOutputParser()
        except Exception:
            return None

    def _reply_live(
        self, user_id: str, thread_id: str, message: str
    ) -> dict[str, Any]:
        # Persist facts
        facts = extract_profile_updates(message)
        for field, value in facts.items():
            self._store_fact(user_id, field, value)

        # Compact memory
        self.compact_memory.append(thread_id, "user", message)

        profile_text = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary = str(ctx.get("summary", ""))
        recent: list[dict[str, str]] = list(ctx.get("messages", []))  # type: ignore[arg-type]
        # Drop the most recent user message from history (it goes in {input})
        history = recent[:-1] if recent and recent[-1].get("role") == "user" else recent

        try:
            response = self.langchain_agent.invoke(
                {
                    "profile": profile_text,
                    "summary": summary,
                    "history": history,
                    "input": message,
                }
            )
        except Exception:
            return self._reply_offline(user_id, thread_id, message)

        self.compact_memory.append(thread_id, "assistant", response)
        prompt_tokens_now = self._estimate_prompt_context_tokens(user_id, thread_id)
        agent_tokens = estimate_tokens(response)
        self.thread_tokens[thread_id] = (
            self.thread_tokens.get(thread_id, 0) + agent_tokens
        )
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens_now
        )
        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens_processed": prompt_tokens_now,
        }
