from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Baseline agent: short-term memory only.

    - No persistent ``User.md``
    - No compact memory
    - Forgets long-term facts across new threads
    - Carries the *entire* in-thread history into every prompt — this is
      the exact cost pattern the advanced agent is designed to avoid.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        # Optionally try to build a live agent; offline mode stays the
        # default and is what the benchmark relies on.
        self.langchain_agent: Any = None
        if not force_offline:
            self.langchain_agent = self._maybe_build_langchain_agent()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return agent response and accounting for one turn.

        Note: ``user_id`` is intentionally ignored — the baseline must not
        gain long-term memory just because the caller passes a stable id.
        """
        if self.langchain_agent is not None and not self.force_offline:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        st = self.sessions.get(thread_id)
        return int(st.token_usage) if st else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        """Estimate cumulative context tokens carried into this thread.

        The baseline keeps the *full* history — every turn processes all
        previous messages. That is exactly what makes it expensive on long
        threads and what compact memory is meant to fix.
        """
        st = self.sessions.get(thread_id)
        return int(st.prompt_tokens_processed) if st else 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def memory_file_size(self, user_id: str) -> int:
        # Baseline never writes to disk.
        return 0

    # ------------------------------------------------------------------
    # Offline behaviour
    # ------------------------------------------------------------------

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic offline behaviour for the baseline.

        Strategy:
        - Append the user message to the thread-local session.
        - Generate a short acknowledgement that reflects what was said
          *in this thread only* (no cross-thread / persistent memory).
        - Update token counters so the benchmark can compare them.
        """
        state = self.sessions.setdefault(thread_id, SessionState())
        state.messages.append({"role": "user", "content": message})

        # Build a deterministic reply — never invent a name or a fact.
        response = self._build_offline_reply(message, len(state.messages))
        state.messages.append({"role": "assistant", "content": response})

        # Token accounting
        agent_tokens = estimate_tokens(response)
        state.token_usage += agent_tokens

        # Prompt tokens processed = every previous message + the new one.
        # The baseline carries the full history each turn.
        prompt_tokens_now = sum(
            estimate_tokens(m["content"]) for m in state.messages
        )
        state.prompt_tokens_processed += prompt_tokens_now

        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens_processed": prompt_tokens_now,
        }

    def _build_offline_reply(self, message: str, turn_index: int) -> str:
        msg = (message or "").strip()
        if not msg:
            return "Bạn muốn mình hỗ trợ gì tiếp?"
        if msg.endswith("?"):
            return (
                "Mình ghi nhận câu hỏi trong thread này, "
                "nhưng baseline không nhớ các thread trước đó."
            )
        if turn_index == 1:
            return f"Đã ghi nhận thông tin trong thread: \"{msg[:80]}\"."
        return "Mình tiếp tục ghi nhận thông tin trong thread hiện tại."

    # ------------------------------------------------------------------
    # Optional live path
    # ------------------------------------------------------------------

    def _maybe_build_langchain_agent(self):
        """Optionally wire a LangChain agent. Returns ``None`` if unavailable."""
        model = build_chat_model(self.config.model)
        if model is None:
            return None
        try:
            # We keep dependencies minimal: a simple Runnable with a system
            # prompt is enough — the baseline never gains persistent memory
            # even in live mode.
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Bạn là một agent baseline. Bạn chỉ nhớ nội dung "
                        "trong thread hiện tại và không có persistent memory.",
                    ),
                    ("placeholder", "{history}"),
                    ("human", "{input}"),
                ]
            )
            return prompt | model | StrOutputParser()
        except Exception:
            return None

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        state = self.sessions.setdefault(thread_id, SessionState())
        history = state.messages
        try:
            response = self.langchain_agent.invoke(
                {"history": history, "input": message}
            )
        except Exception:
            return self._reply_offline(thread_id, message)

        state.messages.append({"role": "user", "content": message})
        state.messages.append({"role": "assistant", "content": response})
        agent_tokens = estimate_tokens(response)
        state.token_usage += agent_tokens
        prompt_tokens_now = sum(estimate_tokens(m["content"]) for m in state.messages)
        state.prompt_tokens_processed += prompt_tokens_now
        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens_processed": prompt_tokens_now,
        }
