from __future__ import annotations
import re
import json
import yaml
import logging
from typing import Optional, Any

from .llm_providers import call_llm
from .web_tools import web_answer
from .prompts import (
    TIER1_SYSTEM,
    TIER1_USER_TEMPLATE,
    TIER2_SYSTEM,
    TIER2_USER_TEMPLATE,
)

# --- LOGGER CONFIGURATION ---
logger = logging.getLogger("orchestrator")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# --- CONFIGURATION SAFE LOAD ---
def load_config(path: str = "config.yaml") -> dict[str, Any]:
    """Load YAML config safely and validate presence of required keys."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError("Configuration file is malformed or empty.")
        return cfg
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {path}")
    except yaml.YAMLError as e:
        raise ValueError(f"YAML parse error: {e}")


CONFIG = load_config()


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> tags for cleaner LLM output."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


class Orchestrator:
    """
    Orchestrates multi-tier LLM reasoning and retrieval.
    Tier 1 → Fast conversational RAG.
    Tier 2 → Deep reasoning (invoked when low confidence or reasoning cues detected).
    """

    def __init__(self, rag):
        self.rag = rag
        self.policy = CONFIG.get("policy", {})
        self.features = CONFIG.get("features", {})
        self.cooldown_turns = int(self.features.get("tier2_cooldown_turns", 2))
        self._session_meta: dict[str, dict[str, int]] = {}

        logger.info("Orchestrator initialized — cooldown=%s", self.cooldown_turns)

    # --------------------------------------------------------------
    async def handle_chat(self, user_id: str, message: str, namespace: Optional[str] = None) -> dict[str, Any]:
        """
        Handle user message with tiered reasoning logic.
        """
        turn = self._bump_turn(user_id)
        logger.info("User=%s Turn=%d Message='%s'", user_id, turn, message)

        # 1. Retrieve contextual docs (RAG)
        try:
            docs = self.rag.retrieve(
                message, top_k=self.policy.get("top_k", 3), namespace=namespace
            )
        except Exception as e:
            logger.warning("RAG retrieval failed: %s", e)
            docs = []

        context_text = "\n\n".join(f"[{d['source']}] {d['text']}" for d in docs)
        sims = [d.get("score", 0.0) for d in docs]
        avg_sim = sum(sims) / len(sims) if sims else 0.0
        logger.debug("Avg similarity=%.3f (Docs=%d)", avg_sim, len(docs))

        # 2. Tier-1 inference
        t1_prompt = {
            "system": TIER1_SYSTEM,
            "user": TIER1_USER_TEMPLATE.format(user_message=message, context=context_text),
        }

        try:
            t1_raw = await call_llm(tier="tier1", prompt=t1_prompt)
        except Exception as e:
            logger.error("Tier-1 LLM call failed: %s", e)
            return {
                "tier": "tier1",
                "answer": f"(Tier 1 call failed: {e})",
                "citations": [d.get("source") for d in docs],
                "difficulty_report": {"escalated": False, "error": str(e)},
            }

        t1_json = self._parse_tier1_json(t1_raw)
        conf = float(t1_json.get("confidence", 0.0) or 0.0)
        logger.info("Tier-1 confidence=%.2f, AvgSim=%.2f", conf, avg_sim)

        # 3. Escalation decision
        if self._is_trivial_smalltalk(message) or self._is_gratitude_or_closing(message):
            logger.info("Smalltalk or closing detected — skipping escalation.")
            return self._t1_response(t1_json, docs)

        reasons: list[str] = []
        if avg_sim < self.policy.get("min_similarity", 0.3):
            reasons.append(f"low_retrieval({avg_sim:.2f})")
        if conf < self.policy.get("min_self_confidence", 0.5):
            reasons.append(f"low_confidence({conf:.2f})")
        if bool(t1_json.get("needs_web", False)):
            reasons.append("tier1_requested_web")

        fresh = self._looks_fresh(message) if self.features.get("tier2_on_freshness", True) else False
        intent_reason = self._looks_reasoning_intent(message) if self.features.get("tier2_on_reasoning_intent", True) else False

        weak_t1 = any(k in " ".join(reasons) for k in ["low_retrieval", "low_confidence", "tier1_requested_web"])
        cooldown_active = self._within_tier2_cooldown(user_id, turn, self.cooldown_turns)

        if cooldown_active:
            should_escalate = (fresh or intent_reason) and weak_t1
        else:
            should_escalate = fresh or intent_reason or weak_t1

        logger.info(
            "Escalation check — WeakT1=%s Fresh=%s ReasoningIntent=%s Cooldown=%s → Escalate=%s",
            weak_t1, fresh, intent_reason, cooldown_active, should_escalate,
        )

        if not should_escalate:
            return self._t1_response(t1_json, docs)

        # 4. Tier-2 reasoning (optional web)
        needs_web = bool(self.features.get("use_web", False)) and fresh
        web_ctx, web_citations = ("", [])
        if needs_web:
            try:
                web_ctx, web_citations = await web_answer(message)
                logger.info("Web context fetched (%d citations)", len(web_citations))
            except Exception as e:
                web_ctx = f"(Web retrieval failed: {e})"
                logger.warning("Web answer failed: %s", e)
                web_citations = []

        difficulty = {
            "avg_retrieval_similarity": avg_sim,
            "tier1_confidence": conf,
            "reasons": sorted(
                set(reasons + (["freshness"] if fresh else []) + (["reasoning_intent"] if intent_reason else []))
            ),
            "needs_web": needs_web,
        }

        t2_prompt = {
            "system": TIER2_SYSTEM,
            "user": TIER2_USER_TEMPLATE.format(
                user_message=message,
                retrieved_context=context_text,
                difficulty_report=json.dumps(difficulty),
                web_context=web_ctx,
            ),
        }

        try:
            t2 = await call_llm(tier="tier2", prompt=t2_prompt)
            t2 = _strip_think_blocks(t2)
            logger.info("Tier-2 response generated successfully.")
        except Exception as e:
            t2 = f"(Tier 2 call failed: {e})"
            logger.error("Tier-2 LLM call failed: %s", e)

        self._mark_tier2(user_id, turn)

        logger.info("Returning Tier-2 answer for user %s (Reasons: %s)", user_id, difficulty["reasons"])
        return {
            "tier": "tier2",
            "answer": t2,
            "citations": [d.get("source") for d in docs] + web_citations,
            "difficulty_report": {"escalated": True, **difficulty},
        }

    # --------------------------------------------------------------
    # Helper methods

    def _t1_response(self, t1_json: dict[str, Any], docs: list[dict[str, Any]]) -> dict[str, Any]:
        """Return standardized Tier-1 response structure."""
        logger.info("Returning Tier-1 response.")
        return {
            "tier": "tier1",
            "answer": t1_json.get("answer", "") or "",
            "citations": [d.get("source") for d in docs],
            "difficulty_report": {"escalated": False, "reasons": t1_json.get("reasons", [])},
        }

    def _parse_tier1_json(self, t1_raw: str) -> dict[str, Any]:
        """Attempt to parse Tier-1 JSON, falling back gracefully."""
        try:
            data = json.loads(t1_raw)
            if not isinstance(data, dict):
                raise ValueError("Tier 1 did not return JSON.")
            data.setdefault("answer", "")
            data.setdefault("confidence", 0.0)
            data.setdefault("needs_web", False)
            data.setdefault("reasons", [])
            return data
        except Exception:
            logger.warning("Tier-1 JSON parsing failed, using fallback.")
            return {
                "answer": (t1_raw or "").strip() or "I'm not fully sure.",
                "confidence": 0.0,
                "needs_web": False,
                "reasons": ["unparseable self-report"],
            }

    def _looks_fresh(self, message: str) -> bool:
        """Check if query requires fresh or recent information."""
        return any(re.search(rx, message) for rx in self.policy.get("needs_freshness_patterns", []))

    def _looks_reasoning_intent(self, message: str) -> bool:
        """Detect if message implies reasoning or analytical intent."""
        return any(re.search(rx, message) for rx in self.policy.get("reasoning_intent_patterns", []))

    def _is_trivial_smalltalk(self, message: str) -> bool:
        """Detect simple greetings or non-informational messages."""
        m = message.strip().lower()
        common = {
            "hi", "hey", "hello", "hi there", "hey there", "yo", "sup", "hiya",
            "hola", "good morning", "good afternoon", "good evening"
        }
        return len(m) <= 20 and m in common

    def _is_gratitude_or_closing(self, message: str) -> bool:
        """Detect thank-you or conversation-closing phrases."""
        m = message.strip().lower()
        patterns = [
            r"^thanks[.!]?$", r"^thank you[.!]?$", r"^thx[.!]?$", r"^ty[.!]?$",
            r"^ok(ay)?[.!]?$", r"^got it[.!]?$", r"^cool[.!]?$",
            r"^bye[.!]?$", r"^goodbye[.!]?$", r"^see you[.!]?$"
        ]
        return any(re.match(p, m) for p in patterns)

    def _bump_turn(self, user_id: str) -> int:
        """Increment and return current user conversation turn."""
        meta = self._session_meta.setdefault(user_id, {"turn": 0, "last_tier2_turn": -999})
        meta["turn"] += 1
        return meta["turn"]

    def _mark_tier2(self, user_id: str, turn: int):
        """Mark this turn as Tier-2 escalation for cooldown tracking."""
        meta = self._session_meta.setdefault(user_id, {"turn": 0, "last_tier2_turn": -999})
        meta["last_tier2_turn"] = turn

    def _within_tier2_cooldown(self, user_id: str, turn: int, cooldown_turns: int) -> bool:
        """Check whether the user is within the Tier-2 cooldown window."""
        meta = self._session_meta.setdefault(user_id, {"turn": 0, "last_tier2_turn": -999})
        last_t2 = meta.get("last_tier2_turn", -999)
        return (turn - last_t2) <= cooldown_turns
