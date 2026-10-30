"""
Ampolin AI Trading Agent - Multi-Provider LLM Client
Seamlessly orchestrates Google Gemini (Primary) and Groq (High-Speed Fallback)
Ensures zero downtime and continuous operation even when Gemini free tier hits 429 quota limits.
"""

import json
import logging
import os
import urllib.request
from typing import Optional, Dict, Any, List

from config import GEMINI_API_KEY, GEMINI_MODEL, GROQ_API_KEY, GROQ_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AmpolinLLM")

# Try importing official google-genai SDK
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class MultiProviderLLM:
    """
    Unified LLM manager with automatic failover:
    Primary: Google Gemini (gemini-3.5-flash-lite / gemini-3.8-flash)
    Secondary / Fallback: Groq (qwen/qwen3.8-27b / openai/gpt-oss-120b)
    """

    def __init__(self):
        self.gemini_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        self.gemini_model = GEMINI_MODEL or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.groq_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        self.groq_model = GROQ_MODEL or os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

        self.gemini_client = None
        if self.gemini_key and GENAI_AVAILABLE:
            try:
                self.gemini_client = genai.Client(api_key=self.gemini_key)
                logger.info(f"Gemini client initialized with model '{self.gemini_model}'")
            except Exception as e:
                logger.warning(f"Could not initialize Gemini Client: {e}")

        if self.groq_key:
            logger.info(f"Groq high-speed fallback initialized with model '{self.groq_model}'")

    def call_groq(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.4,
    ) -> Optional[str]:
        """Direct, high-speed call to Groq API using standard OpenAI-compatible completions."""
        api_key = self.groq_key or os.getenv("GROQ_API_KEY", "")
        if not api_key:
            logger.warning("Groq API key not configured.")
            return None

        primary_model = model or self.groq_model or "qwen/qwen3.8-27b"
        groq_models_to_try = [primary_model, "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
        # Deduplicate
        seen = set()
        models = [m for m in groq_models_to_try if not (m in seen or seen.add(m))]

        for mod in models:
            try:
                payload = {
                    "model": mod,
                    "messages": messages,
                    "temperature": 0.1 if json_mode else temperature,
                }
                if json_mode:
                    payload["response_format"] = {"type": "json_object"}

                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "User-Agent": "GroqPythonClient/1.0",
                    },
                    data=data,
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                    content = res["choices"][0]["message"]["content"]
                    logger.info(f"[LLM] Groq ({mod}) responded successfully.")
                    return content
            except Exception as e:
                logger.warning(f"[LLM] Groq ({mod}) failed: {e}. Trying fallback model...")
                continue

        return None

    def generate_chat_response(
        self,
        user_prompt: str,
        system_prompt: str = "",
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[str]:
        """
        Chat completion with multi-turn conversation memory and automatic failover:
        1. Attempts Gemini.
        2. If rate-limited (HTTP 429 / Quota / Timeout), immediately falls back to Groq.
        """
        # Format conversation history string
        history_lines = []
        if history:
            for item in history[-8:]:  # Preserve last 8 turns for continuity
                r = "Trader" if item.get("role") == "user" else "Ampolin AI"
                history_lines.append(f"{r}: {item.get('content', '')}")
        history_str = "\n".join(history_lines)

        full_prompt = system_prompt
        if history_str:
            full_prompt += f"\n\n=== RECENT DIALOGUE THREAD ===\n{history_str}"
        full_prompt += f"\n\nTrader: {user_prompt}\nAmpolin AI:"

        # 1. Try Gemini
        if self.gemini_client:
            models_to_try = [self.gemini_model, "gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.5-flash"]
            seen = set()
            gemini_models = [m for m in models_to_try if not (m in seen or seen.add(m))]

            for model in gemini_models:
                try:
                    res = self.gemini_client.models.generate_content(
                        model=model,
                        contents=full_prompt,
                    )
                    if res and res.text:
                        logger.info(f"[LLM] Gemini ({model}) responded successfully.")
                        return res.text.strip()
                except Exception as e:
                    logger.warning(f"[LLM] Gemini ({model}) rate limit or error: {e}")
                    continue

        # 2. Seamless Fallback to Groq
        logger.info("[LLM] Gemini quota or availability exhausted. Seamlessly switching to Groq...")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            for item in history[-8:]:
                role = "assistant" if item.get("role") in ("assistant", "model") else "user"
                messages.append({"role": role, "content": item.get("content", "")})
        messages.append({"role": "user", "content": user_prompt})

        groq_resp = self.call_groq(messages=messages, json_mode=False)
        if groq_resp:
            return groq_resp.strip()

        return None

    def evaluate_setup_json(self, prompt: str, schema_class: Any) -> Optional[Dict[str, Any]]:
        """
        Evaluates a trading candidate setup with strict JSON structure:
        1. Tries Gemini with Pydantic JSON schema.
        2. On quota limit, falls back to Groq in json_object mode.
        """
        # 1. Try Gemini
        if self.gemini_client:
            models_to_try = [self.gemini_model, "gemini-3.5-flash-lite", "gemini-3.8-flash"]
            for model in models_to_try:
                try:
                    res = self.gemini_client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=schema_class,
                            temperature=0.1,
                        ),
                    )
                    if res and res.text:
                        data = json.loads(res.text)
                        logger.info(f"[LLM] Gemini ({model}) generated trade decision JSON.")
                        return data
                except Exception as e:
                    logger.warning(f"[LLM] Gemini ({model}) evaluation error: {e}")
                    continue

        # 2. Fallback to Groq
        logger.info("[LLM] Gemini unavailable for decision evaluation. Switching to Groq JSON mode...")
        system_instruction = (
            "You are an elite institutional quantitative swing strategist. "
            "Evaluate the candidate setup and return a strict JSON object with fields: "
            "ticker (str), action (BUY, SELL, or PASS), confidence (float between 0.0 and 1.0), "
            "entry_price (float), stop_loss (float), take_profit (float), risk_reward_ratio (float), "
            "holding_period_days (int), rationale (str), key_risks (str)."
        )
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]
        groq_resp = self.call_groq(messages=messages, json_mode=True)
        if groq_resp:
            try:
                data = json.loads(groq_resp)
                # Normalize key_risks if returned as list
                if isinstance(data.get("key_risks"), list):
                    data["key_risks"] = " | ".join(data["key_risks"])
                logger.info("[LLM] Groq generated valid trade decision JSON.")
                return data
            except Exception as e:
                logger.error(f"[LLM] Failed to parse Groq JSON: {e}")

        return None
