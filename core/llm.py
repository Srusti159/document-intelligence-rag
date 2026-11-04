"""
Groq LLM Engine
Handles ultra-low-latency generation, hallucination guardrails, and streaming token delivery.
"""

import os
import time
from typing import Generator, List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


SYSTEM_GROUNDING_PROMPT = """You are an expert Document Intelligence AI assistant.
Your mission is to provide accurate, strictly grounded answers based ONLY on the provided context citations.

STRICT INSTRUCTIONS:
1. Grounding Rule: Answer the user's question using ONLY the provided facts in the citations. Do not fabricate or assume external information.
2. Citation Requirement: Whenever you state a claim or fact, cite the reference using [Citation X] format (e.g. "According to the financial breakdown [Citation 1], Q3 revenue rose 14%...").
3. Low Confidence / Missing Evidence: If the provided context does not contain enough information to answer the question confidently, explicitly state:
   "I cannot answer this question based on the provided document context, as the confidence threshold was not met or the information is not present."
4. Be clear, professional, structured, and concise.
"""


class GroqLLMEngine:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model_name = model_name
        self.temperature = temperature
        self._init_client()

    def _init_client(self):
        if not self.api_key:
            self.client = None
            return

        self.client = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=self.temperature,
            streaming=True
        )

    def is_configured(self) -> bool:
        return bool(self.api_key and self.client)

    def generate_grounded_response_stream(
        self,
        query: str,
        context_str: str,
        chat_history: List[Dict[str, str]] = None
    ) -> Generator[str, None, Dict[str, Any]]:
        """
        Streams response tokens in real-time, yielding chunks and returning generation metrics.
        """
        if not self.is_configured():
            yield "⚠️ Groq API key is not configured. Please enter a valid GROQ_API_KEY in the sidebar."
            return

        messages = [SystemMessage(content=SYSTEM_GROUNDING_PROMPT)]

        # Append recent conversation history (up to last 4 turns)
        if chat_history:
            for turn in chat_history[-4:]:
                if turn.get("role") == "user":
                    messages.append(HumanMessage(content=turn.get("content", "")))
                elif turn.get("role") == "assistant":
                    messages.append(AIMessage(content=turn.get("content", "")))

        # Current grounded user prompt
        user_prompt = f"""DOCUMENT CONTEXT WITH CITATIONS:
----------------------------------------
{context_str}
----------------------------------------

USER QUESTION:
{query}

Please provide your grounded response with citations."""

        messages.append(HumanMessage(content=user_prompt))

        start_time = time.perf_counter()
        token_count = 0

        try:
            for chunk in self.client.stream(messages):
                token_count += 1
                yield chunk.content
        except Exception as e:
            yield f"\n\n[Error during LLM generation: {str(e)}]"

        latency_ms = round((time.perf_counter() - start_time) * 1000, 1)
        tokens_per_sec = round(token_count / max(0.001, (latency_ms / 1000)), 1)

        return {
            "latency_ms": latency_ms,
            "token_count": token_count,
            "tokens_per_sec": tokens_per_sec
        }
