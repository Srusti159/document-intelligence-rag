"""
Conversation Memory Manager
Tracks query history, maintains context sessions, and supports history pruning.
"""

from typing import List, Dict, Any


class ConversationMemoryManager:
    def __init__(self, max_history_turns: int = 10):
        self.max_history_turns = max_history_turns
        self.history: List[Dict[str, Any]] = []

    def add_turn(
        self,
        query: str,
        response: str,
        citations: List[Dict[str, Any]],
        stats: Dict[str, Any]
    ):
        turn = {
            "query": query,
            "response": response,
            "citations": citations,
            "retrieval_stats": stats
        }
        self.history.append(turn)
        if len(self.history) > self.max_history_turns:
            self.history = self.history[-self.max_history_turns:]

    def get_chat_history_for_llm(self) -> List[Dict[str, str]]:
        formatted = []
        for turn in self.history:
            formatted.append({"role": "user", "content": turn["query"]})
            formatted.append({"role": "assistant", "content": turn["response"]})
        return formatted

    def clear(self):
        self.history.clear()
