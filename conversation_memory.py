import json
import sqlite3
from contextlib import closing
from pathlib import Path


DB_PATH = (
    Path(__file__).resolve().parent
    / "memory_db"
    / "conversations.sqlite3"
)


class ConversationMemory:
    def __init__(
        self,
        conversation_id: str = "default",
        db_path: Path = DB_PATH,
    ):
        if not conversation_id.strip():
            raise ValueError("The conversation ID cannot be empty.")

        self.conversation_id = conversation_id
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    user_query TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            connection.commit()

    def save_turn(self, user_query: str, result: dict):
        """Save one user request and its assistant response."""
        if not user_query.strip():
            raise ValueError("The query cannot be empty.")

        answer = result.get("answer")

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(
                "The result must contain a non-empty answer."
            )

        result_json = json.dumps(result, ensure_ascii=False)

        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute("""
                INSERT INTO conversation_turns
                    (conversation_id, user_query, result_json)
                VALUES (?, ?, ?)
            """, (
                self.conversation_id,
                user_query,
                result_json,
            ))
            connection.commit()

    def get_turns(self, limit_turns: int = 4) -> list[dict]:
        """Load recent complete turns in chronological order."""
        if limit_turns < 1:
            raise ValueError("limit_turns must be at least 1.")

        with closing(sqlite3.connect(self.db_path)) as connection:
            rows = connection.execute("""
                SELECT user_query, result_json
                FROM conversation_turns
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
            """, (
                self.conversation_id,
                limit_turns,
            )).fetchall()

        return [
            {
                "user_query": query,
                "result": json.loads(result_json),
            }
            for query, result_json in reversed(rows)
        ]

    def get_messages(self, limit_turns: int = 4) -> list[dict]:
        """Convert stored turns into user and assistant messages."""
        messages = []

        for turn in self.get_turns(limit_turns):
            messages.append({
                "role": "user",
                "content": turn["user_query"],
            })
            messages.append({
                "role": "assistant",
                "content": turn["result"]["answer"],
            })

        return messages


if __name__ == "__main__":
    memory = ConversationMemory("memory_demo")

    memory.save_turn(
        "Explain this code: print(6)",
        {
            "intent": "EXPLAIN",
            "status": "EXPLAINED",
            "answer": "It prints the number 6.",
            "source_ids": [],
            "evaluation_reason": None,
        },
    )

    # Open another instance and reload the saved turn.
    reopened_memory = ConversationMemory("memory_demo")
    messages = reopened_memory.get_messages(limit_turns=1)

    print("Saved one conversation turn.")
    print(f"Loaded messages: {len(messages)}")

    for message in messages:
        print(f"{message['role']}: {message['content']}")

    print(f"Database file: {DB_PATH}")