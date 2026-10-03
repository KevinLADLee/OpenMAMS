"""Configure and run the ReMEmbR question-answering agent."""
import ast
import json
import math
from pathlib import Path
import re
import signal


def load_config(path=None):
    from remembr.config.config_loader import Config
    return Config(json.loads(Path(path or Path(__file__).with_name("config.json")).read_text()))


def safe_literal(value):
    """Accept model data without executing it; retain unknown coordinates as None."""
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            if re.sub(r"\s+", "", value).lower() in {
                "(x,y,z)", "[x,y,z]", "(x,y,yaw)", "[x,y,yaw]"
            }:
                return None
            raise ValueError("Model returned an invalid literal") from None
    if isinstance(parsed, dict) and any(k in parsed for k in ("text", "description", "position")):
        position = parsed.get("position")
        if isinstance(position, str):
            try:
                position = ast.literal_eval(position)
            except (ValueError, SyntaxError):
                position = None
        if isinstance(position, dict) and "x" in position and "y" in position:
            position = [position["x"], position["y"], position.get("yaw", position.get("z"))]
        valid = isinstance(position, (list, tuple)) and len(position) == 3
        valid = valid and all(type(x) in (int, float) and math.isfinite(x) for x in position)
        parsed["position"] = list(position) if valid else None
        parsed.setdefault("text", str(parsed.get("description", "")))
        for key in ("time", "binary", "duration"):
            parsed.setdefault(key, None)
    return parsed


def bounded_retry(state, func):
    for attempt in range(3):
        try:
            return func(state)
        except Exception:
            if attempt == 2:
                raise


class QuestionTimeout(BaseException):
    """Escape the retained core's broad exception handlers on Linux."""


def on_timeout(signum, frame):
    raise QuestionTimeout("Question timed out; no accuracy result was produced")


class QA:
    def __init__(self, db, uavs, config=None, time_offset=None):
        from remembr.agents import remembr_agent
        from remembr.memory.milvus_memory import MilvusMemory, FIXED_SUBTRACT
        from .filtered_memory import DroneFilteredMemory

        self.db = Path(db).resolve()
        if not self.db.is_file() or self.db.suffix != ".db":
            raise ValueError("--db must be an existing ReMEmbR .db file")
        if not uavs or any(type(u) is not int or u < 1 for u in uavs):
            raise ValueError("Supply at least one positive UAV ID")
        # Validate before the original constructor can create an empty collection.
        from pymilvus import MilvusClient
        client = MilvusClient(uri=str(self.db))
        try:
            if self.db.stem not in client.list_collections():
                raise ValueError("Database filename must match an existing ReMEmbR collection")
            fields = {f["name"] for f in client.describe_collection(self.db.stem)["fields"]}
            if not {"caption", "position", "theta", "time", "text_embedding"} <= fields:
                raise ValueError("This is not the retained ReMEmbR database schema")
        finally:
            client.close()
        self.uavs = sorted(set(uavs))
        self.config = config or load_config()
        meta_path = self.db.parent / "memory.json"
        self.metadata = json.loads(meta_path.read_text()) if meta_path.exists() else None
        if self.metadata:
            if not self.metadata.get("completed"):
                raise ValueError("Memory build is incomplete")
            if self.metadata["embedding"] != self.config.get("embedding"):
                raise ValueError("Use the same embedding settings used to build the database")
            missing = set(self.uavs) - set(self.metadata["uavs"])
            if missing:
                raise ValueError(f"UAVs absent from this memory: {sorted(missing)}")
        offset = time_offset if time_offset is not None else (
            self.metadata["time_offset"] if self.metadata else FIXED_SUBTRACT)

        remembr_agent.eval = safe_literal
        remembr_agent.try_except_continue = bounded_retry
        self.agent = remembr_agent.ReMEmbRAgent(config=self.config)
        self.base = MilvusMemory(self.db.stem, data_dir=str(self.db.parent),
                                 time_offset=offset, config=self.config, embeddings=self.agent.embeddings)
        self.memory = DroneFilteredMemory(self.base, self.uavs)
        self.agent.set_memory(self.memory)

    def ask(self, question, timeout=90):
        from langchain_community.chat_message_histories import ChatMessageHistory
        if not question.strip() or timeout <= 0:
            raise ValueError("Question must be nonempty and timeout positive")
        self.agent.chat_history = ChatMessageHistory()
        self.agent.agent_call_count = 0
        self.agent.previous_tool_requests = "These are the tools I have previously used so far: \n"
        self.base.working_memory = []
        stages, answer = [], None
        previous = signal.signal(signal.SIGALRM, on_timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            # Follow the original evaluator's graph.stream entry, not AgentOutput coercion.
            for step in self.agent.graph.stream({"messages": [("user", question)]}):
                for name, value in step.items():
                    for item in value["messages"]:
                        content = item if isinstance(item, str) else item.content
                        stages.append({"stage": name, "content": content})
                        if name == "generate":
                            answer = safe_literal(content)
            if not isinstance(answer, dict):
                raise RuntimeError("Agent produced no structured final answer")
        except QuestionTimeout as exc:
            raise TimeoutError(str(exc)) from None
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        docs = [{"caption": d.page_content,
                 "position": d.metadata.get("position"), "time": d.metadata.get("time")}
                for d in self.memory.get_working_memory()]
        return {"question": question, "answer": answer, "allowed_uavs": self.uavs,
                "last_retrieved_memories": docs, "trace": stages}

    def close(self):
        self.base.cleanup()
