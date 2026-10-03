from langchain_core.embeddings import Embeddings

from multi_agent_qa.filtered_memory import DroneFilteredMemory
from multi_agent_qa.runtime import load_config
from remembr.memory.memory import MemoryItem
from remembr.memory.milvus_memory import MilvusMemory


class TestEmbeddings(Embeddings):
    def embed_query(self, text):
        # Disallowed UAV has closer vectors, exposing post-ranking-only filtering.
        return [1.0 if "UAV 2" in text or text == "car" else 2.0] + [0.0] * 1023
    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


def test_uav_filter_precedes_all_three_searches(tmp_path):
    base = MilvusMemory("test", data_dir=str(tmp_path), time_offset=0,
                        config=load_config(), embeddings=TestEmbeddings())
    try:
        for uav in (1,2):
            for i in range(7):
                base.insert(MemoryItem(f"[UAV {uav}] car {i}", float(i), [float(i),0.0,0.0], 0.0))
        base.milv_wrapper.collection.flush()
        memory = DroneFilteredMemory(base, [1])
        for method, value, count in (("search_by_text", "car", 5),
                                      ("search_by_position", (0,0,0), 4),
                                      ("search_by_time", "00:00:00", 4)):
            getattr(memory, method)(value)
            docs = memory.get_working_memory()
            assert len(docs) == count
            assert all(d.page_content.startswith("[UAV 1]") for d in docs)
    finally:
        base.cleanup()
