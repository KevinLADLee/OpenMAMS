import asyncio
import datetime
import os
import time
from dataclasses import asdict
from pathlib import Path
from time import strftime, localtime
from typing import Any, List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_milvus import Milvus
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

from remembr.memory.memory import Memory, MemoryItem
from remembr.config.config_loader import Config, get_config
from remembr.factories import load_embeddings

FIXED_SUBTRACT=1721761000 # this is just a large value that brings us close to 1970
TEXT_SEARCH_TOP_K = 5
SPATIOTEMPORAL_SEARCH_TOP_K = 4


def build_drone_filter_expr(allowed_drones: Optional[List[int]]) -> Optional[str]:
    """Build a Milvus scalar filter for captions prefixed with ``[UAV N]``.

    The filter is passed into Milvus before ANN ranking so ``k`` is applied to
    the allowed UAV subset instead of to the full collection.
    """
    if allowed_drones is None:
        return None

    drone_ids = sorted({int(drone_id) for drone_id in allowed_drones})
    if not drone_ids:
        return 'caption == "__NO_ALLOWED_UAV__"'

    clauses = [f'caption LIKE "[UAV {drone_id}]%"' for drone_id in drone_ids]
    return "(" + " or ".join(clauses) + ")"


# Path resolution utilities for Milvus Lite
_project_root_cache = None

def get_project_root() -> Path:
    """
    Find the project root directory by searching for .git or requirements.txt.
    Searches upwards from the current file location.
    """
    global _project_root_cache
    if _project_root_cache is not None:
        return _project_root_cache

    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / '.git').exists() or (current / 'requirements.txt').exists():
            _project_root_cache = current
            return current
        current = current.parent

    # Fallback to current working directory
    _project_root_cache = Path.cwd()
    return _project_root_cache


def get_data_dir() -> Path:
    """
    Get the data directory for Milvus Lite databases.
    Uses REMEMBR_DATA_DIR environment variable if set, otherwise defaults to {project_root}/data.
    Creates the directory if it doesn't exist.
    """
    if 'REMEMBR_DATA_DIR' in os.environ:
        data_dir = Path(os.environ['REMEMBR_DATA_DIR'])
    else:
        data_dir = get_project_root() / 'data'

    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir



def ensure_event_loop():
    """Create an asyncio event loop for the current thread if needed."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


class MilvusWrapper:

    def __init__(self, collection_name='test', db_path=None, drop_collection=False):
        """
        Initialize MilvusWrapper with file-based Milvus Lite.

        Args:
            collection_name: Name of the collection
            db_path: Path to the Milvus Lite database file (if None, uses default data dir)
            drop_collection: Whether to drop existing collection
        """
        self.collection_name = collection_name
        if db_path is None:
            db_path = str(get_data_dir() / f"{collection_name}.db")
        self.db_path = db_path
        self.collection = self.connect_to_milvus_collection(collection_name, 1024, db_path=db_path, drop_collection=drop_collection)


    def drop_collection(self):
        utility.drop_collection(self.collection_name)

    def connect_to_milvus_collection(self, collection_name, dim, db_path, drop_collection=False):
        """
        Connect to Milvus Lite collection using file-based storage.

        Args:
            collection_name: Name of the collection
            dim: Dimension of the embedding vectors
            db_path: Path to the Milvus Lite database file
            drop_collection: Whether to drop existing collection
        """
        # Check if connection already exists
        alias = "default"
        try:
            # Try to get existing connection info
            existing_connections = connections.list_connections()
            if alias in existing_connections:
                # Get current connection address
                addr = connections.get_connection_addr(alias)
                current_uri = addr.get('uri', '')

                # If URI is the same, reuse the connection
                if current_uri == db_path:
                    print(f"Reusing existing connection to {db_path}")
                else:
                    # URI is different, disconnect and reconnect
                    print(f"Disconnecting from {current_uri} to connect to {db_path}")
                    connections.disconnect(alias)
                    connections.connect(alias=alias, uri=db_path)
            else:
                # No existing connection, create new one
                connections.connect(alias=alias, uri=db_path)
        except Exception as e:
            # If anything goes wrong, try to disconnect and reconnect
            print(f"Connection check failed: {e}. Attempting to reconnect...")
            try:
                connections.disconnect(alias)
            except:
                pass
            connections.connect(alias=alias, uri=db_path)

        if drop_collection:
            utility.drop_collection(collection_name)

        fields = [
            FieldSchema(name='id', dtype=DataType.VARCHAR, description='ids', is_primary=True, auto_id=False, max_length=1000),
            FieldSchema(name='text_embedding', dtype=DataType.FLOAT_VECTOR, description='embedding vectors', dim=dim),
            FieldSchema(name='position', dtype=DataType.FLOAT_VECTOR, description='position of robot', dim=3),
            FieldSchema(name='theta', dtype=DataType.FLOAT, description='rotation of robot', dim=1),
            FieldSchema(name='time', dtype=DataType.FLOAT_VECTOR, description='time', dim=2),
            FieldSchema(name='caption', dtype=DataType.VARCHAR, description='caption string', max_length=3000),

        ]
        schema = CollectionSchema(fields=fields, description='text image search')
        collection = Collection(name=collection_name, schema=schema)

        # create IVF_FLAT index for collection.
        index_params = {
            'metric_type':'L2',
            'index_type':"IVF_FLAT",
            'params':{"nlist":1024}
        }
        collection.create_index(field_name="text_embedding", index_params=index_params)

        index_params = {
            'metric_type':'L2',
            'index_type':"IVF_FLAT",
            'params':{"nlist":2}
        }
        collection.create_index(field_name="position", index_params=index_params)

        index_params = {
            'metric_type':'L2',
            'index_type':"IVF_FLAT",
            'params':{"nlist":2}
        }
        collection.create_index(field_name="time", index_params=index_params)

        return collection
    
    def insert(self, data_list):
        res = self.collection.insert(data_list)

    def search(self, data):

        self.collection.load()

        BATCH_SIZE = 2
        LIMIT = 10

        param = {
            "metric_type": "L2",
            "params": {
                "nprobe": 1024,
            }
        }

        res = self.collection.search(
            data=[data],
            anns_field="text_embedding",
            param=param,
            batch_size=BATCH_SIZE,
            limit=LIMIT,
            # expr="id > 3",
            output_fields=["id", "text_embedding"]
        )

        return res




class MilvusMemory(Memory):


    def __init__(
        self,
        db_collection_name: str,
        data_dir: Optional[str] = None,
        time_offset=FIXED_SUBTRACT,
        config: Optional[Config] = None,
        embeddings=None
    ):
        """
        Initialize MilvusMemory with Milvus Lite file-based storage.

        Args:
            db_collection_name: Name of the collection
            data_dir: Directory to store database files (if None, uses config or default from get_data_dir())
            time_offset: Time offset for timestamp calculations
            config: Configuration object (if None, will auto-load from config.yaml)
            embeddings: Optional pre-loaded embeddings instance (if None, will load from config)
                       Pass this to share embeddings across components and avoid reloading

        Examples:
            # New config-based approach (recommended)
            >>> from remembr.config.config_loader import load_config
            >>> config = load_config()
            >>> memory = MilvusMemory("my_collection", config=config)

            # Sharing embeddings (optimal for performance)
            >>> embeddings = load_embeddings(config)
            >>> memory = MilvusMemory("my_collection", config=config, embeddings=embeddings)

            # Legacy approach (backward compatible)
            >>> memory = MilvusMemory("my_collection", data_dir="./data")
        """
        # Load config if not provided
        if config is None:
            try:
                config = get_config()
            except:
                # If config not loaded, create minimal config for backward compatibility
                from remembr.config.defaults import get_default_config
                config = Config(get_default_config())

        self.config = config
        self.db_collection_name = db_collection_name
        self.time_offset = time_offset

        # Determine data directory (priority: parameter > config > default)
        if data_dir is None:
            data_dir = config.get('milvus.data_dir')
            if data_dir is None:
                data_dir = str(get_data_dir())

        # Construct database file path
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        self.db_path = str(data_path / f"{db_collection_name}.db")

        # Use provided embeddings or load from config
        if embeddings is not None:
            print(f"✅ Using shared embeddings instance ")
            self.embedder = embeddings
        else:
            print(f"⏳ Loading embeddings from config...")
            self.embedder = load_embeddings(config)

        self.working_memory = []

        self.reset(drop_collection=False)

    def __del__(self):
        """Destructor to ensure cleanup on garbage collection."""
        try:
            self.cleanup()
        except:
            # Silently ignore errors during destruction
            pass


    def insert(self, item: MemoryItem, text_embedding=None):

        memory_dict = asdict(item)
        # Generate unique ID: use timestamp with microseconds + random suffix for uniqueness
        # This prevents overwrites when multiple items are inserted at the same time
        import uuid
        unique_suffix = uuid.uuid4().hex[:8]
        memory_dict['id'] = f"{time.time():.6f}_{unique_suffix}"

        if text_embedding is None:
            text_embedding = self.embedder.embed_query(memory_dict['caption'])

        memory_dict['time'] =  [(memory_dict['time'] - self.time_offset), 0.0]

        memory_dict['text_embedding'] = text_embedding

        self.milv_wrapper.insert([memory_dict])

    def get_working_memory(self) -> list[MemoryItem]:
        return self.working_memory

    def cleanup(self):
        """
        Release all Collection resources and disconnect to prevent conflicts.
        Call this before re-setting or switching to a new collection.
        """
        try:
            # Release MilvusWrapper collection (silent on success, warn on failure)
            if hasattr(self, 'milv_wrapper') and hasattr(self.milv_wrapper, 'collection'):
                if self.milv_wrapper.collection is not None:
                    try:
                        self.milv_wrapper.collection.release()
                    except Exception as e:
                        print(f"Warning: Failed to release MilvusWrapper collection: {e}")

            # Release text_retriever collection
            if hasattr(self, 'text_retriever'):
                try:
                    if hasattr(self.text_retriever, 'vectorstore') and hasattr(self.text_retriever.vectorstore, 'col'):
                        if self.text_retriever.vectorstore.col is not None:
                            self.text_retriever.vectorstore.col.release()
                except Exception as e:
                    print(f"Warning: Failed to release text_retriever collection: {e}")

            # Release position_vector_db collection
            if hasattr(self, 'position_vector_db') and hasattr(self.position_vector_db, 'col'):
                if self.position_vector_db.col is not None:
                    try:
                        self.position_vector_db.col.release()
                    except Exception as e:
                        print(f"Warning: Failed to release position_vector_db collection: {e}")

            # Release time_vector_db collection
            if hasattr(self, 'time_vector_db') and hasattr(self.time_vector_db, 'col'):
                if self.time_vector_db.col is not None:
                    try:
                        self.time_vector_db.col.release()
                    except Exception as e:
                        print(f"Warning: Failed to release time_vector_db collection: {e}")

            # Disconnect to allow reconnection with different URI
            try:
                connections.disconnect("default")
            except Exception as e:
                print(f"Warning: Failed to disconnect: {e}")

            print(f"✅ Cleaned up collection: {self.db_collection_name}")

        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")

    def reset(self, drop_collection=True):
        # Clean up existing collections first to prevent resource conflicts
        self.cleanup()

        if drop_collection:
            print("Resetting memory. We are dropping the current collection")

        ensure_event_loop()

        self.milv_wrapper = MilvusWrapper(self.db_collection_name, db_path=self.db_path, drop_collection=drop_collection)

        text_vector_db = Milvus(
            self.embedder,
            connection_args={"uri": self.db_path},
            collection_name=self.db_collection_name,
            vector_field='text_embedding',
            text_field='caption',
            primary_field='id',
        )
        self.text_retriever = text_vector_db.as_retriever(search_kwargs={"k": 5})


        self.position_vector_db = Milvus(
            self.embedder, # we will ignore this
            connection_args={"uri": self.db_path},
            collection_name=self.db_collection_name,
            vector_field='position',
            text_field='caption',
            primary_field='id',
        )

        self.time_vector_db = Milvus(
            self.embedder, # we will ignore this
            connection_args={"uri": self.db_path},
            collection_name=self.db_collection_name,
            vector_field='time',
            text_field='caption',
            primary_field='id',
        )


    def search_by_position(
        self,
        query: tuple,
        allowed_drones: Optional[List[int]] = None,
        k: int = SPATIOTEMPORAL_SEARCH_TOP_K,
    ) -> str:
        # docs = pos_db.similarity_search_by_vector(np.array(query))
        docs = similarity_search_with_score_by_vector(
            self.position_vector_db,
            np.array(query).astype(float),
            k=k,
            expr=build_drone_filter_expr(allowed_drones),
        )

        self.working_memory += docs 

        docs = self.memory_to_string(docs)

        """Look up things online."""
        return docs

    def search_by_time(
        self,
        hms_time: str,
        allowed_drones: Optional[List[int]] = None,
        k: int = SPATIOTEMPORAL_SEARCH_TOP_K,
    ) -> str:

        # Input is time like 08:20:30
        # need to convert to searchable time
        t = localtime(self.time_offset)
        mdy_date = strftime('%m/%d/%Y', t)
        template = "%m/%d/%Y %H:%M:%S"

        # if the hms_time is already in the mdy hms format without me doing anything, let's just use that.
        # bad llms don't listen :(
        try:
            res = bool(datetime.datetime.strptime(hms_time, template))
        except ValueError:
            res = False

        hms_time = hms_time.strip()
        if not res: # convert to the right format then
            hms_time = mdy_date + ' ' + hms_time

        query = time.mktime(datetime.datetime.strptime(hms_time,template).timetuple()) - self.time_offset
        # convert from hms_time to something searchable


        docs = similarity_search_with_score_by_vector(
            self.time_vector_db,
            np.array([query, 0]),
            k=k,
            expr=build_drone_filter_expr(allowed_drones),
        )

        self.working_memory += docs

        docs = self.memory_to_string(docs)
        # np.unique([doc.metadata['time'][0] for doc in docs])
        """Look up things online."""
        return docs



    def search_by_text(
        self,
        query: str,
        allowed_drones: Optional[List[int]] = None,
        k: int = TEXT_SEARCH_TOP_K,
    ) -> str:
        expr = build_drone_filter_expr(allowed_drones)
        docs = self.text_retriever.vectorstore.similarity_search(
            query,
            k=k,
            expr=expr,
        )
        
        self.working_memory += docs

        docs = self.memory_to_string(docs)

        """Look up things online."""
        return docs
    

    ### Doc formatting for the last LLM
    def memory_to_string(self, memory_list: list[MemoryItem], ref_time: float=None):
        if ref_time == None:
            ref_time = self.time_offset

        out_string = ""
        for doc in memory_list:
            if len(doc.metadata['time']) == 2:
                t = doc.metadata['time'][0]
            else:
                t = doc.metadata['time']
            
            if ref_time:
                t += ref_time
            t = localtime(t)
            t = strftime('%Y-%m-%d %H:%M:%S', t)

            s = f"At time={t}, the robot was at an average position of {np.array(doc.metadata['position']).round(3).tolist()}."
            s += f"The robot saw the following: {doc.page_content}\n\n"
            out_string += s
        return out_string


# NOTE: This version of the code returns the vector
def similarity_search_with_score_by_vector(
        pos_db,
        embedding: List[float],
        k: int = 4,
        param: Optional[dict] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a search on a query string and return results with score.

        For more information about the search parameters, take a look at the pymilvus
        documentation found here:
        https://milvus.io/api-reference/pymilvus/v2.2.6/Collection/search().md

        Args:
            embedding (List[float]): The embedding vector being searched.
            k (int, optional): The amount of results to return. Defaults to 4.
            param (dict): The search params for the specified index.
                Defaults to None.
            expr (str, optional): Filtering expression. Defaults to None.
            timeout (float, optional): How long to wait before timeout error.
                Defaults to None.
            kwargs: Collection.search() keyword arguments.

        Returns:
            List[Tuple[Document, float]]: Result doc and score.
        """
        if pos_db.col is None:
            print("No existing collection to search.")
            return []

        if param is None:
            param = pos_db.search_params

        # Determine result metadata fields with PK
        output_fields = pos_db.fields[:]

        # Fix: Explicitly add vector_field to output_fields if not present
        # Milvus needs vector_field in output_fields to return it
        if pos_db._vector_field not in output_fields:
            output_fields.append(pos_db._vector_field)

        timeout = pos_db.timeout or timeout
        # Perform the search.
        res = pos_db.col.search(
            data=[embedding],
            anns_field=pos_db._vector_field,
            param=param,
            limit=k,
            expr=expr,
            output_fields=output_fields,
            timeout=timeout,
            **kwargs,
        )
        # Organize results
        ret = []
        for result in res[0]:
            # Fix: Use result.entity.fields instead of result.entity.get()
            # result.entity.get() doesn't properly return vector fields
            if hasattr(result.entity, 'fields'):
                data = {x: result.entity.fields.get(x) for x in output_fields}

                # Additional fallback: try direct attribute access if field is None
                vector_field = pos_db._vector_field
                if vector_field in data and data[vector_field] is None:
                    if hasattr(result.entity, vector_field):
                        data[vector_field] = getattr(result.entity, vector_field)
                    elif hasattr(result, 'entity') and hasattr(result.entity, 'get'):
                        alt_val = result.entity.get(vector_field)
                        if alt_val is not None:
                            data[vector_field] = alt_val
            else:
                data = {x: result.entity.get(x) for x in output_fields}

            # IMPORTANT: Save vector_field value BEFORE calling _parse_document
            # _parse_document will remove vector_field from the data dict
            vector_field_name = pos_db._vector_field
            vector_field_value = data.get(vector_field_name)

            doc = pos_db._parse_document(data)

            # Fix: Add vector_field back to metadata after parsing
            # langchain_milvus._parse_document removes vector_field from metadata,
            # but downstream code expects it (e.g., memory_to_string accessing position/time)
            if vector_field_value is not None and vector_field_name not in doc.metadata:
                doc.metadata[vector_field_name] = vector_field_value

            pair = (doc, result.score)
            ret.append(pair)

        return [doc for doc, _ in ret]
