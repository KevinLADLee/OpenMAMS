"""Original multi-UAV memory wrapper; see provenance.json."""
import re
from typing import Optional, List
from remembr.memory.milvus_memory import MilvusMemory

class DroneFilteredMemory:
    """Wrapper for MilvusMemory that filters by drone IDs."""

    def __init__(self, base_memory: MilvusMemory, allowed_drones: Optional[List[int]] = None):
        """
        Initialize filtered memory.

        Args:
            base_memory: Base MilvusMemory instance
            allowed_drones: List of allowed drone IDs (e.g., [1, 2] for drone_01, drone_02)
        """
        self.base_memory = base_memory
        self.allowed_drones = allowed_drones
        self.working_memory = []

    def _extract_uav_id_from_caption(self, caption: str) -> Optional[int]:
        """
        Extract UAV ID from caption text.

        Args:
            caption: Caption text that may contain "[UAV X]" prefix

        Returns:
            UAV ID as integer, or None if not found
        """
        match = re.match(r'\[UAV\s+(\d+)\]', caption)
        if match:
            return int(match.group(1))
        return None

    def _filter_by_drone(self, docs: list) -> list:
        """
        Filter documents by allowed drone IDs.

        Args:
            docs: List of Document objects with page_content containing captions

        Returns:
            Filtered list of documents
        """
        if self.allowed_drones is None:
            return docs

        filtered_docs = []
        for doc in docs:
            # Extract UAV ID from caption
            uav_id = self._extract_uav_id_from_caption(doc.page_content)
            if uav_id in self.allowed_drones:
                filtered_docs.append(doc)

        return filtered_docs

    def search_by_text(self, query: str) -> str:
        """Search by text and filter results by drone ID."""
        # Clear working_memory before search to avoid stale state
        self.base_memory.working_memory = []

        # Get raw results from base memory
        # This returns documents in working_memory
        result_str = self.base_memory.search_by_text(
            query, allowed_drones=self.allowed_drones
        )

        # Filter the working_memory by drone ID
        if self.allowed_drones is not None:
            # Apply filter and update working_memory
            filtered = self._filter_by_drone(self.base_memory.working_memory)
            self.base_memory.working_memory = filtered
            # Use filtered results for generating the response string
            result_str = self.base_memory.memory_to_string(filtered)

        return result_str

    def search_by_position(self, query: tuple) -> str:
        """Search by position and filter results by drone ID."""
        # Clear working_memory before search to avoid stale state
        self.base_memory.working_memory = []

        result_str = self.base_memory.search_by_position(
            query, allowed_drones=self.allowed_drones
        )

        # Filter the working_memory by drone ID
        if self.allowed_drones is not None:
            # Apply filter and update working_memory
            filtered = self._filter_by_drone(self.base_memory.working_memory)
            self.base_memory.working_memory = filtered
            # Use filtered results for generating the response string
            result_str = self.base_memory.memory_to_string(filtered)

        return result_str

    def search_by_time(self, query: str) -> str:
        """Search by time and filter results by drone ID."""
        # Clear working_memory before search to avoid stale state
        self.base_memory.working_memory = []

        result_str = self.base_memory.search_by_time(
            query, allowed_drones=self.allowed_drones
        )

        # Filter the working_memory by drone ID
        if self.allowed_drones is not None:
            # Apply filter and update working_memory
            filtered = self._filter_by_drone(self.base_memory.working_memory)
            self.base_memory.working_memory = filtered
            # Use filtered results for generating the response string
            result_str = self.base_memory.memory_to_string(filtered)

        return result_str

    def get_working_memory(self) -> list:
        """Get working memory from base memory."""
        return self.base_memory.get_working_memory()

    def __getattr__(self, name):
        """Delegate all other attributes to base memory."""
        return getattr(self.base_memory, name)
