"""
Default configuration values for ReMEmbR
"""
from typing import Dict, Any


def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration dictionary.

    Returns:
        Dictionary with default configuration values
    """
    return {
        'llm': {
            'backend': 'qwen3:8b',
            'api_base': 'http://localhost:11434/v1',
            'api_key': '',
            'temperature': 0.0,
            'max_tokens': 8192,
            'num_ctx': 8192,
        },
        'embedding': {
            'model': 'mixedbread-ai/mxbai-embed-large-v1',
            'provider': 'huggingface',
            'device': 'cuda',
        },
        'captioner': {
            'type': 'remote',
            'api_base': 'http://localhost:11434/v1',
            'model': 'qwen2.5vl:7b',
        },
        'milvus': {
            'data_dir': './data',
        },
        'gradio': {
            'host': '0.0.0.0',
            'port': 7860,
            'share': False,
        },
    }
