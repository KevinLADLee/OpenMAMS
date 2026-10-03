"""
Factory functions for creating components from configuration.

This module provides factory functions to instantiate:
- Embeddings (HuggingFace, Ollama, ModelScope)
- LLMs (OpenAI, Ollama, Qwen, DeepSeek)

All factories use the configuration system for initialization.
"""
import os
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from remembr.config.config_loader import Config


def _get_logger():
    """Lazy import to avoid circular dependency."""
    try:
        from remembr.utils.logging import factory_logger
        return factory_logger
    except ImportError:
        import logging
        return logging.getLogger(__name__)


def load_embeddings(config: Config, device: Optional[str] = None):
    """
    Create embeddings instance from configuration.

    Args:
        config: Configuration object
        device: Optional device override (cuda, cpu, mps)

    Returns:
        Embeddings instance (HuggingFaceEmbeddings, OllamaEmbeddings, or ModelScope)

    Example:
        >>> config = load_config()
        >>> embeddings = load_embeddings(config)
    """
    logger = _get_logger()
    provider = config.get('embedding.provider', 'huggingface')
    model_name = config.get('embedding.model', 'mixedbread-ai/mxbai-embed-large-v1')
    device = device or config.get('embedding.device', 'cpu')

    try:
        if provider == 'ollama':
            base_url = config.get('llm.api_base', 'http://localhost:11434')
            # Strip /v1 suffix if present (Ollama client doesn't need it)
            if base_url.endswith('/v1'):
                base_url = base_url[:-3]
            logger.info(f"Loading Ollama embeddings: {model_name} from {base_url}")
            return OllamaEmbeddings(
                model=model_name,
                base_url=base_url
            )
        elif provider == 'modelscope':
            # ModelScope embeddings using sentence-transformers
            logger.info(f"Loading ModelScope embeddings: {model_name} on {device}")
            try:
                from sentence_transformers import SentenceTransformer

                # Create a wrapper for ModelScope embeddings
                class ModelScopeEmbeddings(HuggingFaceEmbeddings):
                    """Wrapper for ModelScope embeddings using sentence-transformers."""

                    def __init__(self, model_name: str, device: str = 'cpu'):
                        # Download from ModelScope hub and use with sentence-transformers
                        # ModelScope models are compatible with sentence-transformers
                        super().__init__(
                            model_name=model_name,
                            model_kwargs={'device': device},
                            encode_kwargs={'normalize_embeddings': True}
                        )

                return ModelScopeEmbeddings(
                    model_name=model_name,
                    device=device
                )
            except ImportError as e:
                logger.error("sentence-transformers or modelscope not installed")
                raise ImportError(
                    "ModelScope embeddings require sentence-transformers and modelscope packages. "
                    "Install with: pip install sentence-transformers modelscope"
                ) from e
        else:  # huggingface
            logger.info(f"Loading HuggingFace embeddings: {model_name} on {device}")
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device}
            )
    except Exception as e:
        logger.error(f"Failed to load embeddings (provider={provider}, model={model_name}): {e}")
        raise RuntimeError(
            f"Failed to initialize embeddings with provider '{provider}' and model '{model_name}'. "
            f"Error: {e}\n"
            f"Troubleshooting:\n"
            f"- For HuggingFace: Ensure model name is correct and you have internet connection for first download\n"
            f"- For Ollama: Ensure Ollama is running and model is pulled (ollama pull {model_name})\n"
            f"- For ModelScope: Ensure modelscope package is installed and model ID is correct (e.g., damo/nlp_gte_sentence-embedding_multilingual-base)"
        ) from e


def load_llm(config: Config, llm_type: Optional[str] = None, **kwargs):
    """
    Create LLM instance from configuration.

    Supports multiple providers:
    - OpenAI-compatible APIs (GPT-4, GPT-3.5, etc.)
    - Ollama (local models)
    - Qwen (Aliyun)
    - DeepSeek (SiliconFlow)

    Args:
        config: Configuration object
        llm_type: Optional LLM type override (e.g., 'gpt-4o', 'qwen3:8b')
        **kwargs: Additional arguments to pass to LLM constructor

    Returns:
        LLM instance (ChatOpenAI or ChatOllama)

    Example:
        >>> config = load_config()
        >>> llm = load_llm(config)
        >>> llm = load_llm(config, llm_type='gpt-4o')  # Override from config
    """
    logger = _get_logger()

    # Use override or config value
    llm_type = llm_type or config.get('llm.backend', 'qwen3:8b')

    # Get common parameters from config
    temperature = kwargs.get('temperature', config.get('llm.temperature', 0.0))
    max_tokens = kwargs.get('max_tokens', config.get('llm.max_tokens', 2048))
    num_ctx = kwargs.get('num_ctx', config.get('llm.num_ctx', 8192))

    # Get explicit provider from config (new feature)
    provider = config.get('llm.provider', None)

    try:
        # Determine provider: use explicit config first, then fall back to model name heuristics
        if provider:
            # Explicit provider specified - more reliable and clear
            logger.info(f"Using explicit provider: {provider}")

            if provider in ['openai', 'vllm', 'openai-compatible']:
                # OpenAI-compatible API (OpenAI, VLLM, or other compatible services)
                api_base = config.get('llm.api_base', 'https://api.openai.com/v1')
                api_key = config.get('llm.api_key') or os.getenv('OPENAI_API_KEY', '')

                if provider == 'openai' and not api_key:
                    logger.warning(f"No API key configured for OpenAI. Set OPENAI_API_KEY environment variable.")

                logger.info(f"Loading {provider.upper()} LLM: {llm_type}")
                return ChatOpenAI(
                    model=llm_type,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    base_url=api_base,
                    api_key=api_key or "EMPTY"  # Some services don't need API key
                )

            elif provider == 'ollama':
                # Ollama native API
                api_base = config.get('llm.api_base', 'http://localhost:11434')

                # Strip /v1 suffix if present (Ollama client doesn't need it)
                if api_base.endswith('/v1'):
                    api_base = api_base[:-3]
                    logger.info(f"Stripped /v1 suffix from api_base for Ollama: {api_base}")

                logger.info(f"Loading Ollama LLM: {llm_type} from {api_base}")
                return ChatOllama(
                    model=llm_type,
                    temperature=temperature,
                    num_ctx=num_ctx,
                    base_url=api_base,
                    reasoning=False
                )

            elif provider == 'dashscope':
                # Aliyun DashScope (Qwen cloud)
                api_base = config.get('llm.api_base', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
                api_key = config.get('llm.api_key') or os.getenv('DASHSCOPE_API_KEY', '')

                if not api_key:
                    logger.warning(f"No API key configured for DashScope. Set DASHSCOPE_API_KEY environment variable.")

                logger.info(f"Loading DashScope LLM: {llm_type}")
                return ChatOpenAI(
                    model=llm_type,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    base_url=api_base,
                    api_key=api_key
                )

            elif provider == 'siliconflow':
                # SiliconFlow (DeepSeek, etc.)
                api_base = config.get('llm.api_base', 'https://api.siliconflow.cn/v1')
                api_key = config.get('llm.api_key') or os.getenv('SILICONFLOW_API_KEY', '')

                if not api_key:
                    logger.warning(f"No API key configured for SiliconFlow. Set SILICONFLOW_API_KEY environment variable.")

                logger.info(f"Loading SiliconFlow LLM: {llm_type}")
                return ChatOpenAI(
                    model=llm_type,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    base_url=api_base,
                    api_key=api_key
                )

            else:
                raise ValueError(
                    f"Unknown provider '{provider}'. "
                    f"Supported providers: openai, vllm, openai-compatible, ollama, dashscope, siliconflow"
                )

        # Fall back to heuristic-based provider detection (for backward compatibility)
        logger.info("No explicit provider specified, using model name heuristics")

        if 'gpt' in llm_type.lower():
            # OpenAI models (GPT-4, GPT-3.5, etc.)
            api_base = config.get('llm.api_base', 'https://api.openai.com/v1')
            api_key = config.get('llm.api_key') or os.getenv('OPENAI_API_KEY', '')

            if not api_key:
                logger.warning(f"No API key configured for OpenAI model '{llm_type}'. Set OPENAI_API_KEY environment variable.")

            logger.info(f"Loading OpenAI LLM: {llm_type}")
            return ChatOpenAI(
                model=llm_type,
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=api_base,
                api_key=api_key
            )

        elif 'qwen' in llm_type.lower() and ('235b' in llm_type or 'plus' in llm_type or '72b' in llm_type):
            # Qwen cloud models via Aliyun/DashScope
            api_base = config.get('llm.api_base', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
            api_key = config.get('llm.api_key') or os.getenv('DASHSCOPE_API_KEY', '')

            if not api_key:
                logger.warning(f"No API key configured for Qwen cloud model '{llm_type}'. Set DASHSCOPE_API_KEY environment variable.")

            logger.info(f"Loading Qwen cloud LLM: {llm_type}")
            return ChatOpenAI(
                model=llm_type,
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=api_base,
                api_key=api_key
            )

        elif 'deepseek' in llm_type.lower():
            # DeepSeek via SiliconFlow
            api_base = config.get('llm.api_base', 'https://api.siliconflow.cn/v1')
            api_key = config.get('llm.api_key') or os.getenv('SILICONFLOW_API_KEY', '')

            if not api_key:
                logger.warning(f"No API key configured for DeepSeek model '{llm_type}'. Set SILICONFLOW_API_KEY environment variable.")

            logger.info(f"Loading DeepSeek LLM: {llm_type}")
            return ChatOpenAI(
                model=llm_type,
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=api_base,
                api_key=api_key
            )

        else:
            # Default to Ollama for local models
            api_base = config.get('llm.api_base', 'http://localhost:11434')

            # Strip /v1 suffix if present (Ollama client doesn't need it)
            if api_base.endswith('/v1'):
                api_base = api_base[:-3]

            logger.info(f"Loading Ollama LLM: {llm_type} from {api_base}")
            return ChatOllama(
                model=llm_type,
                temperature=temperature,
                num_ctx=num_ctx,
                base_url=api_base,
                reasoning=False,
            )
    except Exception as e:
        logger.error(f"Failed to load LLM (type={llm_type}): {e}")
        raise RuntimeError(
            f"Failed to initialize LLM '{llm_type}'. Error: {e}\n"
            f"Troubleshooting:\n"
            f"- For Ollama: Ensure Ollama is running and model is pulled (ollama pull {llm_type})\n"
            f"- For cloud APIs: Check API key is set and base URL is correct\n"
            f"- Check network connection and firewall settings"
        ) from e
