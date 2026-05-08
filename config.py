"""
Configuration for task-agent.
Reads settings from environment variables with sensible defaults.
"""

import os

# Anthropic
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
ANTHROPIC_MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))

# External tool endpoints (optional — tools degrade gracefully if unavailable)
DOC_RAG_URL = os.getenv("DOC_RAG_URL", "http://localhost:8001")
SMART_API_URL = os.getenv("SMART_API_URL", "http://localhost:8002")

# FastAPI
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8080"))
