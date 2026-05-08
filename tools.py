"""
Tool definitions and executors for task-agent.

Each tool has:
  - A schema (passed to the Anthropic API as a tool definition)
  - An executor function (called when the model invokes the tool)

Tools degrade gracefully — if an external service is unavailable,
they return an informative error rather than crashing the agent.
"""

import math
import httpx

import config

# ── Tool schemas (passed to Anthropic API) ────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "search_docs",
        "description": (
            "Search a document knowledge base using a natural language query. "
            "Returns the most relevant passages found. "
            "Use this when you need to find information from uploaded documents."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "classify_text",
        "description": (
            "Classify a piece of text into a category. "
            "Returns the predicted category and confidence score. "
            "Use this when you need to categorise or label text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to classify",
                }
            },
            "required": ["text"],
        },
    },
    {
        "name": "summarise_text",
        "description": (
            "Summarise a piece of text into a concise paragraph. "
            "Use this when you need to condense a long document or passage."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to summarise",
                }
            },
            "required": ["text"],
        },
    },
    {
        "name": "calculate",
        "description": (
            "Evaluate a mathematical expression and return the result. "
            "Supports standard arithmetic, math functions (sqrt, log, sin, cos, etc), "
            "and constants (pi, e). "
            "Use this for any numerical calculations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate, e.g. '2 ** 10' or 'sqrt(144)'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "finish",
        "description": (
            "Signal that the task is complete and return the final answer. "
            "Always call this tool when you have reached a conclusion. "
            "Do not call any other tools after calling finish."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The final answer or result to return to the user",
                }
            },
            "required": ["answer"],
        },
    },
]

# ── Tool executors ────────────────────────────────────────────────────────────

def execute_tool(name: str, inputs: dict) -> str:
    """Dispatch a tool call to the appropriate executor."""
    executors = {
        "search_docs": _search_docs,
        "classify_text": _classify_text,
        "summarise_text": _summarise_text,
        "calculate": _calculate,
        "finish": _finish,
    }
    executor = executors.get(name)
    if not executor:
        return f"Error: Unknown tool '{name}'"
    return executor(**inputs)


def _search_docs(query: str) -> str:
    """Query the doc-rag service."""
    try:
        response = httpx.post(
            f"{config.DOC_RAG_URL}/query",
            json={"query": query, "n_results": 3},
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if not results:
            return "No relevant documents found."
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"[{i}] {r.get('text', '').strip()}")
        return "\n\n".join(parts)
    except httpx.ConnectError:
        return "Error: doc-rag service is not available (is it running on port 8001?)"
    except Exception as e:
        return f"Error querying doc-rag: {e}"


def _classify_text(text: str) -> str:
    """Call the smart-api classify endpoint."""
    try:
        response = httpx.post(
            f"{config.SMART_API_URL}/classify",
            json={"text": text},
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()
        category = data.get("category", "unknown")
        confidence = data.get("confidence", 0.0)
        return f"Category: {category} (confidence: {confidence:.2f})"
    except httpx.ConnectError:
        return "Error: smart-api service is not available (is it running on port 8002?)"
    except Exception as e:
        return f"Error calling classify: {e}"


def _summarise_text(text: str) -> str:
    """Call the smart-api summarise endpoint."""
    try:
        response = httpx.post(
            f"{config.SMART_API_URL}/summarise",
            json={"text": text},
            timeout=15.0,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("summary", "No summary returned.")
    except httpx.ConnectError:
        return "Error: smart-api service is not available (is it running on port 8002?)"
    except Exception as e:
        return f"Error calling summarise: {e}"


def _calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    # Allowlist of safe names
    safe_names = {
        k: v for k, v in math.__dict__.items()
        if not k.startswith("_")
    }
    safe_names["abs"] = abs
    safe_names["round"] = round
    safe_names["min"] = min
    safe_names["max"] = max

    try:
        result = eval(expression, {"__builtins__": {}}, safe_names)  # noqa: S307
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error evaluating expression '{expression}': {e}"


def _finish(answer: str) -> str:
    """Signal completion — the agent loop checks for this."""
    return answer
