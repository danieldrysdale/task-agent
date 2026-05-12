"""
Unit tests for task-agent tools.

External HTTP tools (search_docs, classify_text, summarise_text) are tested
with mocked HTTP responses — no external services required.

The calculate tool is tested directly since it runs entirely locally.
"""

import pytest
from unittest.mock import patch, MagicMock
import httpx

from tools import execute_tool, _calculate, _search_docs, _classify_text, _summarise_text


# ── calculate ─────────────────────────────────────────────────────────────────

class TestCalculate:

    def test_basic_arithmetic(self):
        assert _calculate("2 + 2") == "4"

    def test_multiplication(self):
        assert _calculate("6 * 7") == "42"

    def test_power(self):
        assert _calculate("2 ** 10") == "1024"

    def test_sqrt(self):
        result = _calculate("sqrt(144)")
        assert result == "12.0"

    def test_pi(self):
        result = _calculate("round(pi, 4)")
        assert result == "3.1416"

    def test_division(self):
        result = _calculate("10 / 4")
        assert result == "2.5"

    def test_integer_division(self):
        result = _calculate("10 // 4")
        assert result == "2"

    def test_modulo(self):
        result = _calculate("17 % 5")
        assert result == "2"

    def test_zero_division(self):
        result = _calculate("1 / 0")
        assert "Error" in result
        assert "zero" in result.lower()

    def test_invalid_expression(self):
        result = _calculate("not_a_function(42)")
        assert "Error" in result

    def test_complex_expression(self):
        result = _calculate("round(sqrt(2) * pi, 4)")
        assert result == "4.4429"

    def test_abs(self):
        assert _calculate("abs(-42)") == "42"

    def test_min_max(self):
        assert _calculate("min(3, 1, 4, 1, 5)") == "1"
        assert _calculate("max(3, 1, 4, 1, 5)") == "5"

    def test_no_builtins_injection(self):
        """Security: __builtins__ should not be accessible."""
        result = _calculate("__import__('os').system('echo hack')")
        assert "Error" in result


# ── search_docs ───────────────────────────────────────────────────────────────

class TestSearchDocs:

    @patch("tools.httpx.post")
    def test_successful_search(self, mock_post):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "found_answer": True,
            "answer": "The refund policy allows returns within 30 days.",
            "sources": [
                {"text": "Products are eligible for a full refund within 30 days.", "source": "refund_policy.md", "page": None, "score": 0.38}
            ]
        }
        mock_post.return_value = mock_response

        result = _search_docs("refund policy")
        assert "refund" in result.lower()
        assert "30 days" in result

    @patch("tools.httpx.post")
    def test_no_results_found(self, mock_post):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "found_answer": False,
            "answer": "",
            "sources": []
        }
        mock_post.return_value = mock_response

        result = _search_docs("something obscure")
        assert "No relevant documents found" in result

    @patch("tools.httpx.post")
    def test_service_unavailable(self, mock_post):
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        result = _search_docs("anything")
        assert "Error" in result
        assert "not available" in result


# ── classify_text ─────────────────────────────────────────────────────────────

class TestClassifyText:

    @patch("tools.httpx.post")
    def test_successful_classification(self, mock_post):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "category": "billing",
            "confidence": 0.92
        }
        mock_post.return_value = mock_response

        result = _classify_text("I need a refund for my order")
        assert "billing" in result
        assert "0.92" in result

    @patch("tools.httpx.post")
    def test_service_unavailable(self, mock_post):
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        result = _classify_text("some text")
        assert "Error" in result
        assert "not available" in result


# ── summarise_text ────────────────────────────────────────────────────────────

class TestSummariseText:

    @patch("tools.httpx.post")
    def test_successful_summarise(self, mock_post):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "summary": "The document describes a 30-day refund policy."
        }
        mock_post.return_value = mock_response

        result = _summarise_text("A very long document about refunds...")
        assert "30-day refund policy" in result

    @patch("tools.httpx.post")
    def test_service_unavailable(self, mock_post):
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        result = _summarise_text("some text")
        assert "Error" in result
        assert "not available" in result


# ── execute_tool dispatcher ───────────────────────────────────────────────────

class TestExecuteTool:

    def test_unknown_tool(self):
        result = execute_tool("nonexistent_tool", {})
        assert "Unknown tool" in result

    def test_calculate_via_dispatcher(self):
        result = execute_tool("calculate", {"expression": "2 + 2"})
        assert result == "4"

    def test_finish_via_dispatcher(self):
        result = execute_tool("finish", {"answer": "The answer is 42."})
        assert result == "The answer is 42."
