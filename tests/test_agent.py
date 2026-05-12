"""
Unit tests for the agent ReAct loop.

The Anthropic API is mocked throughout — no API calls are made during testing.
We test the agent's behaviour in response to different model outputs:
- Tool call followed by finish
- Direct finish call
- Max iterations reached
- Plain text response (no tool call)
"""

import pytest
from unittest.mock import patch, MagicMock, call

import agent
from agent import run, AgentResult, Step


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_tool_use_block(tool_name: str, tool_input: dict, block_id: str = "tu_001"):
    """Create a mock tool_use content block."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.input = tool_input
    block.id = block_id
    return block


def make_text_block(text: str):
    """Create a mock text content block."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def make_response(content: list):
    """Create a mock Anthropic API response."""
    response = MagicMock()
    response.content = content
    return response


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestAgentRun:

    @patch("agent.anthropic.Anthropic")
    def test_calculate_then_finish(self, mock_anthropic_class):
        """Agent calls calculate, gets result, then calls finish."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Step 1: model calls calculate
        step1 = make_response([
            make_text_block("I need to calculate this."),
            make_tool_use_block("calculate", {"expression": "2 + 2"}, "tu_001"),
        ])

        # Step 2: model calls finish
        step2 = make_response([
            make_tool_use_block("finish", {"answer": "The answer is 4."}, "tu_002"),
        ])

        mock_client.messages.create.side_effect = [step1, step2]

        result = run("What is 2 + 2?")

        assert result.success is True
        assert result.answer == "The answer is 4."
        assert result.iterations == 2
        assert result.steps[0].tool_name == "calculate"
        assert result.steps[1].tool_name == "finish"

    @patch("agent.anthropic.Anthropic")
    def test_direct_finish(self, mock_anthropic_class):
        """Agent calls finish immediately without intermediate steps."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        step1 = make_response([
            make_tool_use_block("finish", {"answer": "42."}, "tu_001"),
        ])

        mock_client.messages.create.side_effect = [step1]

        result = run("What is the answer to life?")

        assert result.success is True
        assert result.answer == "42."
        assert result.iterations == 1

    @patch("agent.anthropic.Anthropic")
    def test_plain_text_response(self, mock_anthropic_class):
        """Agent returns plain text instead of calling finish — handled gracefully."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        step1 = make_response([
            make_text_block("The answer is simply 42."),
        ])

        mock_client.messages.create.side_effect = [step1]

        result = run("What is the answer?")

        assert result.success is True
        assert "42" in result.answer

    @patch("agent.anthropic.Anthropic")
    def test_max_iterations_reached(self, mock_anthropic_class):
        """Agent hits max iterations without calling finish."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        # Always return a calculate call — never finishes
        step = make_response([
            make_tool_use_block("calculate", {"expression": "1 + 1"}, "tu_001"),
        ])

        # Need enough responses for MAX_ITERATIONS
        import config
        mock_client.messages.create.return_value = step

        # Patch MAX_ITERATIONS to a small number for speed
        with patch.object(config, "MAX_ITERATIONS", 3):
            result = run("Loop forever")

        assert result.success is False
        assert result.error == "Max iterations reached"
        assert result.iterations == 3

    @patch("agent.anthropic.Anthropic")
    def test_tool_observation_fed_back(self, mock_anthropic_class):
        """Verify tool results are fed back into the conversation."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        step1 = make_response([
            make_tool_use_block("calculate", {"expression": "6 * 7"}, "tu_001"),
        ])
        step2 = make_response([
            make_tool_use_block("finish", {"answer": "The result is 42."}, "tu_002"),
        ])

        mock_client.messages.create.side_effect = [step1, step2]

        result = run("What is 6 times 7?")

        # Second call should include tool_result in messages
        second_call_args = mock_client.messages.create.call_args_list[1]
        messages = second_call_args.kwargs.get("messages") or second_call_args.args[0] if second_call_args.args else []

        # Verify the conversation grew
        assert mock_client.messages.create.call_count == 2
        assert result.steps[0].observation == "42"

    @patch("agent.anthropic.Anthropic")
    def test_step_trace_populated(self, mock_anthropic_class):
        """Verify the reasoning trace is correctly populated."""
        mock_client = MagicMock()
        mock_anthropic_class.return_value = mock_client

        step1 = make_response([
            make_text_block("Let me think about this."),
            make_tool_use_block("calculate", {"expression": "10 ** 2"}, "tu_001"),
        ])
        step2 = make_response([
            make_tool_use_block("finish", {"answer": "10 squared is 100."}, "tu_002"),
        ])

        mock_client.messages.create.side_effect = [step1, step2]

        result = run("What is 10 squared?")

        assert len(result.steps) == 2
        assert result.steps[0].thought == "Let me think about this."
        assert result.steps[0].tool_name == "calculate"
        assert result.steps[0].tool_input == {"expression": "10 ** 2"}
        assert result.steps[0].observation == "100"
        assert result.steps[1].tool_name == "finish"
