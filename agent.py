"""
ReAct agent loop.

The agent runs a Think → Act → Observe cycle:
  1. Send the goal (and any previous observations) to the model
  2. Model responds with either a tool call or a finish signal
  3. Execute the tool, record the observation
  4. Repeat until finish() is called or MAX_ITERATIONS is reached

The full reasoning trace is returned alongside the final answer,
making the agent's decision-making process transparent.
"""

from dataclasses import dataclass, field
from typing import Optional

import anthropic

import config
from tools import TOOL_SCHEMAS, execute_tool

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Step:
    """A single step in the agent's reasoning trace."""
    iteration: int
    thought: Optional[str]       # Model's reasoning (text before tool call)
    tool_name: Optional[str]     # Tool called (None if finishing with text)
    tool_input: Optional[dict]   # Tool inputs
    observation: Optional[str]   # Tool output


@dataclass
class AgentResult:
    """The complete result of an agent run."""
    goal: str
    answer: str
    steps: list[Step] = field(default_factory=list)
    iterations: int = 0
    success: bool = True
    error: Optional[str] = None


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a task-solving agent. You have access to a set of tools \
to help you complete the user's goal.

For each step:
1. Think about what you need to do next
2. Call the most appropriate tool
3. Observe the result and decide on the next step
4. When you have enough information, call the finish tool with your final answer

Rules:
- Always call finish() when you have reached a conclusion — never stop without it
- Use search_docs to find information from documents before answering questions about documents
- Use calculate for any mathematical operations rather than doing them yourself
- Be concise and direct in your reasoning
- If a tool returns an error, try a different approach or explain the limitation in your final answer
"""

# ── Agent loop ────────────────────────────────────────────────────────────────

def run(goal: str) -> AgentResult:
    """
    Run the ReAct agent loop for a given goal.
    Returns an AgentResult with the answer and full reasoning trace.
    """
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": goal}]
    steps: list[Step] = []
    final_answer: Optional[str] = None

    for iteration in range(1, config.MAX_ITERATIONS + 1):

        # ── Call the model ────────────────────────────────────────────────────
        response = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=config.ANTHROPIC_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            tool_choice={"type": "any"},
            messages=messages,
        )

        # ── Extract thought and tool use from response ────────────────────────
        thought = None
        tool_use_block = None

        for block in response.content:
            if block.type == "text" and block.text.strip():
                thought = block.text.strip()
            elif block.type == "tool_use":
                tool_use_block = block

        # ── No tool call — model gave a plain text response ──────────────────
        if tool_use_block is None:
            # Treat the text response as the final answer
            final_answer = thought or "No answer produced."
            steps.append(Step(
                iteration=iteration,
                thought=thought,
                tool_name=None,
                tool_input=None,
                observation=None,
            ))
            break

        tool_name = tool_use_block.name
        tool_input = tool_use_block.input

        # ── Execute the tool ──────────────────────────────────────────────────
        observation = execute_tool(tool_name, tool_input)

        steps.append(Step(
            iteration=iteration,
            thought=thought,
            tool_name=tool_name,
            tool_input=tool_input,
            observation=observation,
        ))

        # ── finish() signals completion ───────────────────────────────────────
        if tool_name == "finish":
            final_answer = tool_input.get("answer", observation)
            break

        # ── Feed the tool result back to the model ────────────────────────────
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_block.id,
                    "content": observation,
                }
            ],
        })

    else:
        # Max iterations reached without finish()
        final_answer = "Max iterations reached without a final answer."
        return AgentResult(
            goal=goal,
            answer=final_answer,
            steps=steps,
            iterations=len(steps),
            success=False,
            error="Max iterations reached",
        )

    return AgentResult(
        goal=goal,
        answer=final_answer or "",
        steps=steps,
        iterations=len(steps),
        success=True,
    )
