"""
FastAPI REST interface for task-agent.

Endpoints:
    POST /run     — Run the agent with a goal, returns answer + trace
    GET  /health  — Health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

import agent
import config

app = FastAPI(
    title="task-agent",
    description="ReAct pattern AI agent with tool use",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ─────────────────────────────────────────────────

class RunRequest(BaseModel):
    goal: str = Field(..., description="The task or question for the agent to solve")


class StepResponse(BaseModel):
    iteration: int
    thought: Optional[str]
    tool_name: Optional[str]
    tool_input: Optional[dict]
    observation: Optional[str]


class RunResponse(BaseModel):
    goal: str
    answer: str
    iterations: int
    success: bool
    error: Optional[str]
    steps: list[StepResponse]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/run", response_model=RunResponse)
def run(request: RunRequest) -> RunResponse:
    """
    Run the ReAct agent with a goal.
    Returns the final answer and the full reasoning trace.
    """
    if not request.goal.strip():
        raise HTTPException(status_code=400, detail="Goal cannot be empty")

    try:
        result = agent.run(request.goal)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return RunResponse(
        goal=result.goal,
        answer=result.answer,
        iterations=result.iterations,
        success=result.success,
        error=result.error,
        steps=[
            StepResponse(
                iteration=s.iteration,
                thought=s.thought,
                tool_name=s.tool_name,
                tool_input=s.tool_input,
                observation=s.observation,
            )
            for s in result.steps
        ],
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": config.ANTHROPIC_MODEL,
        "max_iterations": config.MAX_ITERATIONS,
    }
