# task-agent — Project Context

## What this is
A ReAct (Reason + Act) pattern AI agent built on the Anthropic API. Given a goal, the agent reasons step by step, calls tools, observes results, and loops until it reaches a conclusion.

## Stack
- Python 3.12
- Anthropic API (`claude-sonnet-4-6`) — no date suffix on model string
- FastAPI + uvicorn
- httpx for HTTP tool calls
- pytest for testing

## Project structure
```
task-agent/
├── main.py          — CLI entry point, prints readable trace
├── api.py           — FastAPI REST interface (POST /run, GET /health)
├── agent.py         — ReAct loop (Think → Act → Observe)
├── tools.py         — Tool schemas and executors
├── config.py        — Environment variable config
├── conftest.py      — Adds project root to sys.path for pytest
├── requirements.txt      — Runtime dependencies
├── requirements-dev.txt  — Adds pytest, pytest-asyncio, pytest-cov
└── tests/
    ├── test_tools.py  — Unit tests for all tools (HTTP calls mocked)
    └── test_agent.py  — Unit tests for ReAct loop (Anthropic API mocked)
```

## Key design decisions
- `tool_choice={"type": "any"}` forces the model to always call a tool — prevents plain text responses that bypass finish()
- Tools degrade gracefully — external service unavailable returns informative error, agent adapts
- Full reasoning trace returned alongside final answer
- `finish()` tool signals task completion — agent loop checks for this specifically
- Anthropic API is mocked in all tests — no real API calls during CI

## External service dependencies
- `doc-rag` on port 8001 — provides `search_docs` tool backend
- `smart-api` on port 8002 — provides `classify_text` and `summarise_text` tool backends
- Both are optional — tools return friendly errors if unavailable

## Running locally
```bash
source venv/bin/activate
python main.py "What is the square root of 1764?"
uvicorn api:app --reload --port 8080
pytest tests/ -v
```

## CI/CD
- Uses shared reusable workflow from `danieldrysdale/.github`
- Triggers on push/PR to main
- Builds multi-platform Docker image (amd64 + arm64) to GHCR
- Image: `ghcr.io/danieldrysdale/task-agent:latest`

## Environment variables
- `ANTHROPIC_API_KEY` — required
- `ANTHROPIC_MODEL` — default: `claude-sonnet-4-6`
- `MAX_ITERATIONS` — default: 10
- `DOC_RAG_URL` — default: `http://localhost:8001`
- `SMART_API_URL` — default: `http://localhost:8002`

## Conventions
- Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `ci:`)
- All external HTTP calls in tools.py, never in agent.py
- Agent loop logic in agent.py only — no tool execution there
