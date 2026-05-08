# task-agent

A ReAct (Reason + Act) pattern AI agent built on the Anthropic API. Given a goal, the agent reasons step by step, calls tools to gather information or perform actions, observes the results, and loops until it reaches a conclusion.

## What it demonstrates

The ReAct pattern — one of the foundational patterns in agentic AI — where the model:

1. **Thinks** — reasons about what it needs to do next
2. **Acts** — calls a tool to gather information or perform an action
3. **Observes** — reads the tool result
4. **Repeats** — until the goal is achieved

The full reasoning trace is returned alongside the final answer, making the agent's decision-making process transparent and inspectable.

```
Goal: "What is the square root of 1764, and is it a prime number?"

── Step 1 ──────────────────────────────────
Thought: I need to calculate the square root of 1764 first.
Action: calculate
  expression: sqrt(1764)
Observation: 42.0

── Step 2 ──────────────────────────────────
Thought: The square root is 42. Now I need to check if 42 is prime.
Action: calculate
  expression: all(42 % i != 0 for i in range(2, int(42**0.5) + 1))
Observation: False

── Step 3 ──────────────────────────────────
Action: finish
  answer: The square root of 1764 is 42. It is not a prime number — 42 is divisible by 2, 3, 6, 7, 14, and 21.

ANSWER: The square root of 1764 is 42. It is not a prime number...
```

## Tools

| Tool | Description | Requires |
|---|---|---|
| `calculate` | Evaluate mathematical expressions | Nothing — runs locally |
| `search_docs` | Search a document knowledge base | doc-rag service on port 8001 |
| `classify_text` | Classify text into a category | smart-api service on port 8002 |
| `summarise_text` | Summarise a piece of text | smart-api service on port 8002 |
| `finish` | Signal task completion | Nothing |

Tools degrade gracefully — if an external service is unavailable, the tool returns an informative error and the agent adapts.

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

```bash
export ANTHROPIC_API_KEY=your_key_here
```

### 3. Run from the CLI

```bash
python main.py "What is 2 to the power of 32?"
python main.py "What is the circumference of a circle with radius 7?"
python main.py "Search my documents for information about machine learning"
```

### 4. Run as an API

```bash
uvicorn api:app --reload --port 8080
```

Then POST to `/run`:

```bash
curl -X POST http://localhost:8080/run \
  -H "Content-Type: application/json" \
  -d '{"goal": "What is the square root of 1764?"}'
```

## Configuration

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required. Your Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Model to use |
| `ANTHROPIC_MAX_TOKENS` | `4096` | Max tokens per response |
| `MAX_ITERATIONS` | `10` | Max agent steps before stopping |
| `DOC_RAG_URL` | `http://localhost:8001` | doc-rag service URL |
| `SMART_API_URL` | `http://localhost:8002` | smart-api service URL |
| `API_PORT` | `8080` | Port for the FastAPI server |

## Project structure

```
task-agent/
├── main.py          — CLI entry point
├── api.py           — FastAPI REST interface
├── agent.py         — ReAct loop
├── tools.py         — Tool definitions and executors
├── config.py        — Configuration
└── requirements.txt
```

## Architecture

```
User goal
    │
    ▼
┌─────────────────────────────────────┐
│           ReAct Loop (agent.py)     │
│                                     │
│  ┌─────────┐    ┌─────────────────┐ │
│  │  Think  │───▶│  Act (tool use) │ │
│  └─────────┘    └────────┬────────┘ │
│       ▲                  │          │
│       │         ┌────────▼────────┐ │
│       └─────────│    Observe      │ │
│                 └─────────────────┘ │
└─────────────────────────────────────┘
         │
         ▼
    Final answer + trace
```

## Wiring with other portfolio projects

This agent is designed to work with the other projects in this portfolio:

- **doc-rag** — provides the `search_docs` tool backend (semantic document search)
- **smart-api** — provides the `classify_text` and `summarise_text` tool backends

Run all three together for a fully connected agent system.
