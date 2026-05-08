"""
CLI entry point for task-agent.

Usage:
    python main.py "What is the square root of 144 multiplied by pi?"
    python main.py "Search my documents for information about neural networks and summarise what you find"
"""

import sys
import json
import agent


def print_trace(result: agent.AgentResult) -> None:
    """Print the agent's reasoning trace in a readable format."""
    print(f"\n{'='*60}")
    print(f"GOAL: {result.goal}")
    print(f"{'='*60}\n")

    for step in result.steps:
        print(f"── Step {step.iteration} {'─'*40}")

        if step.thought:
            print(f"\nThought:\n  {step.thought}\n")

        if step.tool_name:
            print(f"Action: {step.tool_name}")
            if step.tool_input:
                for k, v in step.tool_input.items():
                    val = str(v)
                    if len(val) > 120:
                        val = val[:120] + "..."
                    print(f"  {k}: {val}")

        if step.observation and step.tool_name != "finish":
            obs = step.observation
            if len(obs) > 300:
                obs = obs[:300] + "..."
            print(f"\nObservation:\n  {obs}\n")

    print(f"\n{'='*60}")
    print(f"ANSWER ({result.iterations} step{'s' if result.iterations != 1 else ''}):")
    print(f"\n{result.answer}")
    print(f"{'='*60}\n")

    if not result.success:
        print(f"Warning: {result.error}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py \"<your goal here>\"")
        print("\nExamples:")
        print("  python main.py \"What is 2 to the power of 32?\"")
        print("  python main.py \"Search my documents for neural networks and summarise\"")
        sys.exit(1)

    goal = " ".join(sys.argv[1:])
    result = agent.run(goal)
    print_trace(result)


if __name__ == "__main__":
    main()
