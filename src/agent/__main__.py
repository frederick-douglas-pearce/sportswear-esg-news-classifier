#!/usr/bin/env python3
"""CLI entry point for the agent orchestrator.

Usage:
    # Run a workflow
    uv run python -m src.agent run daily_labeling
    uv run python -m src.agent run website_export --dry-run

    # Check workflow status
    uv run python -m src.agent status

    # Resume a paused workflow
    uv run python -m src.agent continue model_training

    # List available workflows
    uv run python -m src.agent list
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from .config import agent_settings
from .state import state_manager
from .workflows import WorkflowRegistry

# Import workflow modules to register them
from .workflows import daily_labeling, drift_monitoring, website_export  # noqa: F401


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the agent."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def cmd_run(args: argparse.Namespace) -> int:
    """Run a workflow."""
    workflow_name = args.workflow

    try:
        workflow = WorkflowRegistry.create(
            name=workflow_name,
            dry_run=args.dry_run,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    print(f"\n{'=' * 60}")
    print(f"RUNNING WORKFLOW: {workflow_name}")
    print(f"Dry Run: {args.dry_run}")
    print(f"{'=' * 60}\n")

    try:
        result = workflow.run()
        print(f"\nWorkflow finished with status: {result.status.value}")

        if result.status.value == "completed":
            return 0
        elif result.status.value == "paused":
            print(f"Workflow paused: {result.context.get('pause_reason', 'unknown')}")
            print(f"Resume with: uv run python -m src.agent continue {workflow_name}")
            return 0
        else:
            print(f"Error: {result.error}")
            return 1

    except Exception as e:
        print(f"Workflow failed with exception: {e}")
        return 1


def cmd_continue(args: argparse.Namespace) -> int:
    """Resume a paused workflow."""
    workflow_name = args.workflow

    try:
        workflow = WorkflowRegistry.create(
            name=workflow_name,
            dry_run=args.dry_run,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    print(f"\n{'=' * 60}")
    print(f"RESUMING WORKFLOW: {workflow_name}")
    print(f"{'=' * 60}\n")

    try:
        result = workflow.resume()
        print(f"\nWorkflow finished with status: {result.status.value}")

        if result.status.value == "completed":
            return 0
        elif result.status.value == "paused":
            print(f"Workflow paused again: {result.context.get('pause_reason', 'unknown')}")
            return 0
        else:
            print(f"Error: {result.error}")
            return 1

    except Exception as e:
        print(f"Workflow resume failed: {e}")
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Show status of all workflows."""
    status = state_manager.get_status()

    print(f"\n{'=' * 60}")
    print("WORKFLOW STATUS")
    print(f"{'=' * 60}")
    print(f"State file: {agent_settings.state_file}")
    print()

    if not status:
        print("No active workflows.")
        return 0

    for name, wf_status in status.items():
        print(f"Workflow: {name}")
        print(f"  Status: {wf_status['status']}")
        if wf_status.get('current_step'):
            print(f"  Current Step: {wf_status['current_step']}")
        if wf_status.get('started_at'):
            print(f"  Started: {wf_status['started_at']}")
        print()

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List available workflows."""
    workflows = WorkflowRegistry.list()

    print(f"\n{'=' * 60}")
    print("AVAILABLE WORKFLOWS")
    print(f"{'=' * 60}\n")

    for name in workflows:
        workflow_class = WorkflowRegistry.get(name)
        if workflow_class:
            print(f"  {name}")
            print(f"    {workflow_class.description}")
            print(f"    Steps: {len(workflow_class.steps)}")
            print()

    print("Run with: uv run python -m src.agent run <workflow>")
    print("Dry run:  uv run python -m src.agent run <workflow> --dry-run")
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    """Show workflow execution history."""
    history_dir = agent_settings.history_dir

    if not history_dir.exists():
        print("No workflow history found.")
        return 0

    print(f"\n{'=' * 60}")
    print("WORKFLOW HISTORY")
    print(f"{'=' * 60}")
    print(f"History directory: {history_dir}")
    print()

    # List history files
    history_files = sorted(history_dir.glob("*.yaml"), reverse=True)

    if not history_files:
        print("No history files found.")
        return 0

    # Show last N entries
    limit = args.limit if hasattr(args, 'limit') else 10
    for history_file in history_files[:limit]:
        print(f"  {history_file.name}")

    if len(history_files) > limit:
        print(f"\n  ... and {len(history_files) - limit} more")

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="agent",
        description="ESG News Classifier Agent Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # run command
    run_parser = subparsers.add_parser("run", help="Run a workflow")
    run_parser.add_argument("workflow", help="Workflow name")
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without making changes",
    )
    run_parser.set_defaults(func=cmd_run)

    # continue command
    continue_parser = subparsers.add_parser("continue", help="Resume a paused workflow")
    continue_parser.add_argument("workflow", help="Workflow name")
    continue_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Continue in dry-run mode",
    )
    continue_parser.set_defaults(func=cmd_continue)

    # status command
    status_parser = subparsers.add_parser("status", help="Show workflow status")
    status_parser.set_defaults(func=cmd_status)

    # list command
    list_parser = subparsers.add_parser("list", help="List available workflows")
    list_parser.set_defaults(func=cmd_list)

    # history command
    history_parser = subparsers.add_parser("history", help="Show workflow history")
    history_parser.add_argument(
        "-n", "--limit",
        type=int,
        default=10,
        help="Number of entries to show (default: 10)",
    )
    history_parser.set_defaults(func=cmd_history)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    setup_logging(args.verbose)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
