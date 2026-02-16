"""CLI entry point for workflow learning module.

Usage:
    # Start a recording session
    uv run python -m src.workflow_learning start "workflow-name" [--description "..."]

    # Stop the current session
    uv run python -m src.workflow_learning stop

    # List all sessions
    uv run python -m src.workflow_learning list

    # Analyze a session and generate skill
    uv run python -m src.workflow_learning analyze <session-id> [--skill-name "..."]

    # Show session details
    uv run python -m src.workflow_learning show <session-id>
"""

import argparse
import logging
import sys
from datetime import datetime, timezone

from .analyzer import RecordingAnalyzer
from .models import SessionStatus
from .screenpipe_client import ScreenpipeClient, ScreenpipeError
from .session_manager import SessionManager
from .skill_generator import SkillGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_start(args: argparse.Namespace) -> int:
    """Start a new recording session."""
    manager = SessionManager()

    # Check Screenpipe health
    client = ScreenpipeClient()
    if not client.health_check():
        print("ERROR: Screenpipe is not running or not accessible.")
        print("Start Screenpipe first: screenpipe")
        print(f"Expected API at: {client.api_url}")
        return 1

    try:
        session = manager.start_session(
            workflow_name=args.name,
            description=args.description or "",
        )
        print(f"Started recording session: {session.session_id}")
        print(f"Workflow: {session.workflow_name}")
        print(f"Started at: {session.started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print()
        print("Screenpipe is now recording your screen and audio.")
        print("Demonstrate your workflow and narrate what you're doing.")
        print()
        print("When finished, run:")
        print("  uv run python -m src.workflow_learning stop")
        return 0
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1


def cmd_stop(args: argparse.Namespace) -> int:
    """Stop the current recording session."""
    manager = SessionManager()
    client = ScreenpipeClient()

    try:
        session = manager.stop_session()
        print(f"Stopped recording session: {session.session_id}")
        print(f"Duration: {_format_duration(session.started_at, session.stopped_at)}")
        print()

        # Retrieve content from Screenpipe
        print("Retrieving recorded content from Screenpipe...")
        try:
            screen_content = client.get_screen_content(
                start_time=session.started_at,
                end_time=session.stopped_at,
            )
            audio_transcripts = client.get_audio_transcriptions(
                start_time=session.started_at,
                end_time=session.stopped_at,
            )

            session.screen_content = screen_content
            session.audio_transcripts = audio_transcripts
            manager.update_session(session)

            print(f"  - Screen frames: {len(screen_content)}")
            print(f"  - Audio chunks: {len(audio_transcripts)}")
            print()

        except ScreenpipeError as e:
            print(f"WARNING: Failed to retrieve content: {e}")
            print("You may need to analyze manually after retrieving content.")
            print()

        print("To analyze and generate a skill, run:")
        print(f"  uv run python -m src.workflow_learning analyze {session.session_id}")
        return 0

    except ValueError as e:
        print(f"ERROR: {e}")
        return 1


def cmd_list(args: argparse.Namespace) -> int:
    """List all recording sessions."""
    manager = SessionManager()
    sessions = manager.list_sessions()

    if not sessions:
        print("No recording sessions found.")
        return 0

    print(f"{'Session ID':<45} {'Status':<15} {'Started':<20} {'Duration':<10}")
    print("-" * 90)

    for session in sessions:
        duration = ""
        if session.stopped_at:
            duration = _format_duration(session.started_at, session.stopped_at)

        print(
            f"{session.session_id:<45} "
            f"{session.status.value:<15} "
            f"{session.started_at.strftime('%Y-%m-%d %H:%M'):<20} "
            f"{duration:<10}"
        )

    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Show details for a specific session."""
    manager = SessionManager()
    session = manager.get_session(args.session_id)

    if not session:
        print(f"ERROR: Session '{args.session_id}' not found.")
        return 1

    print(f"Session ID: {session.session_id}")
    print(f"Workflow: {session.workflow_name}")
    print(f"Description: {session.description or '(none)'}")
    print(f"Status: {session.status.value}")
    print(f"Started: {session.started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    if session.stopped_at:
        print(f"Stopped: {session.stopped_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Duration: {_format_duration(session.started_at, session.stopped_at)}")

    print()
    print("Content:")
    print(f"  - Screen frames: {len(session.screen_content)}")
    print(f"  - Audio chunks: {len(session.audio_transcripts)}")
    print(f"  - Extracted steps: {len(session.extracted_steps)}")

    if session.skill_path:
        print(f"  - Skill file: {session.skill_path}")

    if session.error:
        print()
        print(f"Error: {session.error}")

    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze a session and generate a skill."""
    manager = SessionManager()
    session = manager.get_session(args.session_id)

    if not session:
        print(f"ERROR: Session '{args.session_id}' not found.")
        return 1

    if session.status == SessionStatus.RECORDING:
        print("ERROR: Session is still recording. Stop it first.")
        return 1

    # Check if we have content
    if not session.screen_content and not session.audio_transcripts:
        print("WARNING: No recorded content found in session.")
        print("Attempting to retrieve from Screenpipe...")

        if not session.stopped_at:
            print("ERROR: Session has no stop time. Cannot retrieve content.")
            return 1

        client = ScreenpipeClient()
        try:
            session.screen_content = client.get_screen_content(
                start_time=session.started_at,
                end_time=session.stopped_at,
            )
            session.audio_transcripts = client.get_audio_transcriptions(
                start_time=session.started_at,
                end_time=session.stopped_at,
            )
            manager.update_session(session)
            print(f"  - Retrieved {len(session.screen_content)} screen frames")
            print(f"  - Retrieved {len(session.audio_transcripts)} audio chunks")
        except ScreenpipeError as e:
            print(f"ERROR: Failed to retrieve content: {e}")
            return 1

    # Analyze with Claude
    print()
    print("Analyzing recording with Claude...")
    try:
        analyzer = RecordingAnalyzer()
        result = analyzer.analyze_session(session)

        if not result.success:
            print(f"ERROR: Analysis failed: {result.error}")
            session.status = SessionStatus.FAILED
            session.error = result.error
            manager.update_session(session)
            return 1

        print(f"  - Extracted {len(result.steps)} workflow steps")
        print(f"  - Tokens used: {result.input_tokens} in, {result.output_tokens} out")

        # Update session with analysis
        session.extracted_steps = result.steps
        session.analysis_summary = result.summary
        session.status = SessionStatus.ANALYZED
        session.analyzed_at = datetime.now(timezone.utc)
        manager.update_session(session)

    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    # Generate skill
    print()
    print("Generating skill...")
    generator = SkillGenerator()
    skill_path = generator.generate_skill(
        session=session,
        analysis=result,
        skill_name=args.skill_name,
    )

    session.skill_path = str(skill_path)
    session.status = SessionStatus.SKILL_GENERATED
    session.skill_generated_at = datetime.now(timezone.utc)
    manager.update_session(session)

    print(f"  - Skill generated: {skill_path}")
    print()
    print("Done! Review the generated skill at:")
    print(f"  {skill_path}")

    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    """Delete a session."""
    manager = SessionManager()

    if not manager.delete_session(args.session_id):
        print(f"ERROR: Session '{args.session_id}' not found.")
        return 1

    print(f"Deleted session: {args.session_id}")
    return 0


def _format_duration(start: datetime, end: datetime | None) -> str:
    """Format duration between two timestamps."""
    if not end:
        return ""
    delta = end - start
    minutes = int(delta.total_seconds() / 60)
    seconds = int(delta.total_seconds() % 60)
    return f"{minutes}m {seconds}s"


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="workflow_learning",
        description="Record and analyze workflows to generate Agent Skills",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # start command
    start_parser = subparsers.add_parser("start", help="Start a recording session")
    start_parser.add_argument("name", help="Workflow name (e.g., 'model-training-fp')")
    start_parser.add_argument(
        "--description", "-d", help="Description of the workflow"
    )

    # stop command
    subparsers.add_parser("stop", help="Stop the current recording session")

    # list command
    subparsers.add_parser("list", help="List all recording sessions")

    # show command
    show_parser = subparsers.add_parser("show", help="Show session details")
    show_parser.add_argument("session_id", help="Session ID to show")

    # analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze session and generate skill"
    )
    analyze_parser.add_argument("session_id", help="Session ID to analyze")
    analyze_parser.add_argument(
        "--skill-name", "-n", help="Override skill name (default: from analysis)"
    )

    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a session")
    delete_parser.add_argument("session_id", help="Session ID to delete")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "start": cmd_start,
        "stop": cmd_stop,
        "list": cmd_list,
        "show": cmd_show,
        "analyze": cmd_analyze,
        "delete": cmd_delete,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
