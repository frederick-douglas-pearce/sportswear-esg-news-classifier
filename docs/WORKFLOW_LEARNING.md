# Workflow Learning

Record user workflows via Screenpipe and generate replayable Agent Skills using Claude analysis.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
  - [System Dependencies](#system-dependencies)
  - [Bun (JavaScript Runtime)](#bun-javascript-runtime)
  - [Screenpipe](#screenpipe)
  - [API Keys](#api-keys)
- [Getting Started](#getting-started)
  - [1. Install Dependencies](#1-install-dependencies)
  - [2. Start Screenpipe](#2-start-screenpipe)
  - [3. Verify Screenpipe](#3-verify-screenpipe)
- [Usage](#usage)
  - [Recording a Workflow](#recording-a-workflow)
  - [Analyzing and Generating a Skill](#analyzing-and-generating-a-skill)
  - [Managing Sessions](#managing-sessions)
- [Example: FP Classifier Training](#example-fp-classifier-training)
- [Screenpipe Details](#screenpipe-details)
  - [Default Settings](#default-settings)
  - [Data Storage](#data-storage)
  - [Cleanup](#cleanup)
- [Tips](#tips)
- [Configuration](#configuration)
- [Limitations](#limitations)

## Overview

The workflow learning module records your screen and audio while you demonstrate a workflow, then uses Claude to analyze the recording and generate a SKILL.md file that can be replayed as an Agent Skill.

**Flow:** Start Screenpipe → Start session → Demonstrate workflow (narrate aloud) → Stop session → Analyze → Review generated skill

## Prerequisites

### System Dependencies

Install the required system packages:

```bash
sudo apt install tesseract-ocr libasound2-dev libpulse0 ffmpeg
```

| Package | Purpose |
|---------|---------|
| `tesseract-ocr` | OCR engine for reading text from screen captures |
| `libasound2-dev` | ALSA audio development libraries |
| `libpulse0` | PulseAudio client libraries |
| `ffmpeg` | Video/audio encoding for screen recordings |

### Bun (JavaScript Runtime)

Screenpipe is distributed via `bunx`, which requires the Bun runtime:

```bash
curl -fsSL https://bun.sh/install | bash
source ~/.bashrc  # or open a new terminal
```

Verify installation:

```bash
bun --version
```

### Screenpipe

Screenpipe is installed and run via `bunx` (no separate install step). The first time you run it, it will download required models automatically (~870 MB total):

- Whisper large-v3-turbo (speech-to-text): ~834 MB
- Speaker identification models: ~35 MB
- Silero VAD (voice activity detection): ~2 MB

### API Keys

- `ANTHROPIC_API_KEY` must be set in your `.env` file (used by Claude for the analysis step)

## Getting Started

### 1. Install Dependencies

```bash
# System packages
sudo apt install tesseract-ocr libasound2-dev libpulse0 ffmpeg

# Bun runtime
curl -fsSL https://bun.sh/install | bash
source ~/.bashrc
```

### 2. Start Screenpipe

In a dedicated terminal (it runs in the foreground):

```bash
bunx screenpipe record
```

On first run, Screenpipe will download models. Wait until you see the settings table and `Server listening on 0.0.0.0:3030` in the output.

### 3. Verify Screenpipe

In a separate terminal:

```bash
curl http://localhost:3030/health
```

A successful response confirms Screenpipe is ready.

## Usage

### Recording a Workflow

```bash
# Start a recording session
uv run python -m src.workflow_learning start "workflow-name" -d "description of what you'll demonstrate"

# ... demonstrate the workflow while narrating what you're doing ...

# Stop the recording session
uv run python -m src.workflow_learning stop
```

### Analyzing and Generating a Skill

```bash
# List sessions to find the session ID
uv run python -m src.workflow_learning list

# Analyze the recording and generate a skill
uv run python -m src.workflow_learning analyze <session-id>

# Optionally override the skill name
uv run python -m src.workflow_learning analyze <session-id> --skill-name "custom-name"
```

The generated skill is saved to `.claude/skills/learned/<workflow-name>/SKILL.md`.

### Managing Sessions

```bash
# List all sessions
uv run python -m src.workflow_learning list

# Show details for a specific session
uv run python -m src.workflow_learning show <session-id>

# Delete a session
uv run python -m src.workflow_learning delete <session-id>
```

## Example: FP Classifier Training

The intended MVP use case is recording the FP classifier model training workflow:

```bash
# 1. Start Screenpipe in a dedicated terminal
bunx screenpipe record

# 2. Start recording
uv run python -m src.workflow_learning start "model-training-fp" \
  -d "FP classifier model training: export data, run notebooks, evaluate, deploy"

# 3. Demonstrate the workflow (narrate each step aloud):
#    a. Export training data
uv run python scripts/export_training_data.py --dataset fp
#    b. Open and run FP notebooks (fp1, fp2, fp3)
#    c. Train the model
uv run python scripts/train.py --classifier fp
#    d. Register/deploy
uv run python scripts/register_model.py --classifier fp --bump minor

# 4. Stop recording
uv run python -m src.workflow_learning stop

# 5. Analyze and generate skill
uv run python -m src.workflow_learning analyze <session-id> --skill-name "model-training-fp"

# 6. Review the generated skill
cat .claude/skills/learned/model-training-fp/SKILL.md

# 7. Stop Screenpipe (Ctrl+C in its terminal)
```

**Note:** The notebooks take a long time to run. Since Screenpipe captures at 1 FPS continuously, long idle periods generate many frames. For a trial run, consider recording in short segments (e.g., just the export + notebook launch step) to validate the pipeline before committing to a full recording.

## Screenpipe Details

### Default Settings

| Setting | Default |
|---------|---------|
| Screen capture FPS | 1.0 (0.5 on macOS) |
| Audio chunk duration | 30 seconds |
| Video chunk duration | 60 seconds |
| Audio sample rate | 48000 Hz |
| API port | 3030 |
| OCR engine | Tesseract |
| Speech-to-text engine | Whisper Large V3 Turbo (quantized) |

### Data Storage

Screenpipe stores all data locally in `~/.screenpipe/`:

```
~/.screenpipe/
├── data/          # Screen frames and video chunks
├── db.sqlite      # SQLite database (OCR text, audio transcripts, metadata)
└── pipes/         # Built-in pipes (obsidian-sync, idea-tracker)
```

Estimated storage: ~5-10 GB per month of continuous recording.

### Cleanup

Screenpipe does not have built-in data retention or auto-cleanup. To manage disk space:

- **Best practice for trial runs:** Only run Screenpipe while actively recording. Start it before your session, Ctrl+C it when done.
- **Full reset:** Remove all Screenpipe data between runs:
  ```bash
  rm -rf ~/.screenpipe/data ~/.screenpipe/db.sqlite
  ```
- **Selective cleanup:** Not currently supported via CLI. Would require querying the SQLite database directly.

## Tips

- **Keep recordings short** for trial runs — record 1-2 steps to validate the pipeline
- **Narrate clearly** — audio transcription helps Claude understand the *why* behind each step
- **Stop Screenpipe when not recording** to save disk space and CPU
- **Check Screenpipe health** before starting a session: `curl http://localhost:3030/health`
- If you have multiple monitors, Screenpipe captures all of them by default (`--use-all-monitors`)

## Configuration

Environment variables (set in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `SCREENPIPE_API_URL` | `http://localhost:3030` | Screenpipe REST API URL |
| `WORKFLOW_RECORDING_DIR` | `data/workflow_recordings` | Session state storage |
| `WORKFLOW_SKILLS_DIR` | `.claude/skills/learned` | Generated skills output |
| `WORKFLOW_ANALYSIS_MODEL` | `claude-sonnet-4-20250514` | Model for Claude analysis |
| `WORKFLOW_MAX_SCREEN_FRAMES` | `1000` | Max screen frames to retrieve per session |
| `WORKFLOW_MAX_AUDIO_CHUNKS` | `500` | Max audio chunks to retrieve per session |

## Limitations

- **No pause/resume:** Each recording is a single continuous session. For long workflows (e.g., notebook training), consider recording in segments.
- **No multi-session analysis:** Each recording creates a new session. Combining multiple sessions into a single skill is not yet supported.
- **Overwrites on re-analyze:** Running `analyze` with the same skill name overwrites the previous skill file.
- **No selective Screenpipe cleanup:** Cannot delete recordings for a specific time window via CLI.
- **Future planned:** Pause/resume support, iterative skill refinement, Screenpipe data trimming to session windows only.
