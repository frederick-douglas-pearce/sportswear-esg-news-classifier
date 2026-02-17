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
  - [Refining a Skill with Additional Recordings](#refining-a-skill-with-additional-recordings)
  - [Managing Sessions](#managing-sessions)
- [Notebook-Aware Skills](#notebook-aware-skills)
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

### Refining a Skill with Additional Recordings

Multiple recording sessions can contribute to a single skill using the `--refine` flag. This is useful for:

- Adding detail to an existing skill (e.g., a first session captures the overall flow, a second adds notebook-specific metrics and decision points)
- Filling in gaps discovered after reviewing the initial skill
- Updating a skill when the workflow changes

```bash
# 1. Record an additional session focused on specific details
uv run python -m src.workflow_learning start "fp-notebook-detail" \
  -d "Detailed walkthrough of FP notebook cells and expected metrics"

# ... demonstrate and narrate ...

uv run python -m src.workflow_learning stop

# 2. Refine the existing skill with the new recording
uv run python -m src.workflow_learning analyze <new-session-id> --refine "model-training-fp"
```

When refining, the analyzer:
- Preserves existing steps that the new recording doesn't contradict
- Adds detail to steps that the new recording elaborates on (e.g., specific metric thresholds, expected outputs)
- Inserts new steps if the recording reveals previously missed actions
- Updates `success_criteria` and `expected_output` with specifics from narration

The `--refine` flag requires an existing skill at `.claude/skills/learned/<skill-name>/SKILL.md`.

### Managing Sessions

```bash
# List all sessions
uv run python -m src.workflow_learning list

# Show details for a specific session
uv run python -m src.workflow_learning show <session-id>

# Delete a session
uv run python -m src.workflow_learning delete <session-id>
```

## Notebook-Aware Skills

When recordings involve Jupyter notebooks, the analyzer generates skills with notebook-specific step types and checkpoint logic.

### Step Types

Each workflow step has a `tool_type` that determines how it's formatted in the generated skill:

| Tool Type | Use Case | Rendered As |
|-----------|----------|-------------|
| `bash` | Shell commands (default) | ` ```bash ` code block |
| `jupyter` | Notebook cell execution | ` ```python ` code block with `mcp__ide__executeCode` reference |
| `review` | Output/figure inspection | Expected output description with success criteria |
| `manual` | Browser or GUI actions | Generic code block |

### Checkpoint Fields

Steps can include decision-making metadata for verifying results:

- **`expected_output`** — What the output should look like (e.g., "Confusion matrix with diagonal dominance")
- **`success_criteria`** — How to verify success (e.g., "F2 score > 0.95, Recall > 98%")
- **`on_failure`** — What to do if criteria aren't met (e.g., "Adjust class weights and rerun from Step 3")

These fields are populated when the user narrates what to look for during the recording. They enable the generated skill to guide an agent through iterative notebook workflows where results need verification before proceeding.

### Example Generated Step

A jupyter step in the generated SKILL.md looks like:

```markdown
### Step 3: Train Random Forest model

Fit the model on feature-engineered training data.

**Tool**: Jupyter MCP (`mcp__ide__executeCode`)

    ```python
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    ```

**Expected output**: Training completes, shows feature importances

**Success criteria**: Model trains without errors

**If criteria not met**: Check for data loading errors in previous cells
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

# 7. (Optional) Record a detailed notebook walkthrough and refine
uv run python -m src.workflow_learning start "fp-notebook-detail" \
  -d "Detailed walkthrough of FP notebook cells, metrics to check, and decision points"
# ... walk through notebook cells, narrate expected metrics and what to look for ...
uv run python -m src.workflow_learning stop
uv run python -m src.workflow_learning analyze <new-session-id> --refine "model-training-fp"

# 8. Stop Screenpipe (Ctrl+C in its terminal)
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
- **Overwrites on re-analyze:** Running `analyze` with the same skill name (or `--refine`) overwrites the previous skill file.
- **Refinement requires existing skill:** The `--refine` flag requires a SKILL.md file already exists at `.claude/skills/learned/<name>/SKILL.md`.
- **No selective Screenpipe cleanup:** Cannot delete recordings for a specific time window via CLI.
- **Future planned:** Pause/resume support, Screenpipe data trimming to session windows only.
