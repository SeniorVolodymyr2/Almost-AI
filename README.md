# Almost-AI — Python client for AlmostRacing

This folder contains the Python side of the AlmostRacing project: a TCP client that connects to the Unity game, reads the agent state, and either trains a DQN policy or runs a rule-based baseline.

Unity acts as the **server** (`127.0.0.1:5005`). Python is the **client** and must be started **after** Unity is in Play mode with AI enabled.
# Unity Game URL:
### https://drive.google.com/drive/folders/1YSKSSqYz0qgACl7XVEAZ0lfeylgjKusw?usp=sharing

## Prerequisites

- Python 3.10+ (3.11 recommended)
- Unity project **AlmostRacing** running `MainScene` with AI mode on (`useAI = 1`, single agent)
- Network: default host/port must match between Unity and `.env` (`127.0.0.1:5005`)

## Setup

From this directory:

```bash
pip install -r requirements.txt
```

Copy or edit `.env` for your run. All runtime options are read from that file via `config.py`.

## Quick start

1. Open **AlmostRacing** in Unity and press **Play** (wait until the game is running and listening on port 5005).
2. In a terminal, from this folder:

```bash
python client.py
```

`train.py` is an alias entry point and does the same thing:

```bash
python train.py
```

The client reconnects automatically if Unity restarts. Press `Ctrl+C` to stop; the current model is saved on exit.

Console output includes request/response rate (`req/s resp/s`) and per-episode training logs when `TRAIN=true`.

## Configuration (`.env`)

| Variable | Description |
|----------|-------------|
| `HOST` | Unity TCP host (default `127.0.0.1`) |
| `PORT` | Unity TCP port (default `5005`) |
| `TRAIN` | `true` = train DQN; `false` = inference only (no gradient updates) |
| `BASELINE` | `true` = rule-based agent (ignores neural net for actions) |
| `MODEL_NAME` | Prefix for checkpoints and evaluation files (default `racing_dqn`) |
| `MODELS_DIR` | Folder for `.pt` weights (default `models`) |
| `MODEL_LOAD_EPISODE` | Episode number to load, e.g. `25` → `models/racing_dqn_25.pt`. Leave empty for no explicit load |
| `LOAD_LATEST` | `true` = load highest episode checkpoint in `MODELS_DIR` when `MODEL_LOAD_EPISODE` is empty |
| `EPSILON` | Initial exploration rate (training) |
| `EPSILON_MIN` | Minimum epsilon after decay |
| `EPSILON_DECAY` | Multiplier applied **once per completed episode** |
| `MAX_EPISODES` | Stop training after this many episodes |
| `SAVE_EVERY_EPISODES` | Save checkpoint every N episodes |
| `TRAIN_EVERY_N_STEPS` | Run one gradient step every N environment steps |
| `EVALUATION_DIR` | CSV and chart output folder (default `evaluation`) |
| `RECONNECT_DELAY` | Seconds to wait before reconnecting to Unity |

### Common run modes

**Train from scratch**

```env
TRAIN=true
BASELINE=false
MODEL_LOAD_EPISODE=
LOAD_LATEST=false
EPSILON=1.0
```

**Resume training from a checkpoint**

```env
TRAIN=true
MODEL_LOAD_EPISODE=25
LOAD_LATEST=false
```

**Evaluate a trained model (no learning)**

```env
TRAIN=false
BASELINE=false
MODEL_LOAD_EPISODE=25
```

**Rule-based sanity check (no ML)**

```env
TRAIN=false
BASELINE=true
```

## Outputs

| Path | Content |
|------|---------|
| `models/{MODEL_NAME}_{episode}.pt` | PyTorch checkpoint (saved every `SAVE_EVERY_EPISODES` and on exit) |
| `evaluation/{MODEL_NAME}.metrics.csv` | Per-episode score, reward, epsilon |
| `evaluation/{MODEL_NAME}_training.png` | Reward and score charts |

## Project layout

| File | Role |
|------|------|
| `client.py` | TCP client loop; connects to Unity and drives the agent |
| `train.py` | Entry point that calls `client.main()` |
| `agent.py` | `RacingBrain` — epsilon-greedy policy, episode handling, saves |
| `train_dqn.py` | DQN trainer (replay + target network) |
| `model.py` | Q-network architecture |
| `replay_buffer.py` | Experience replay deque |
| `env.py` | Reward shaping on top of Unity state |
| `protocol.py` | JSON parsing and action → `ForceX` mapping |
| `baseline.py` | Simple rule-based steering for testing |
| `config.py` | Loads settings from `.env` |
| `model_paths.py` | Checkpoint path helpers |
| `tracker.py` / `plot.py` | Metrics CSV and training plots |

## Protocol (short)

Unity sends newline-delimited JSON state (e.g. `HasObstacle`, `GapX`, `GapZ`, `Vx`, `Score`, `AgentIsDone`). Python replies with:

```json
{"ForceX": 0.0, "IsDone": false}
```

Actions are discrete: left (0), stay (1), right (2), mapped to `ForceX` in `protocol.py`.

## Troubleshooting

- **Connection refused** — Start Unity Play mode first; confirm port `5005` and `HOST`/`PORT` in `.env`.
- **Agent does not move** — Ensure Unity AI mode is on and only one agent is active.
- **No learning** — Set `TRAIN=true`, `BASELINE=false`, and avoid loading old weights unless resuming (`LOAD_LATEST=false` for a fresh run).
- **Wrong checkpoint loaded** — `MODEL_LOAD_EPISODE` is the number in the filename (`racing_dqn_38.pt` → `38`), not the Unity episode counter.
