> If your AI tool doesn't read this file automatically, paste it into the chat before you start working.

# SoundSense

## Project summary
SoundSense is a privacy-preserving household sound alert system for deaf and hard-of-hearing users. It listens through the laptop microphone (or a phone streaming its mic to the laptop), classifies sounds with YAMNet (e.g. doorbell, smoke alarm, baby crying, knocking), and alerts the user visually through desktop notifications, an always-on-top screen flash and a web dashboard. Users can teach it their own custom sounds. All processing runs locally. No audio leaves the user's own devices, and no audio is ever stored.

## Tech stack
- **Language:** Python 3.11
- **API:** FastAPI (served with Uvicorn)
- **ML:** TensorFlow (CPU), YAMNet via TensorFlow Hub, scikit-learn
- **Audio:** Librosa, sounddevice
- **Alerts:** desktop-notifier, tkinter
- **Storage:** SQLite
- **Frontend:** React + Vite + Tailwind CSS + Recharts

## Setup
- **Python 3.11 is required.** Not 3.12 or 3.13, because TensorFlow 2.16 doesn't support them.
- Create a virtual environment and install the backend dependencies:
  ```bash
  python3.11 -m venv .venv
  source .venv/bin/activate        # Windows: .venv\Scripts\activate
  pip install -r requirements.txt
  ```
- **Node 20+** is required for the frontend.

## Roles and folder ownership

**Shreeya** is the team leader and repo owner.

| Owner | Role | Owns |
|---|---|---|
| **Shreeya** | Role 1: AI & Audio Engineer | `backend/audio/`, `backend/ml/`, `backend/custom/`, `backend/tests/test_role1/`, `eval/model/`, `data/` |
| **Vaidehi** | Role 2: Backend & Alert Systems Engineer | `backend/engine/`, `backend/api/`, `backend/storage/`, `backend/notify/` (incl. the tkinter always-on-top flash), `backend/main.py`, `backend/tests/test_role2/`, `eval/system/` |
| **Sneha** | Role 3: Frontend & Visualization Engineer | `frontend/`, `listener/` (Phase 5 phone listener web page), `eval/report/` |
| **Shared** (team agreement required) | — | `contracts/`, `requirements.txt`, `AGENTS.md`, `CLAUDE.md`, `.gitignore` |

Each person works on their own branch and merges at the end. Because no two people edit the same files, merges should be conflict-free.

## Git workflow
- **Branches:** `shreeya-ai-audio` (Shreeya), `vaidehi-backend` (Vaidehi), `sneha-frontend` (Sneha).
- **Never push directly to `main`.** Changes reach `main` through pull requests only.
- **Merge order is flexible** as long as each PR touches only its owner's folders; the merges are followed by a joint integration session.
- **Pull `main` into your branch only when a `contracts/` change is announced.** Otherwise stay on your branch.

## Rules
**contracts/ is the source of truth. Code against it exactly. If it must change, tell the whole team and open a small pull request that changes only `contracts/`; Shreeya reviews and merges it, then announces it, and everyone pulls `main` into their branch.**

1. **Only edit files in your own role's folders.** If you need something from another role's code, depend on the interface in `contracts/` and don't change their files.
2. **Never modify `contracts/`, `requirements.txt`, `AGENTS.md` or `CLAUDE.md` without team agreement.** Propose changes to the team first, then open a small pull request that changes only those files; Shreeya reviews and merges it, then announces it.
3. **Never commit audio, models, `.db` files or `node_modules`.** `.gitignore` enforces this. Don't force-add around it.
4. **No audio is ever saved to disk**, with two exceptions: custom-sound embeddings (numeric vectors, not audio). Raw audio and spectrograms stay in memory only. Evaluation dataset recordings (WAV/m4a under data/indian/, recorded with consent via dev-only tools like --record-test) are allowed. The running app never saves audio.
