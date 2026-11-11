# SoundSense: A Privacy-Preserving, Personalized Household Sound Alert System for Deaf and Hard-of-Hearing Users

SoundSense listens for important household sounds (smoke alarms, doorbells, a baby crying, pressure-cooker whistles and more) and turns them into clear visual alerts on the laptop and a web dashboard.
Sounds are classified on-device with YAMNet, and users can teach it their own custom sounds.
Everything runs locally: no audio leaves the user's devices, and no audio is ever saved.

## Team

| Member | Role |
|---|---|
| **Shreeya** (team leader) | AI & Audio |
| **Vaidehi** | Backend & Alerts |
| **Sneha** | Frontend & Visualization |

## Quick setup

Requires **Python 3.11** (not 3.12/3.13) and **Node 20+**.

**Backend** (from the repo root):

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

The API runs at http://localhost:8000. Check http://localhost:8000/api/status.

**Frontend:**

```bash
cd frontend && cp .env.example .env && npm install && npm run dev
```

The app runs at http://localhost:5173. `.env` uses mock data by default; set `VITE_USE_MOCK=false` to use the real backend.

## Contributing

**Read [AGENTS.md](AGENTS.md) before contributing.** It covers folder ownership, the git workflow and the rules for `contracts/`.
