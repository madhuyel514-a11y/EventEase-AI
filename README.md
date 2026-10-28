# EventEase AI — A Budget-Aware Celebration Planner

A responsive celebration-planning app built for PromptWars Virtual 2026, Challenge 5. Create a practical plan for birthdays, weddings, corporate events, anniversaries, and community events; see an independently calculated budget, an event-specific checklist, and vendor categories to research.

## Features

- Personalized event form: type (including corporate events), date, description, INR budget, guest count, venue, food, dietary, cultural, and accessibility preferences.
- Event-specific preparation checklists cover ceremony and guest coordination for weddings; cake, invitations and activities for birthdays; and agenda, speakers, AV and attendee coordination for corporate events, with templates for the other supported types.
- Gemini-generated structured plans through a Python FastAPI backend, when `GEMINI_API_KEY` is configured.
- Explicit deterministic demo mode when no Gemini key is present. Demo plans are labelled; they are never represented as AI output.
- BudgetGuard calculates allocation totals and per-guest estimates in Python, flags overspending, and suggests cost-saving changes.
- Clickable preparation checklist, vendor-category prompts, and a what-if form to regenerate for a changed budget or guest count.
- Budget Shock Simulator models one category increase, added guests, or fixed expense without modifying the original plan or calling Gemini.
- Adaptive Recovery Planner analyzes the existing task deadlines deterministically and offers date-only suggestions for explicit local application; it does not call Gemini or alter the event date.
- AI Trade-Off Explorer compares deterministic Budget Saver, Balanced, and Premium allocation scenarios without changing the current plan or calling Gemini.
- Clear estimate and vendor-availability disclaimers. No bookings, real quotations, or vendor availability are claimed.

## Architecture

- `frontend/`: React 19 and Vite SPA, with responsive CSS and lucide icons.
- `backend/app/main.py`: FastAPI routes, Pydantic request/response validation, Gemini integration, demo plan, and BudgetGuard calculations.
- `backend/tests/`: API, validation, and budget calculation tests.
- `POST /api/plan`: validates an event request and returns an AI or demo plan with server-calculated budget totals.
- `POST /api/timeline-recovery`: evaluates supplied event/task dates against the current pre-event window and returns advisory deadline suggestions.
- `POST /api/trade-offs`: returns deterministic budget-safe alternatives based on the submitted current plan.
- `GET /api/health`: simple backend readiness endpoint.

The Gemini key is read only by the backend from `GEMINI_API_KEY`; it is never included in the frontend bundle. AI plan fields are requested as JSON and revalidated with Pydantic before use.

Budget Shock recovery uses deterministic maximum reductions against each original allocation: decor/decoration/flowers up to 25%, entertainment/music/DJ up to 20%, and favors/favours up to 50%. Categories marked essential, required, or mandatory are excluded. Other categories are not reduced. Recovery reports any remaining shortfall; all scenario amounts are estimates, not vendor quotes.

The simulator checks the plan cost sent by the frontend against the sum of its baseline allocations exactly; amounts are whole INR, so no rounding tolerance is applied. Currency averages and per-guest estimates use round-half-up to the nearest rupee.

Generated checklists use deterministic event-type task templates and event-relative lead dates, ordered chronologically; short-notice events use today as the earliest checklist deadline. The templates incorporate guest count, budget and relevant food, dietary, cultural, and accessibility preferences. They are preparation suggestions, not evidence of bookings, vendor availability, or task completion.

Timeline recovery accepts an event date and up to 12 validated tasks. Its available window is today through the day before the event; for a same-day event, today is the only available date. Past event dates have no available recovery window. It classifies tasks as on schedule, needing adjustment, or unresolved. Title-based lead-time guidelines use longer planning windows for venue/logistics, catering, and photography tasks, with shorter windows for nearer-term preparation. If no date remains that meets a task's guideline (for example, a venue task for an event tomorrow), the task is unresolved rather than assigned a misleading deadline. Suggestions only adjust dates and do not assess task duration, dependencies, vendor availability, or operational feasibility; the original checklist and event date are not silently changed.

The Trade-Off Explorer is named for the feature but is not AI-generated and does not search live vendor prices. It first normalizes category names only by case-folding and collapsing whitespace, then maps explicitly listed exact aliases into canonical groups; no substring or rationale matching is used. Decor aliases include `Decor`, `Decor & Ambient Lighting`, `Decor & Ambient Styling`, `Decor & Styling`, `Decor & Lighting`, `Decor, Lighting & Backdrop`, `Thematic Decor & Lighting`, `Themed Decor & Backdrop`, `Decor & Floral Accents`, `Floral & Mandap Decor`, `Floral Arrangements`, `Event Decorations`, and `Table Decorations`. Entertainment/DJ aliases include `Entertainment`, `Audio & Entertainment`, `Audio & Digital Invitations`, `Entertainment & Sound`, `Sound & Entertainment`, `Sound, DJ & Entertainment`, `Sound & DJ Setup`, `DJ & Sound Equipment`, `Sound System & Music`, `Music`, and `DJ`. Favor aliases include `Favor`, `Favors`, `Favour`, `Favours`, `Favors & Stationery`, and `Party Favors & Digital Invites`. Budget Saver reductions are capped at 25% for decor, 20% for entertainment/music/DJ, and 50% for favors. Premium increases are capped at 20% for decor, 15% for entertainment/music/DJ, and 25% for favors. Caps use whole-rupee floor arithmetic against baseline amounts. Premium applies available headroom in that order and never exceeds the budget; any headroom left after caps is reported and remains unspent. Venue (`Venue Rental`, `Venue Rental & Setup`), catering (`Catering & Beverages`, `Catering (Vegetarian & Gluten-Free)`), cake/desserts, photography, contingency, and unknown categories remain unchanged. All scenario figures are planning estimates, not verified vendor quotations.

## Prerequisites

- Node.js 20.19+ (or 22.12+), npm
- Python 3.10+
- Optional: a Google Gemini API key for AI generation. Without it, the app works in clearly labelled demo mode.

## Run locally

Open two terminals from the project root.

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Add your key to backend\.env to enable Gemini; leave it blank for demo mode.
uvicorn app.main:app --reload
```

The API runs at `http://127.0.0.1:8000`. FastAPI docs are at `http://127.0.0.1:8000/docs`.

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal (normally `http://127.0.0.1:5173`). The frontend uses `http://127.0.0.1:8000` by default. For a different API host, set `VITE_API_URL` before starting Vite.

## Environment variables

| Variable | Used by | Description |
|---|---|---|
| `GEMINI_API_KEY` | Backend only | Google AI Studio API key; leave empty for deterministic demo mode |
| `GEMINI_MODEL` | Backend | Recommended and live-tested model: `gemini-3.1-flash-lite`; code defaults to `gemini-3.8-flash` if unset |
| `CORS_ORIGINS` | Backend | Comma-separated exact frontend origins; defaults to the two local Vite origins |
| `VITE_API_URL` | Frontend build/dev | Optional backend base URL; defaults to `http://127.0.0.1:8000` |

Do not commit `.env` files or paste API keys into frontend source. `.gitignore` excludes local environment files and generated dependencies/builds.

## Tests and build

```powershell
# Backend tests
cd backend
python -m pytest -q
```

Checklist matching helper tests use Node's built-in test runner and need no additional dependencies:

```powershell
cd frontend
node --test src/taskChecklist.test.js

# Frontend production build
cd ..\frontend
npm run build
```

## Deployment

Deploy the frontend static output from `frontend/dist` to any static host. Set `VITE_API_URL` to the publicly reachable HTTPS URL of the backend at frontend build time. Deploy the FastAPI application to a Python ASGI host, configure `GEMINI_MODEL=gemini-3.1-flash-lite`, store `GEMINI_API_KEY` as a server-side secret, and run:

```sh
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

For deployment, configure `CORS_ORIGINS` on the backend to only the deployed frontend origin (for example, `https://events.example.com`) and build the frontend with `VITE_API_URL` set to the backend HTTPS URL. Start the API with `uvicorn app.main:app --host 0.0.0.0 --port 8000`; set the port and host flags to match the hosting platform. Use HTTPS for the API and frontend.
