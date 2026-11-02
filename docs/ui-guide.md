# Web UI · LLM Sampling Lab

A Streamlit dashboard for the repository's existing Python experiment engine. No new runtime dependency is added to `pyproject.toml` or `uv.lock`; Streamlit is installed in an isolated uv environment on demand.

## Start the dashboard

From the project root (PowerShell, Windows Terminal, macOS, or Linux):

```bash
uv run --with "streamlit>=1.45,<2" streamlit run src/llm_sampling_lab/ui.py
```

Streamlit prints a local URL, normally **http://localhost:8501**. Open that URL in a browser.

**Offline Simulator requires no API key.** For live experiments and generated movie reviews, create a private `.env` file as described in the main README, then run:

```bash
uv run --env-file .env --with "streamlit>=1.45,<2" streamlit run src/llm_sampling_lab/ui.py
```

Keep the key out of version control. The UI reads `GROQ_API_KEY` from the process environment; it does not display or store the key in reports.

## Available pages

| Page | Capabilities | API key |
|---|---|---|
| Overview | Base token distribution, experiment overview, status | No |
| Offline Simulator | Temperature, Top-k, Top-p; theoretical vs observed bar charts; reproducible seed; JSON export | No |
| Live Experiments | Temperature / Top-p batch runs, repetition controls, progress, response diversity, JSON reports | Yes |
| Movie Review Studio | Fictional movie inputs, local style sampling using Temperature + Top-k, Groq reviews and structural warnings | Yes |
| Experiment History | Inspect and download previously saved live and movie JSON reports | No |

Live and movie reports are written immediately under `results/` by the existing recorders, including partial results if a request fails. Offline results are downloadable from the simulator page, but are **not** automatically written to `results/`.

## Provider capability boundaries

- The current repository's implementation uses **Groq only** for online text generation.
- Groq's Chat Completions API does **not** expose a Top-k argument.
- Offline Top-k modifies only the educational token distribution; movie-review Top-k chooses a **review style locally**.
- The provider selector can be expanded in a future PR after other provider backends are actually implemented.

## Development and checks

```bash
uv run pytest tests/test_ui_helpers.py
uv run pytest
```

Streamlit is a separately launched interface. The existing CLI, Dockerfile, Compose service, and locked CLI dependencies remain unchanged.

If PowerShell cannot find `uv`, install uv first and restart the terminal. Use Python 3.12 or newer, consistent with `pyproject.toml`.
