"""Interactive Streamlit interface for the existing LLM Sampling Lab backend.

Launch from the repository root:
    uv run --with streamlit streamlit run src/llm_sampling_lab/ui.py
"""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from llm_sampling_lab.config import load_settings, validate_settings
from llm_sampling_lab.experiments import (
    run_temperature_experiment,
    run_top_p_experiment,
)
from llm_sampling_lab.groq_client import create_groq_client
from llm_sampling_lab.movie_reviews import (
    DEFAULT_MOVIE,
    DEFAULT_SCENARIOS,
    Movie,
    run_movie_review_experiment,
)
from llm_sampling_lab.results import ExperimentRecorder, MovieReviewRecorder
from llm_sampling_lab.simulator import BASE_PROBABILITIES, BASE_TOKENS
from llm_sampling_lab.ui_helpers import (
    list_report_files,
    load_report,
    run_offline_simulation,
)


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "results"
CONFIG = ROOT / "settings.toml"
DEFAULT_PROMPT = (
    "Create one short and memorable advertising slogan "
    "for an electric car brand. Return only the slogan."
)

st.set_page_config(
    page_title="LLM Sampling Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp {
          background:
              radial-gradient(ellipse at top right, #182c4a 0%, transparent 37%),
              #090f1f;
          color: #eaf0fc;
      }
      [data-testid="stSidebar"] {
          background: #0c172b;
          border-right: 1px solid #26344e;
      }
      .block-container { max-width: 1360px; padding-top: 2rem; }
      .hero {
          border: 1px solid #30466b;
          border-radius: 22px;
          background: linear-gradient(115deg, #152b4e 0%, #13213b 63%, #212453 100%);
          padding: 2rem 2.25rem;
          margin-bottom: 1.65rem;
      }
      .eyebrow {
          color: #83c7ff;
          letter-spacing: .17em;
          text-transform: uppercase;
          font-size: .75rem;
          font-weight: 700;
      }
      .hero h1 { color: #ffffff; margin: .4rem 0 .55rem; font-size: 2.45rem; }
      .hero p { color: #bac9e5; max-width: 780px; margin: 0; }
      [data-testid="stMetric"] {
          border: 1px solid #2c4060;
          border-radius: 16px;
          background: #122039;
          padding: 1.1rem 1.2rem;
      }
      div.stButton > button[kind="primary"],
      div.stFormSubmitButton > button[kind="primary"] {
          background: linear-gradient(90deg, #317beb, #6667e9);
          border: none;
          border-radius: 11px;
          font-weight: 700;
      }
      div[data-testid="stExpander"] {
          border: 1px solid #263953;
          border-radius: 12px;
      }
      .footnote { color: #9fb0cd; font-size: .85rem; }
      a { color: #7ebeff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def hero(title: str, description: str) -> None:
    """Render a consistent, trusted application header."""

    # Header arguments are controlled constants, never free-form API output.
    st.markdown(
        '<section class="hero">'
        '<div class="eyebrow">Experiment workspace · v1</div>'
        f"<h1>{title}</h1><p>{description}</p>"
        "</section>",
        unsafe_allow_html=True,
    )


def show_download(report_path: Path) -> None:
    if report_path.is_file():
        st.download_button(
            "Download JSON report",
            data=report_path.read_bytes(),
            file_name=report_path.name,
            mime="application/json",
            key=f"download-{report_path.name}",
        )


def show_report(report: dict) -> None:
    """Present an existing JSON report without rendering response HTML."""

    status = report.get("status", "unknown")
    summary = report.get("summary", [])
    results = report.get("results", [])

    first, second, third = st.columns(3)
    first.metric("Run status", str(status).capitalize())
    second.metric("Recorded outputs", len(results))
    third.metric("Model", report.get("model", "Local"))

    if isinstance(summary, list) and summary:
        st.subheader("Output diversity")
        frame = pd.DataFrame(summary)
        st.dataframe(frame, hide_index=True, use_container_width=True)
        if "unique_response_ratio" in frame.columns:
            plotted = frame.dropna(subset=["unique_response_ratio"])
            if not plotted.empty:
                st.bar_chart(
                    plotted.set_index("parameter_value")["unique_response_ratio"],
                    height=230,
                )
    elif isinstance(summary, dict):
        st.json(summary)

    if results:
        st.subheader("Generated outputs")
        for number, result in enumerate(results, start=1):
            if not isinstance(result, dict):
                continue
            label = (
                f"{number:02d} · T={result.get('temperature', '—')}"
                f" · Top-p={result.get('top_p', '—')}"
                if "response" in result
                else f"{number:02d} · {result.get('review_style', 'Movie review')}"
            )
            with st.expander(label, expanded=number == 1):
                st.write(result.get("response") or result.get("review") or "")
                warnings = result.get("validation_issues") or []
                if warnings:
                    st.warning("Structural checks: " + ", ".join(warnings))


def dashboard(settings: dict) -> None:
    hero(
        "LLM Sampling Lab",
        "Explore how decoding parameters shape probability distributions "
        "and generated text. Start with an offline simulation, then compare "
        "real model outputs.",
    )

    a, b, c, d = st.columns(4)
    a.metric("Sampling strategies", "3")
    b.metric("Real API experiments", "2")
    c.metric("Movie review lab", "1")
    d.metric(
        "Groq API key",
        "Available" if os.getenv("GROQ_API_KEY", "").strip() else "Not set",
    )

    st.subheader("Base token distribution")
    frame = pd.DataFrame(
        {"Token": BASE_TOKENS, "Probability": BASE_PROBABILITIES}
    )
    st.bar_chart(frame, x="Token", y="Probability", height=290)
    st.caption(
        "This educational example is a predefined distribution, "
        "not token probabilities returned by Groq."
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### 01 · Offline Simulator")
        st.write(
            "Change Temperature, Top-k or Top-p. Compare theoretical "
            "probabilities with observed token frequencies using a fixed seed."
        )
    with right:
        st.markdown("#### 02 · Live Model Experiments")
        st.write(
            "Run repeated Groq generations, measure response diversity "
            "and export reproducible JSON reports."
        )

    st.info(
        "Top-k is applied locally only. The Groq Chat Completions API "
        "does not expose a Top-k sampling parameter."
    )


def offline_page() -> None:
    hero(
        "Offline Sampling Simulator",
        "Experiment with the distribution in real time, without an API key "
        "or network requests.",
    )

    left, right = st.columns([1, 2], gap="large")
    with left:
        strategy = st.radio(
            "Sampling method",
            ["Temperature", "Top-k", "Top-p"],
            help="Exactly one decoding method is applied per simulation.",
        )
        if strategy == "Temperature":
            value = st.slider(
                "Temperature", min_value=0.1, max_value=2.0,
                value=1.0, step=0.1,
            )
            st.caption("Lower values concentrate probability on likely tokens.")
            key = "temperature"
        elif strategy == "Top-k":
            value = st.slider(
                "Top-k", min_value=1, max_value=len(BASE_TOKENS),
                value=3, step=1,
            )
            st.caption("Retain exactly k candidate tokens.")
            key = "top-k"
        else:
            value = st.slider(
                "Top-p", min_value=0.05, max_value=1.0,
                value=0.7, step=0.05,
            )
            st.caption("Keep the smallest token set reaching the threshold.")
            key = "top-p"

        samples = int(
            st.number_input(
                "Token draws", min_value=100, max_value=100000,
                value=10000, step=100,
            )
        )
        seed = int(
            st.number_input(
                "Random seed", min_value=0, max_value=2147483647,
                value=42, step=1,
            )
        )
        run = st.button("Run simulation", type="primary", use_container_width=True)

    if run or "offline_result" not in st.session_state:
        try:
            rows = run_offline_simulation(key, value, samples, seed)
            st.session_state["offline_result"] = {
                "rows": rows,
                "strategy": key,
                "value": value,
                "samples": samples,
                "seed": seed,
            }
        except ValueError as exc:
            st.error(str(exc))
            return

    result = st.session_state["offline_result"]
    rows = result["rows"]
    frame = pd.DataFrame(rows)
    with right:
        st.subheader("Expected vs observed")
        st.bar_chart(
            frame,
            x="Token",
            y=["Expected", "Observed"],
            stack=False,
            height=355,
        )
        a, b, c = st.columns(3)
        a.metric("Selected method", result["strategy"])
        b.metric("Sample size", f"{result['samples']:,}")
        c.metric(
            "Max absolute gap",
            f"{max(abs(r['Expected'] - r['Observed']) for r in rows):.4f}",
        )

    st.subheader("Distribution details")
    st.dataframe(
        frame,
        column_config={
            "Expected": st.column_config.NumberColumn(format="%.4f"),
            "Observed": st.column_config.NumberColumn(format="%.4f"),
        },
        hide_index=True,
        use_container_width=True,
    )
    st.download_button(
        "Export simulation JSON",
        json.dumps(result, ensure_ascii=False, indent=2),
        file_name="offline_sampling_result.json",
        mime="application/json",
    )
    st.caption(
        "The same parameters and seed reproduce the same observed frequencies. "
        "Only the five educational base tokens are sampled."
    )


def live_page(settings: dict) -> None:
    hero(
        "Live LLM Experiments",
        "Compare repeated Groq responses as one decoding parameter changes. "
        "Each response is saved immediately to a JSON report.",
    )
    model = settings["groq"]["model"]

    left, right = st.columns([1, 2], gap="large")
    with left:
        experiment = st.selectbox("Experiment", ["Temperature", "Top-p"])
        if experiment == "Temperature":
            values = st.multiselect(
                "Temperature values",
                options=settings["temperature"]["values"],
                default=settings["temperature"]["values"],
            )
            fixed = st.slider(
                "Fixed Top-p", min_value=0.05, max_value=1.0,
                value=float(settings["temperature"]["top_p"]), step=0.05,
            )
            parameter = "temperature"
            fixed_name = "top_p"
        else:
            values = st.multiselect(
                "Top-p values",
                options=settings["top_p"]["values"],
                default=settings["top_p"]["values"],
            )
            fixed = st.slider(
                "Fixed Temperature", min_value=0.0, max_value=2.0,
                value=float(settings["top_p"]["temperature"]), step=0.1,
            )
            parameter = "top_p"
            fixed_name = "temperature"

        repetitions = int(
            st.number_input(
                "Repetitions per value",
                min_value=1, max_value=10,
                value=min(int(settings["experiments"]["repetitions"]), 10),
            )
        )
        prompt = st.text_area(
            "Prompt", DEFAULT_PROMPT, height=150,
            help="The same prompt is used for every request.",
        )
        configured = bool(os.getenv("GROQ_API_KEY", "").strip())
        if not configured:
            st.warning(
                "Set GROQ_API_KEY in your environment to run live experiments. "
                "Offline simulations do not require it."
            )
        st.caption(f"Provider: Groq · Model: {model}")
        run = st.button(
            "Run live experiment",
            type="primary",
            disabled=not configured or not values or not prompt.strip(),
            use_container_width=True,
        )

    with right:
        st.subheader("Experiment plan")
        a, b = st.columns(2)
        a.metric("API requests", len(values) * repetitions)
        b.metric("Fixed parameter", f"{fixed_name} = {fixed:.2f}")
        st.write(
            "Output comparison focuses on the unique-response ratio. "
            "Because providers can be nondeterministic, repeated API "
            "requests are not guaranteed to return identical text."
        )
        st.info(
            "Real API Top-k is not available through the current Groq backend. "
            "Use the offline simulator to explore Top-k."
        )

    if run:
        recorder = ExperimentRecorder(
            model=model,
            prompt=prompt.strip(),
            experiment_type=parameter,
            parameter_values=list(values),
            fixed_parameter=fixed_name,
            fixed_value=fixed,
            repetitions=repetitions,
            output_dir=REPORTS,
        )
        progress = st.progress(0, text="Generating responses...")
        total = len(values) * repetitions

        def record_and_update(result) -> None:
            recorder.record(result)
            progress.progress(
                len(recorder.results) / total,
                text=f"Saved {len(recorder.results)} of {total} outputs",
            )

        try:
            with create_groq_client() as client:
                if parameter == "temperature":
                    run_temperature_experiment(
                        client=client,
                        model=model,
                        prompt=prompt.strip(),
                        temperature_values=list(values),
                        top_p=fixed,
                        repetitions=repetitions,
                        on_result=record_and_update,
                    )
                else:
                    run_top_p_experiment(
                        client=client,
                        model=model,
                        prompt=prompt.strip(),
                        top_p_values=list(values),
                        temperature=fixed,
                        repetitions=repetitions,
                        on_result=record_and_update,
                    )
            recorder.complete()
            st.success("Experiment complete. All responses were saved.")
        except Exception as exc:
            recorder.fail(exc)
            st.error(
                "Live generation failed. Partial results were preserved. "
                f"Error type: {type(exc).__name__}"
            )
        st.session_state["live_path"] = str(recorder.file_path)

    if "live_path" in st.session_state:
        path = Path(st.session_state["live_path"])
        if path.is_file():
            st.divider()
            show_report(load_report(path))
            show_download(path)


def movie_page(settings: dict) -> None:
    hero(
        "Movie Review Studio",
        "Use local Temperature and Top-k to select a review style, then "
        "ask Groq to generate a Turkish review.",
    )
    model = settings["groq"]["model"]
    with st.form("movie_options"):
        first, second = st.columns(2)
        with first:
            title = st.text_input("Film title", value=DEFAULT_MOVIE.title)
            genre = st.text_input("Genre", value=DEFAULT_MOVIE.genre)
            seed = int(st.number_input("Random seed", min_value=0, value=42))
        with second:
            temperature = st.slider(
                "Temperature", 0.1, 2.0, 0.7, step=0.1,
            )
            top_k = st.slider("Local Top-k", 1, 4, 3)
            compare = st.checkbox(
                "Compare four preset scenarios instead",
                value=False,
            )
        plot = st.text_area(
            "Plot description", value=DEFAULT_MOVIE.description, height=110,
        )
        submitted = st.form_submit_button(
            "Generate movie reviews",
            type="primary",
            disabled=not bool(os.getenv("GROQ_API_KEY", "").strip()),
        )

    st.info(
        "Top-k applies only to local review-style selection. It does not "
        "change Groq's internal token sampling."
    )
    if not os.getenv("GROQ_API_KEY", "").strip():
        st.warning("A GROQ_API_KEY is required for generated movie reviews.")

    if submitted:
        if not all(text.strip() for text in (title, genre, plot)):
            st.error("Provide a non-empty film title, genre and description.")
        else:
            movie = Movie(title=title, genre=genre, description=plot)
            scenarios = DEFAULT_SCENARIOS if compare else ((temperature, top_k),)
            recorder = MovieReviewRecorder(
                model=model,
                movie=movie,
                scenarios=scenarios,
                seed=seed,
                output_dir=REPORTS,
            )
            progress = st.progress(0, text="Writing reviews...")

            def record_review(result) -> None:
                recorder.record(result)
                progress.progress(
                    len(recorder.results) / len(scenarios),
                    text=f"Saved {len(recorder.results)} of {len(scenarios)} reviews",
                )

            try:
                with create_groq_client() as client:
                    run_movie_review_experiment(
                        client=client,
                        model=model,
                        movie=movie,
                        scenarios=scenarios,
                        seed=seed,
                        on_result=record_review,
                    )
                recorder.complete()
                st.success("Movie review experiment completed.")
            except Exception as exc:
                recorder.fail(exc)
                st.error(
                    "Review generation failed. Partial results were saved. "
                    f"Error type: {type(exc).__name__}"
                )
            st.session_state["movie_path"] = str(recorder.file_path)

    if "movie_path" in st.session_state:
        path = Path(st.session_state["movie_path"])
        if path.is_file():
            st.divider()
            show_report(load_report(path))
            show_download(path)


def history_page() -> None:
    hero(
        "Experiment History",
        "Browse previously saved JSON reports and inspect results "
        "without repeating paid API requests.",
    )
    reports = list_report_files(REPORTS)
    if not reports:
        st.info("No reports yet. Run a simulation or live API experiment.")
        st.caption(
            "Offline simulations can be exported from the simulator; "
            "live and movie experiments are persisted under results/."
        )
        return

    names = {str(path.relative_to(REPORTS)): path for path in reports}
    choice = st.selectbox("Saved experiment", options=list(names))
    path = names[choice]
    try:
        show_report(load_report(path))
        show_download(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"Unable to open this report: {type(exc).__name__}")


def main() -> None:
    try:
        settings = load_settings(CONFIG)
        validate_settings(settings)
    except (OSError, ValueError) as exc:
        st.error(f"Cannot load settings.toml: {exc}")
        st.stop()

    with st.sidebar:
        st.markdown("### LLM Sampling Lab")
        st.caption("Interactive experiment console")
        page = st.radio(
            "Navigation",
            [
                "Overview",
                "Offline Simulator",
                "Live Experiments",
                "Movie Review Studio",
                "Experiment History",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption(f"Current model: {settings['groq']['model']}")
        st.caption(
            "Groq key: "
            + ("configured" if os.getenv("GROQ_API_KEY", "").strip() else "not set")
        )
        st.caption("Offline simulations run without any API calls.")

    if page == "Overview":
        dashboard(settings)
    elif page == "Offline Simulator":
        offline_page()
    elif page == "Live Experiments":
        live_page(settings)
    elif page == "Movie Review Studio":
        movie_page(settings)
    else:
        history_page()


if __name__ == "__main__":
    main()
