"""Plant Health Monitor - Streamlit application."""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from utils.features import FEATURE_NAMES, features_as_dict
from utils.prediction import (
    LABEL_COLUMN,
    MODEL_NAMES,
    create_synthetic_dataset,
    feature_importance,
    load_or_train_model,
    model_path,
    predict_leaf,
    save_bundle,
    train_and_evaluate,
    validate_dataset,
)
from utils.preprocessing import decode_uploaded_image, preprocess_image

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DATASET_DIR = BASE_DIR / "dataset"

for directory in (MODELS_DIR, REPORTS_DIR, DATASET_DIR):
    directory.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Plant Health Monitor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_styles() -> None:
    """Apply the agriculture-themed interface styling."""
    st.markdown(
        """
        <style>
        :root {
            --forest: #12372a;
            --leaf: #2f855a;
            --mint: #e8f5ec;
            --lime: #9bcf53;
            --ink: #183028;
        }
        .stApp {
            background:
                radial-gradient(circle at 85% 5%, rgba(155,207,83,.14), transparent 25%),
                linear-gradient(180deg, #f8fcf8 0%, #eef7f0 100%);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #102d23, #1d513c);
        }
        [data-testid="stSidebar"] * { color: #f3fff5; }
        .hero {
            padding: 3.3rem 3rem;
            border-radius: 28px;
            color: white;
            background:
                linear-gradient(120deg, rgba(11,50,37,.97), rgba(43,112,75,.9)),
                radial-gradient(circle at top right, #9bcf53, transparent 35%);
            box-shadow: 0 18px 45px rgba(18,55,42,.18);
            margin-bottom: 1.5rem;
        }
        .hero h1 {
            font-size: clamp(2.2rem, 5vw, 4.3rem);
            line-height: 1.03;
            margin: 0 0 1rem;
            max-width: 800px;
        }
        .hero p { font-size: 1.12rem; max-width: 720px; opacity: .9; }
        .eyebrow {
            display: inline-block; text-transform: uppercase; letter-spacing: .13em;
            font-weight: 700; font-size: .76rem; color: #c8ef94; margin-bottom: .9rem;
        }
        .feature-card, .result-card {
            background: rgba(255,255,255,.88);
            border: 1px solid rgba(47,133,90,.13);
            border-radius: 20px;
            padding: 1.35rem;
            min-height: 155px;
            box-shadow: 0 8px 25px rgba(23,65,47,.07);
        }
        .feature-card h3 { color: var(--forest); margin: .35rem 0; font-size: 1.1rem; }
        .feature-card p { color: #587064; font-size: .93rem; }
        .icon {
            width: 42px; height: 42px; border-radius: 12px; display: grid;
            place-items: center; background: var(--mint); font-size: 1.3rem;
        }
        .result-card { min-height: 125px; }
        .result-label {
            color: #6d8177; text-transform: uppercase; letter-spacing: .08em;
            font-size: .72rem; font-weight: 700;
        }
        .result-value { color: var(--ink); font-size: 1.55rem; font-weight: 750; margin-top: .4rem; }
        .recommendation {
            border-left: 5px solid var(--lime); background: #f4faee;
            padding: 1.1rem 1.25rem; border-radius: 4px 14px 14px 4px;
            color: #294636;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,.82); border: 1px solid #dcecdf;
            padding: 1rem; border-radius: 16px;
        }
        .section-kicker { color: #2f855a; font-weight: 700; letter-spacing: .08em; }
        .small-note { color: #667a70; font-size: .85rem; }
        @media (max-width: 700px) {
            .hero { padding: 2rem 1.25rem; border-radius: 20px; }
            .hero h1 { font-size: 2.35rem; }
            .feature-card { min-height: auto; margin-bottom: .65rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_model(model_name: str) -> dict:
    return load_or_train_model(MODELS_DIR, model_name)


@st.cache_resource(show_spinner=False)
def get_comparison_metrics() -> pd.DataFrame:
    data = create_synthetic_dataset()
    rows = []
    for name in MODEL_NAMES:
        bundle = load_or_train_model(MODELS_DIR, name)
        rows.append({"Model": name, **bundle["metrics"]})
    return pd.DataFrame(rows)


def initialize_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("custom_bundles", {})
    st.session_state.setdefault("latest_result", None)
    st.session_state.setdefault("latest_image", None)
    st.session_state.setdefault("latest_features", None)


def active_bundle(model_name: str) -> dict:
    return st.session_state.custom_bundles.get(model_name) or get_model(model_name)


def result_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">{label}</div>
            <div class="result-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_pdf_report(
    image_rgb: np.ndarray, result: dict, model_name: str, feature_values: np.ndarray
) -> bytes:
    """Create a two-page, downloadable PDF using Matplotlib."""
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="#f6fbf7")
        grid = fig.add_gridspec(3, 1, height_ratios=[0.18, 0.52, 0.30])
        title_ax = fig.add_subplot(grid[0])
        title_ax.axis("off")
        title_ax.text(0, 0.82, "PLANT HEALTH MONITOR", fontsize=12, color="#2f855a", weight="bold")
        title_ax.text(0, 0.38, "Leaf Analysis Report", fontsize=25, color="#12372a", weight="bold")
        title_ax.text(
            0,
            0.08,
            datetime.now().strftime("Generated %d %B %Y at %I:%M %p"),
            fontsize=10,
            color="#667a70",
        )

        image_ax = fig.add_subplot(grid[1])
        image_ax.imshow(image_rgb)
        image_ax.axis("off")
        image_ax.set_title("Uploaded leaf image", loc="left", fontsize=12, color="#183028")

        detail_ax = fig.add_subplot(grid[2])
        detail_ax.axis("off")
        details = (
            f"Prediction:  {result['prediction']}\n"
            f"Confidence:  {result['confidence']:.1%}\n"
            f"Health score:  {result['health_score']:.1%}\n"
            f"Model:  {model_name}\n\n"
            f"Recommendation\n{result['recommendation']}"
        )
        detail_ax.text(
            0,
            0.95,
            details,
            va="top",
            fontsize=12,
            color="#183028",
            linespacing=1.65,
            wrap=True,
        )
        fig.tight_layout(pad=2.2)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.27, 11.69), facecolor="#f6fbf7")
        important = pd.Series(feature_values, index=FEATURE_NAMES).sort_values(ascending=False).head(15)
        ax.barh(important.index[::-1], important.values[::-1], color="#4c956c")
        ax.set_title("Largest extracted feature values", loc="left", fontsize=20, color="#12372a")
        ax.set_xlabel("Feature value")
        ax.grid(axis="x", alpha=0.2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout(pad=2.5)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()


def render_landing() -> None:
    st.markdown(
        """
        <div class="hero">
            <span class="eyebrow">AI-assisted crop care</span>
            <h1>See plant stress before it spreads.</h1>
            <p>
                Analyze leaf color, texture, and edge patterns with transparent,
                traditional machine learning. Fast screening, clear confidence,
                and practical next steps in one dashboard.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### A practical toolkit for healthier plants")
    columns = st.columns(4)
    cards = [
        ("🔍", "Visual diagnosis", "OpenCV converts every leaf into measurable color and texture signals."),
        ("🧠", "Three ML models", "Compare Random Forest, SVM, and Logistic Regression without neural networks."),
        ("📈", "Clear analytics", "Inspect model quality, prediction history, distributions, and feature influence."),
        ("📄", "Shareable reports", "Download prediction records as CSV and individual assessments as PDF."),
    ]
    for column, (icon, title, text) in zip(columns, cards):
        with column:
            st.markdown(
                f'<div class="feature-card"><div class="icon">{icon}</div>'
                f"<h3>{title}</h3><p>{text}</p></div>",
                unsafe_allow_html=True,
            )

    st.divider()
    left, middle, right = st.columns(3)
    left.metric("Image size", "256 × 256", "Standardized")
    middle.metric("Extracted signals", str(len(FEATURE_NAMES)), "Color + texture")
    right.metric("Supported models", "3", "Scikit-learn")
    st.info(
        "This tool is a screening aid, not a laboratory diagnosis. Confirm serious "
        "or spreading symptoms with a qualified horticulture or agriculture professional."
    )


def render_diagnosis(model_name: str) -> None:
    st.markdown('<div class="section-kicker">LEAF DIAGNOSIS</div>', unsafe_allow_html=True)
    st.title("Analyze a plant leaf")
    st.caption("Use a well-lit, focused image with one leaf filling most of the frame.")

    uploaded_file = st.file_uploader(
        "Upload leaf image",
        type=["jpg", "jpeg", "png"],
        help="Maximum upload size follows your Streamlit server configuration.",
    )
    if uploaded_file is None:
        st.markdown("#### Good image checklist")
        col1, col2, col3 = st.columns(3)
        col1.info("Use natural or even lighting.")
        col2.info("Keep the leaf centered and in focus.")
        col3.info("Use a plain background when possible.")
        return

    try:
        image_bgr = decode_uploaded_image(uploaded_file)
        resized_bgr, image_rgb, image_hsv = preprocess_image(image_bgr)
    except ValueError as error:
        st.error(str(error))
        return

    preview, details = st.columns([1.15, 0.85])
    with preview:
        st.image(image_rgb, caption=uploaded_file.name, use_container_width=True)
    with details:
        st.subheader("Image details")
        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Original dimensions:** {image_bgr.shape[1]} × {image_bgr.shape[0]}")
        st.write("**Processing size:** 256 × 256")
        st.write(f"**Selected model:** {model_name}")
        st.caption("The image is processed in memory and is not uploaded to an external service.")
        analyze = st.button("Run health analysis", type="primary", use_container_width=True)

    if analyze:
        from utils.features import extract_features

        with st.spinner("Extracting visual features and evaluating leaf health..."):
            feature_values = extract_features(resized_bgr, image_rgb, image_hsv)
            bundle = active_bundle(model_name)
            result = predict_leaf(bundle, feature_values)

        timestamp = datetime.now()
        history_row = {
            "timestamp": timestamp.isoformat(timespec="seconds"),
            "file_name": uploaded_file.name,
            "model": model_name,
            "prediction": result["prediction"],
            "confidence": round(result["confidence"], 4),
            "health_score": round(result["health_score"], 4),
        }
        st.session_state.history.append(history_row)
        st.session_state.latest_result = result
        st.session_state.latest_image = image_rgb
        st.session_state.latest_features = feature_values

    result = st.session_state.latest_result
    if result is None:
        return

    st.success("Analysis complete.")
    st.markdown("### Prediction dashboard")
    columns = st.columns(4)
    with columns[0]:
        result_card("Prediction", result["prediction"])
    with columns[1]:
        result_card("Confidence", f"{result['confidence']:.1%}")
    with columns[2]:
        result_card("Health score", f"{result['health_score']:.1%}")
    with columns[3]:
        result_card("Status", result["status"])

    st.markdown(
        f'<div class="recommendation"><strong>Recommendation</strong><br>'
        f"{result['recommendation']}</div>",
        unsafe_allow_html=True,
    )

    with st.expander("View extracted image features"):
        feature_frame = pd.DataFrame(
            [features_as_dict(st.session_state.latest_features)]
        ).T.reset_index()
        feature_frame.columns = ["Feature", "Value"]
        st.dataframe(feature_frame, use_container_width=True, hide_index=True)

    report = build_pdf_report(
        st.session_state.latest_image,
        result,
        model_name,
        st.session_state.latest_features,
    )
    st.download_button(
        "Download PDF report",
        data=report,
        file_name=f"plant_health_report_{datetime.now():%Y%m%d_%H%M%S}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


def render_analytics(model_name: str) -> None:
    st.markdown('<div class="section-kicker">INSIGHTS</div>', unsafe_allow_html=True)
    st.title("Analytics")
    bundle = active_bundle(model_name)

    distribution = bundle.get("class_distribution", {0: 0, 1: 0})
    distribution_frame = pd.DataFrame(
        {
            "Status": ["Healthy", "Diseased"],
            "Samples": [distribution.get(0, 0), distribution.get(1, 0)],
        }
    ).set_index("Status")

    left, right = st.columns(2)
    with left:
        st.subheader("Training disease distribution")
        st.bar_chart(distribution_frame, color="#4c956c")
    with right:
        st.subheader("Accuracy comparison")
        comparison = get_comparison_metrics().set_index("Model")
        st.bar_chart(comparison[["accuracy"]], color="#9bcf53")

    history = pd.DataFrame(st.session_state.history)
    st.subheader("Prediction history")
    if history.empty:
        st.info("Prediction history will appear after you analyze at least one image.")
    else:
        history["timestamp"] = pd.to_datetime(history["timestamp"])
        history = history.sort_values("timestamp")
        chart_data = history.set_index("timestamp")[["confidence", "health_score"]]
        st.line_chart(chart_data, color=["#2f855a", "#d69e2e"])
        st.dataframe(history.iloc[::-1], use_container_width=True, hide_index=True)

    st.subheader("Feature influence")
    importance = feature_importance(bundle)
    if importance.empty:
        st.info(
            "Native feature importance is unavailable for the selected nonlinear SVM. "
            "Choose Random Forest or Logistic Regression to inspect model influence."
        )
    else:
        st.bar_chart(importance.set_index("feature"), color="#2f855a")


def render_dataset_management(model_name: str) -> None:
    st.markdown('<div class="section-kicker">DATA OPERATIONS</div>', unsafe_allow_html=True)
    st.title("Dataset management")
    st.write(
        "Upload a CSV containing the extracted feature columns and a binary `label` "
        "column: `0` for Healthy and `1` for Diseased."
    )

    template = create_synthetic_dataset(30)
    st.download_button(
        "Download dataset template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="plant_feature_dataset_template.csv",
        mime="text/csv",
    )

    dataset_file = st.file_uploader("Upload custom feature dataset", type=["csv"])
    if dataset_file is not None:
        try:
            uploaded_data = pd.read_csv(dataset_file)
            clean_data = validate_dataset(uploaded_data)
            st.success(f"Validated {len(clean_data):,} rows.")

            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", f"{len(clean_data):,}")
            col2.metric("Features", len(FEATURE_NAMES))
            col3.metric("Disease rate", f"{clean_data[LABEL_COLUMN].mean():.1%}")
            st.dataframe(clean_data.head(25), use_container_width=True, hide_index=True)

            with st.expander("Dataset statistics"):
                st.dataframe(clean_data.describe().T, use_container_width=True)

            if st.button(f"Train {model_name} on this dataset", type="primary"):
                with st.spinner(f"Training {model_name}..."):
                    bundle = train_and_evaluate(clean_data, model_name)
                    bundle["source"] = dataset_file.name
                    destination = model_path(MODELS_DIR, model_name)
                    save_bundle(bundle, destination)
                    st.session_state.custom_bundles[model_name] = bundle
                st.success(f"{model_name} was trained and saved successfully.")
        except (ValueError, pd.errors.ParserError) as error:
            st.error(f"Dataset could not be used: {error}")

    st.divider()
    st.subheader("Prediction report")
    history = pd.DataFrame(st.session_state.history)
    if history.empty:
        st.caption("Analyze an image to create downloadable prediction records.")
    else:
        st.download_button(
            "Download prediction history CSV",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="plant_prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_evaluation(model_name: str) -> None:
    st.markdown('<div class="section-kicker">MODEL QUALITY</div>', unsafe_allow_html=True)
    st.title("Model evaluation")
    bundle = active_bundle(model_name)
    metrics = bundle["metrics"]

    columns = st.columns(4)
    columns[0].metric("Accuracy", f"{metrics['accuracy']:.1%}")
    columns[1].metric("Precision", f"{metrics['precision']:.1%}")
    columns[2].metric("Recall", f"{metrics['recall']:.1%}")
    columns[3].metric("F1 score", f"{metrics['f1']:.1%}")

    st.caption(
        f"Model: {model_name} · Training rows: {bundle['training_rows']:,} · "
        f"Source: {bundle.get('source', 'unknown')}"
    )

    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    image = ax.imshow(matrix, cmap="Greens")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, matrix[row, column], ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], labels=["Healthy", "Diseased"])
    ax.set_yticks([0, 1], labels=["Healthy", "Diseased"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("Confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    comparison = get_comparison_metrics()
    st.subheader("Baseline model comparison")
    display = comparison[["Model", "accuracy", "precision", "recall", "f1"]].copy()
    for metric in ("accuracy", "precision", "recall", "f1"):
        display[metric] = display[metric].map(lambda value: f"{value:.1%}")
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.warning(
        "Baseline metrics use a generated demonstration dataset. Upload a representative, "
        "labeled dataset before using this application for real agricultural decisions."
    )


apply_styles()
initialize_state()

with st.sidebar:
    st.markdown("## 🌿 Plant Health")
    st.caption("Traditional ML leaf screening")
    page = st.radio(
        "Navigation",
        ["Home", "Diagnose", "Analytics", "Dataset", "Model Evaluation"],
        label_visibility="collapsed",
    )
    st.divider()
    model_name = st.selectbox("Prediction model", MODEL_NAMES)
    bundle = active_bundle(model_name)
    st.caption(f"Active source: {bundle.get('source', 'saved model')}")
    st.caption(f"Validation accuracy: {bundle['metrics']['accuracy']:.1%}")
    st.divider()
    st.caption("Built with Streamlit, OpenCV, and scikit-learn.")

if page == "Home":
    render_landing()
elif page == "Diagnose":
    render_diagnosis(model_name)
elif page == "Analytics":
    render_analytics(model_name)
elif page == "Dataset":
    render_dataset_management(model_name)
else:
    render_evaluation(model_name)
