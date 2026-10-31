"""BlindSpy demo: classify a flow file with a final model and explain one prediction.

Choose a task, pick one of the 5 files in app/samples/ or upload a CSV of flows, and see the
predicted label and its confidence for every flow (blindspy.predict, the same code as
scripts/predict.py). Click a row to see the 10 features that pushed the model most towards its
label (SHAP, blindspy.explain). Only models/final/ is used; nothing is trained here.

Run: uv run streamlit run app/demo.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from blindspy import explain
from blindspy import predict as P
from blindspy.data import CORE_TASKS

SAMPLES = Path(__file__).parent / "samples"
TASKS = {
    "A1": "A1: VPN or non-VPN (2 classes)",
    "A2-VPN": "A2-VPN: traffic type of VPN flows (7 classes)",
    "A2-NonVPN": "A2-NonVPN: traffic type of non-VPN flows (7 classes)",
    "B": "B: VPN or not and traffic type together (14 classes)",
}


def sample_files() -> list[str]:
    return sorted(p.name for p in SAMPLES.glob("*.csv"))


@st.cache_resource
def model_for(task: str):
    return P.load_final(task)


def explain_row(task: str, df: pd.DataFrame, pos: int, label: str, k: int = 10) -> pd.DataFrame:
    """The k features with the largest SHAP value towards `label` for row `pos` of df."""
    model, meta = model_for(task)
    X = P.validate(df, meta).iloc[[pos]]
    values, classes = explain.shap_values(model, X)
    return explain.example_rows(values, classes, X, 0, label, k)


def shap_chart(ex: pd.DataFrame, meta: dict, label: str):
    units = {f["name"]: f["unit"] for f in meta["features"]}
    ex = ex.iloc[::-1]
    names = [f"{f} = -1 (not measurable)" if v == -1 else f"{f} = {v:,.6g} {units[f]}"
             for f, v in zip(ex["feature"], ex["value"], strict=True)]  # fmt: skip
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(ex) + 0.8))
    ax.barh(names, ex["SHAP"], color=["#2f6db5" if s >= 0 else "#b5562f" for s in ex["SHAP"]])
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_xlabel(f"SHAP value towards {label} (log-odds)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="BlindSpy demo", layout="wide")
    st.title("BlindSpy")
    st.caption("Classifying encrypted traffic (ISCXVPN2016) from 23 flow timing features, "
               "never the payload")  # fmt: skip

    def task_changed():
        st.session_state.sample = f"{st.session_state.task}_test_sample.csv"

    with st.sidebar:
        task = st.selectbox("Task", CORE_TASKS, format_func=TASKS.get, key="task",
                            on_change=task_changed)  # fmt: skip
        source = st.radio("Flows", ["Sample file", "Upload a CSV"], key="source")
        if source == "Sample file":
            names = sample_files()
            if "sample" not in st.session_state:
                st.session_state.sample = f"{task}_test_sample.csv"
            name = st.selectbox("Sample file (app/samples/)", names, key="sample")
            file = SAMPLES / name
        else:
            file = st.file_uploader("CSV with the task's 23 features (a label column is optional)",
                                    type="csv")  # fmt: skip
            name = getattr(file, "name", None)

    model, meta = model_for(task)
    st.markdown(f"**Model:** {meta['mlflow_run_name']} ({meta['model']}, trained on "
                f"{meta['trained_on']}); test macro F1 {meta['test_macro_f1']:.4f}. "
                f"**Confidence:** {meta['confidence']}.")  # fmt: skip
    if file is None:
        st.info("Upload a CSV of flows, or choose a sample file in the sidebar.")
        return

    try:
        df = pd.read_csv(file)
        out = P.predict(task, df)
    except (P.InputError, pd.errors.ParserError, pd.errors.EmptyDataError,
            UnicodeDecodeError) as e:  # fmt: skip
        st.error(f"{name} was rejected: {e}")
        return

    if "true_label" in out:
        right = int((out["label"].astype(str) == out["true_label"].astype(str)).sum())
        st.markdown(f"**{name}:** {len(out)} flows, {right} of {len(out)} predicted correctly "
                    "(the file has a label column).")  # fmt: skip
    else:
        st.markdown(f"**{name}:** {len(out)} flows (no label column).")

    event = st.dataframe(
        out, hide_index=True, on_select="rerun", selection_mode="single-row", key="table",
        column_config={"confidence": st.column_config.ProgressColumn(
            "confidence", format="%.4f", min_value=0.0, max_value=1.0)},
    )  # fmt: skip
    rows = event.selection.rows
    if not rows:
        st.info("Click a row to see which features pushed the model towards its label.")
        return
    pos = rows[0]
    label = str(out["label"].iloc[pos])
    st.subheader(f"Row {pos + 1}: predicted {label}, confidence {out['confidence'].iloc[pos]:.4f}")
    st.pyplot(shap_chart(explain_row(task, df, pos, label), meta, label))
    st.caption("Blue bars push towards the predicted label, orange bars away from it. Many "
               "features are correlated, so this shows how the model uses them, not what "
               "causes the traffic pattern.")  # fmt: skip


if __name__ == "__main__":
    main()
