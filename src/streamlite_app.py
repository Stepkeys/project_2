from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
TARGET = "addicted_label"
RANDOM_STATE = 42


@st.cache_data
def load_training_data(path, modified_at):
    return pd.read_csv(path)


def get_features(training_data):
    if TARGET not in training_data.columns:
        raise ValueError(f"Training data must include {TARGET!r}.")
    features = [
        column for column in training_data.columns
        if column not in {"id", TARGET}
    ]
    if not features:
        raise ValueError("No feature columns remain after excluding id and target.")
    if training_data[TARGET].isna().any():
        raise ValueError(f"Training target {TARGET!r} contains missing values.")
    if training_data[TARGET].nunique() != 2:
        raise ValueError("Training data must contain exactly two target classes.")
    return features


def prepare_features(data, features, categorical_columns, category_levels):
    prepared = data[features].copy()
    for column in categorical_columns:
        values = prepared[column].fillna("Missing").astype(str)
        prepared[column] = pd.Categorical(
            values, categories=category_levels[column]
        )
    return prepared


def make_model():
    return HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        categorical_features="from_dtype",
        early_stopping=True,
        n_iter_no_change=20,
        random_state=RANDOM_STATE,
    )


@st.cache_resource
def train_model(path, modified_at):
    training_data = load_training_data(path, modified_at)
    features = get_features(training_data)
    classes = sorted(training_data[TARGET].unique().tolist())
    target = training_data[TARGET].map(
        {label: index for index, label in enumerate(classes)}
    ).astype("int8")
    categorical_columns = training_data[features].select_dtypes(
        include=["object", "str", "category", "bool"]
    ).columns.tolist()
    category_levels = {
        column: pd.Index(
            training_data[column].fillna("Missing").astype(str).unique()
        )
        for column in categorical_columns
    }
    prepared = prepare_features(
        training_data, features, categorical_columns, category_levels
    )
    model = make_model()
    model.fit(prepared, target)
    return model, features, categorical_columns, category_levels, classes


@st.cache_data
def evaluate_validation(path, modified_at):
    validation_data = load_training_data(path, modified_at)
    features = get_features(validation_data)
    classes = sorted(validation_data[TARGET].unique().tolist())
    target = validation_data[TARGET].map(
        {label: index for index, label in enumerate(classes)}
    ).astype("int8")
    categorical_columns = validation_data[features].select_dtypes(
        include=["object", "str", "category", "bool"]
    ).columns.tolist()
    train_rows, validation_rows = train_test_split(
        validation_data.index,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    category_levels = {
        column: pd.Index(
            validation_data.loc[train_rows, column].fillna("Missing").astype(str).unique()
        )
        for column in categorical_columns
    }
    train_features = prepare_features(
        validation_data.loc[train_rows], features, categorical_columns, category_levels
    )
    holdout_features = prepare_features(
        validation_data.loc[validation_rows], features, categorical_columns, category_levels
    )
    model = make_model()
    model.fit(train_features, target.loc[train_rows])
    probability = model.predict_proba(holdout_features)[:, 1]
    prediction = (probability >= 0.5).astype("int8")
    truth = target.loc[validation_rows]
    return (
        roc_auc_score(truth, probability),
        accuracy_score(truth, prediction),
    )


def main():
    st.set_page_config(page_title="Smartphone Addiction Estimate", layout="wide")
    st.title("Smartphone Addiction Estimate")
    st.caption(
        "A model-based estimate from phone-use data. This is not a clinical diagnosis."
    )

    training_path = RAW_DATA_DIR / "train.csv"
    if not training_path.exists():
        st.error("Place train.csv in data/raw to train the model.")
        st.stop()

    modified_at = training_path.stat().st_mtime_ns
    try:
        model, features, categorical_columns, category_levels, classes = train_model(
            str(training_path), modified_at
        )
    except Exception as error:
        st.error(f"Could not train the model: {error}")
        st.stop()

    st.sidebar.header("Prediction data")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            records = pd.read_csv(uploaded_file)
        except Exception as error:
            st.error(f"Could not read the uploaded CSV: {error}")
            st.stop()
        input_name = uploaded_file.name
    else:
        sample_path = RAW_DATA_DIR / "test.csv"
        if not sample_path.exists():
            st.info("Upload a CSV containing the model feature columns to begin.")
            st.stop()
        records = pd.read_csv(sample_path)
        input_name = "test.csv sample"
        st.sidebar.caption("Showing predictions for the included test.csv by default.")

    missing_features = sorted(set(features) - set(records.columns))
    if missing_features:
        st.error(f"The input is missing required columns: {', '.join(missing_features)}")
        st.stop()
    if records.empty:
        st.warning("The input CSV has no rows to predict.")
        st.stop()

    prepared = prepare_features(
        records, features, categorical_columns, category_levels
    )
    positive_probability = model.predict_proba(prepared)[:, 1]
    results = records.copy()
    results["predicted_label"] = [
        classes[int(probability >= 0.5)] for probability in positive_probability
    ]
    results["addiction_probability"] = positive_probability

    first, second, third = st.columns(3)
    first.metric("Rows scored", f"{len(results):,}")
    second.metric(
        "Estimated positive rate",
        f"{(positive_probability >= 0.5).mean():.1%}",
    )
    third.metric("Positive class", str(classes[1]))

    st.subheader(f"Predictions: {input_name}")
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.download_button(
        "Download predictions as CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="smartphone_addiction_predictions.csv",
        mime="text/csv",
    )

    st.subheader("Probability distribution")
    probability_counts, probability_edges = np.histogram(
        positive_probability, bins=np.linspace(0, 1, 11)
    )
    probability_labels = [
        f"{lower:.1f}-{upper:.1f}"
        for lower, upper in zip(probability_edges[:-1], probability_edges[1:])
    ]
    st.bar_chart(
        pd.DataFrame({"Rows": probability_counts}, index=probability_labels)
    )

    validation_roc_auc, validation_accuracy = evaluate_validation(
        str(training_path), modified_at
    )
    st.subheader("Held-out validation")
    metric_one, metric_two = st.columns(2)
    metric_one.metric("ROC-AUC", f"{validation_roc_auc:.3f}")
    metric_two.metric("Accuracy at 0.5", f"{validation_accuracy:.3f}")


if __name__ == "__main__":
    main()