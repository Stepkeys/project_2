from pathlib import Path
import io
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
TARGET = "addicted_label"
RANDOM_STATE = 42

def main():
    st.set_page_config(
        page_title="Smartphone Addiction Insights",
        page_icon="📱",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        :root { --ink:#edf2ff; --muted:#9aa8c2; --panel:#151e31; --line:#293650; }
        .block-container {padding-top:1.8rem; padding-bottom:3rem; max-width:1500px;}
        [data-testid="stAppViewContainer"] {background:radial-gradient(circle at 85% 0%, #1d3150 0, #0b1220 42%, #0b1220 100%);}
        [data-testid="stSidebar"] {background:#101a2b; border-right:1px solid #26354e;}
        .hero {padding:1.5rem 1.7rem; border:1px solid #2c4263; border-radius:8px;
               background:linear-gradient(120deg,rgba(32,65,103,.88),rgba(21,30,49,.94)); margin-bottom:1.2rem;}
        .hero h1 {font-size:2.15rem; margin:0 0 .35rem 0; color:#f3f7ff;}
        .hero p {color:#b7c6df; margin:0; font-size:1rem;}
        .section-note {color:#9aa8c2; font-size:.9rem;}
        div[data-testid="stMetric"] {background:#151e31; border:1px solid #293650; padding:14px 16px; border-radius:8px;}
        .stTabs [data-baseweb="tab-list"] {gap:8px;}
        .stTabs [data-baseweb="tab"] {background:#151e31; border-radius:8px 8px 0 0; padding:10px 18px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


    def read_uploaded_or_local(uploaded, local_path):
        if uploaded is not None:
            return pd.read_csv(uploaded)
        path = Path(local_path)
        if not path.is_absolute():
            path = PROJECT_DIR / path
        if path.exists():
            return pd.read_csv(path)
        return None


    @st.cache_data(show_spinner=False)
    def build_data_example_zip(train_modified_at, test_modified_at):
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(
            archive_buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
        ) as archive:
            archive.write(RAW_DATA_DIR / "train.csv", arcname="train.csv")
            archive.write(RAW_DATA_DIR / "test.csv", arcname="test.csv")
        return archive_buffer.getvalue()


    def make_encoder():
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=True)


    @st.cache_resource(show_spinner="Training and validating the models...")
    def train_model(csv_bytes, target, test_size, seed):
        df = pd.read_csv(io.BytesIO(csv_bytes))
        if target not in df.columns:
            raise ValueError(f"Target column {target!r} was not found.")
        if df[target].isna().any():
            raise ValueError("The target column contains missing values. Clean these rows first.")
        if df[target].nunique() != 2:
            raise ValueError("This dashboard supports a binary target with exactly two classes.")

        features = [column for column in df.columns if column not in {target, "id"}]
        if not features:
            raise ValueError("No usable feature columns were found.")

        X = df[features].copy()
        classes = sorted(df[target].unique().tolist(), key=lambda value: str(value))
        class_mapping = {classes[0]: 0, classes[1]: 1}
        y = df[target].map(class_mapping).astype("int8")

        numeric = X.select_dtypes(include=np.number).columns.tolist()
        categorical = [column for column in features if column not in numeric]
        transformers = []
        if numeric:
            transformers.append((
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scale", StandardScaler()),
                ]),
                numeric,
            ))
        if categorical:
            transformers.append((
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                    ("onehot", make_encoder()),
                ]),
                categorical,
            ))
        preprocessor = ColumnTransformer(transformers, remainder="drop")

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        candidates = {
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=18,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            ),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=100,
                max_depth=18,
                min_samples_leaf=3,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            ),
        }

        results = []
        validation_probabilities = {}
        for name, estimator in candidates.items():
            pipeline = Pipeline([
                ("prep", clone(preprocessor)),
                ("model", clone(estimator)),
            ])
            pipeline.fit(X_train, y_train)
            probability = pipeline.predict_proba(X_valid)[:, 1]
            prediction = (probability >= 0.5).astype("int8")
            results.append({
                "Model": name,
                "ROC-AUC": roc_auc_score(y_valid, probability),
                "PR-AUC": average_precision_score(y_valid, probability),
                "Accuracy": accuracy_score(y_valid, prediction),
                "Balanced accuracy": balanced_accuracy_score(y_valid, prediction),
                "Precision": precision_score(y_valid, prediction, zero_division=0),
                "Recall": recall_score(y_valid, prediction, zero_division=0),
                "F1": f1_score(y_valid, prediction, zero_division=0),
            })
            validation_probabilities[name] = probability

        comparison = (
            pd.DataFrame(results)
            .sort_values("ROC-AUC", ascending=False)
            .reset_index(drop=True)
        )
        best_name = comparison.iloc[0]["Model"]
        final_pipeline = Pipeline([
            ("prep", clone(preprocessor)),
            ("model", clone(candidates[best_name])),
        ])
        final_pipeline.fit(X, y)

        return {
            "df": df,
            "features": features,
            "classes": classes,
            "mapping": class_mapping,
            "numeric": numeric,
            "categorical": categorical,
            "comparison": comparison,
            "best_name": best_name,
            "best_metrics": comparison.iloc[0].to_dict(),
            "validation_y": y_valid,
            "validation_probability": validation_probabilities[best_name],
            "validation_prediction": (validation_probabilities[best_name] >= 0.5).astype("int8"),
            "final_model": final_pipeline,
        }


    st.sidebar.title("📱 Dashboard controls")
    st.sidebar.caption("Load labelled training data and, optionally, an unlabelled test file.")
    train_upload = st.sidebar.file_uploader("Training CSV", type=["csv"], key="train")
    test_upload = st.sidebar.file_uploader("Test CSV (optional)", type=["csv"], key="test")
    train_example_path = RAW_DATA_DIR / "train.csv"
    test_example_path = RAW_DATA_DIR / "test.csv"
    if train_example_path.is_file() and test_example_path.is_file():
        data_example_zip = build_data_example_zip(
            train_example_path.stat().st_mtime_ns,
            test_example_path.stat().st_mtime_ns,
        )
        st.sidebar.download_button(
            "Download DATA_EXAMPLE",
            data=data_example_zip,
            file_name="DATA_EXAMPLE.zip",
            mime="application/zip",
            width="stretch",
        )
    target_col = st.sidebar.text_input("Target column", TARGET)
    validation_size = st.sidebar.slider("Validation split", 0.10, 0.35, 0.20, 0.05)
    seed = st.sidebar.number_input("Random seed", min_value=0, max_value=100000, value=42)
    st.sidebar.markdown("---")
    st.sidebar.caption("The estimate is a machine-learning output, not a clinical diagnosis.")

    try:
        train_df = read_uploaded_or_local(train_upload, RAW_DATA_DIR / "train.csv")
        test_df = read_uploaded_or_local(test_upload, RAW_DATA_DIR / "test.csv")
    except Exception as error:
        st.error(f"Could not read a CSV file: {error}")
        st.stop()

    st.markdown(
        """
        <div class="hero">
          <h1>📱 Smartphone Addiction Insights</h1>
          <p>Explore phone-use patterns, validate machine-learning models, and estimate the probability of the positive class.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if train_df is None:
        st.info("Upload a labelled training CSV or place train.csv in data/raw.")
        st.markdown("**Required:** a binary target column (default: `addicted_label`). An optional `id` column is excluded from model training.")
        st.stop()

    if target_col not in train_df.columns:
        st.error(
            f"Target column {target_col!r} was not found. "
            f"Available columns: {', '.join(map(str, train_df.columns))}"
        )
        st.stop()

    try:
        model_data = train_model(
            train_df.to_csv(index=False).encode("utf-8"),
            target_col,
            validation_size,
            int(seed),
        )
    except Exception as error:
        st.error(f"Could not prepare the model: {error}")
        st.stop()

    df = model_data["df"]
    comparison = model_data["comparison"]
    metrics = model_data["best_metrics"]
    positive_label = model_data["classes"][1]
    positive_rate = (df[target_col] == positive_label).mean()

    metric_columns = st.columns(4)
    metric_columns[0].metric("Training records", f"{len(df):,}")
    metric_columns[1].metric("Features used", f"{len(model_data['features']):,}")
    metric_columns[2].metric("Positive class rate", f"{positive_rate:.1%}")
    metric_columns[3].metric("Best validation ROC-AUC", f"{metrics['ROC-AUC']:.3f}")

    tab_overview, tab_explore, tab_models, tab_predict = st.tabs(
        ["Overview", "Data explorer", "Model performance", "Predictions"]
    )

    with tab_overview:
        st.subheader("Dataset overview")
        overview_left, overview_right = st.columns([1.15, 1])
        with overview_left:
            counts = (
                df[target_col].astype(str).value_counts()
                .rename_axis("Class").reset_index(name="Records")
            )
            figure = px.pie(
                counts,
                names="Class",
                values="Records",
                hole=0.62,
                title="Target class distribution",
                color_discrete_sequence=["#55c2a3", "#6b8de3"],
            )
            figure.update_layout(margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(figure, use_container_width=True)
        with overview_right:
            missing = (
                df[model_data["features"]].isna().mean().mul(100)
                .sort_values(ascending=False).head(12)
            )
            missing_data = missing.rename_axis("Feature").reset_index(name="Missing (%)")
            figure = px.bar(
                missing_data.sort_values("Missing (%)"),
                x="Missing (%)",
                y="Feature",
                orientation="h",
                title="Features with most missing values",
                color="Missing (%)",
                color_continuous_scale="Teal",
            )
            figure.update_layout(margin=dict(l=10, r=10, t=55, b=10), coloraxis_showscale=False)
            st.plotly_chart(figure, use_container_width=True)

        st.subheader("Dataset quality")
        quality_columns = st.columns(3)
        quality_columns[0].metric("Duplicate rows", f"{df.duplicated().sum():,}")
        quality_columns[1].metric(
            "Missing feature cells",
            f"{int(df[model_data['features']].isna().sum().sum()):,}",
        )
        quality_columns[2].metric("Categorical features", f"{len(model_data['categorical']):,}")
        st.caption(f"Positive class: **{positive_label}**. Class mapping: {model_data['mapping']}.")

    with tab_explore:
        st.subheader("Explore the training data")
        numeric_columns = model_data["numeric"]
        if numeric_columns:
            selected_feature = st.selectbox("Choose a numeric feature", numeric_columns)
            figure = px.histogram(
                df,
                x=selected_feature,
                color=df[target_col].astype(str),
                marginal="box",
                barmode="overlay",
                opacity=0.72,
                title=f"Distribution of {selected_feature} by target class",
                color_discrete_sequence=["#6b8de3", "#55c2a3"],
            )
            st.plotly_chart(figure, use_container_width=True)
        else:
            st.info("No numeric features were detected.")
        st.markdown("**Preview**")
        st.dataframe(df.head(100), use_container_width=True, hide_index=True)
        st.download_button(
            "Download training-data preview",
            df.head(1000).to_csv(index=False).encode(),
            "training_preview.csv",
            "text/csv",
        )

    with tab_models:
        st.subheader("Validation model comparison")
        st.caption("Models share a stratified hold-out split. Selection is based on ROC-AUC.")
        st.dataframe(
            comparison.style.format({
                column: "{:.4f}" for column in comparison.columns if column != "Model"
            }),
            use_container_width=True,
            hide_index=True,
        )
        figure = px.bar(
            comparison,
            x="Model",
            y="ROC-AUC",
            text_auto=".4f",
            title="ROC-AUC by model",
            color="ROC-AUC",
            color_continuous_scale="Teal",
        )
        figure.update_layout(coloraxis_showscale=False)
        st.plotly_chart(figure, use_container_width=True)
        confusion = confusion_matrix(
            model_data["validation_y"], model_data["validation_prediction"]
        )
        figure = px.imshow(
            confusion,
            text_auto=True,
            x=["Predicted 0", "Predicted 1"],
            y=["Actual 0", "Actual 1"],
            color_continuous_scale="Blues",
            title="Confusion matrix at probability threshold 0.50",
        )
        st.plotly_chart(figure, use_container_width=True)
        st.success(
            f"Selected model: {model_data['best_name']}; "
            f"validation ROC-AUC {metrics['ROC-AUC']:.4f}, PR-AUC {metrics['PR-AUC']:.4f}."
        )
        st.caption("Hold-out metrics do not guarantee performance on future data.")

    with tab_predict:
        st.subheader("Generate predictions")
        st.write(f"Selected model: **{model_data['best_name']}**")
        if test_df is not None:
            missing = sorted(set(model_data["features"]) - set(test_df.columns))
            if missing:
                st.warning(f"The test CSV is missing required columns: {missing}")
            else:
                probabilities = model_data["final_model"].predict_proba(
                    test_df[model_data["features"]]
                )[:, 1]
                output = pd.DataFrame({
                    "estimated_positive_probability": probabilities,
                    "predicted_class": np.where(
                        probabilities >= 0.5,
                        positive_label,
                        model_data["classes"][0],
                    ),
                })
                if "id" in test_df.columns:
                    output.insert(0, "id", test_df["id"].values)
                prediction_columns = st.columns(2)
                prediction_columns[0].metric("Rows scored", f"{len(output):,}")
                prediction_columns[1].metric(
                    "Predicted positive rate", f"{(probabilities >= 0.5).mean():.1%}"
                )
                st.dataframe(output.head(100), use_container_width=True, hide_index=True)
                st.download_button(
                    "Download all predictions",
                    output.to_csv(index=False).encode(),
                    "smartphone_addiction_predictions.csv",
                    "text/csv",
                    type="primary",
                )
        else:
            st.info("Upload test.csv in the sidebar to score a full dataset.")

        st.markdown("---")
        st.markdown("**Single-record estimate**")
        st.caption("Enter values using the same feature definitions and units as the training data.")
        with st.form("single_prediction"):
            values = {}
            form_columns = st.columns(2)
            for index, feature in enumerate(model_data["features"]):
                with form_columns[index % 2]:
                    if feature in model_data["numeric"]:
                        series = pd.to_numeric(df[feature], errors="coerce").dropna()
                        default = float(series.median()) if len(series) else 0.0
                        values[feature] = st.number_input(
                            feature.replace("_", " ").title(),
                            value=default,
                            key=f"single_{feature}",
                        )
                    else:
                        options = df[feature].dropna().astype(str).value_counts().index.tolist()
                        options = options[:100] if options else ["Missing"]
                        values[feature] = st.selectbox(
                            feature.replace("_", " ").title(),
                            options,
                            key=f"single_{feature}",
                        )
            submitted = st.form_submit_button("Estimate probability", type="primary")

        if submitted:
            record = pd.DataFrame([values], columns=model_data["features"])
            probability = float(model_data["final_model"].predict_proba(record)[:, 1][0])
            predicted = positive_label if probability >= 0.5 else model_data["classes"][0]
            result_columns = st.columns(2)
            result_columns[0].metric("Estimated positive-class probability", f"{probability:.1%}")
            result_columns[1].metric("Predicted class", str(predicted))
            st.progress(probability)
            st.caption("This is a model-based estimate, not a medical assessment or diagnosis.")

    st.markdown("---")
    st.markdown(
        '<p class="section-note">Smartphone Addiction Insights · Research and educational use · Not a clinical diagnostic tool.</p>',
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
