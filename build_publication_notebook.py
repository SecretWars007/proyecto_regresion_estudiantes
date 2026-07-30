import json
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone

import shap
from pdpbox import pdp

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "StudentsPerformance.csv"
PLOTS_DIR = ROOT / "plots"
MODELS_DIR = ROOT / "models"
PLOTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Load and clean data
raw_df = pd.read_csv(DATA_PATH)
for col in [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
]:
    raw_df[col] = raw_df[col].astype(str).str.strip()

# Prepare target and feature sets
analysis_df = raw_df.copy()
analysis_df["test_prep_completed"] = (
    analysis_df["test preparation course"] == "completed"
).astype(int)

target = "math score"
scenario_a_features = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
]
scenario_b_features = scenario_a_features + ["reading score", "writing score"]

# EDA plots
sns.set_theme(style="whitegrid")

# Score distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for ax, col, color in zip(
    axes,
    ["math score", "reading score", "writing score"],
    ["steelblue", "seagreen", "indianred"],
):
    sns.histplot(analysis_df[col], kde=True, ax=ax, color=color)
    ax.axvline(
        analysis_df[col].mean(),
        color="red",
        linestyle="--",
        label=f"Media: {analysis_df[col].mean():.1f}",
    )
    ax.set_title(col)
    ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "distribucion_puntajes.png", dpi=200)
plt.close()

# Correlation heatmap
plt.figure(figsize=(6, 5))
corr = analysis_df[["math score", "reading score", "writing score"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=0, vmax=1)
plt.title("Correlación entre puntajes")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "matriz_correlacion.png", dpi=200)
plt.close()

# Boxplots by category
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.boxplot(
    data=analysis_df, x="gender", y="math score", ax=axes[0, 0], palette="pastel"
)
axes[0, 0].set_title("Math Score por Género")
sns.boxplot(
    data=analysis_df, x="lunch", y="math score", ax=axes[0, 1], palette="pastel"
)
axes[0, 1].set_title("Math Score por Almuerzo")
sns.boxplot(
    data=analysis_df,
    x="test preparation course",
    y="math score",
    ax=axes[1, 0],
    palette="pastel",
)
axes[1, 0].set_title("Math Score por Curso de Preparación")
order = (
    analysis_df.groupby("parental level of education")["math score"]
    .mean()
    .sort_values()
    .index
)
sns.boxplot(
    data=analysis_df,
    x="parental level of education",
    y="math score",
    order=order,
    ax=axes[1, 1],
    palette="pastel",
)
axes[1, 1].set_title("Math Score por Nivel Educativo de los Padres")
axes[1, 1].tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "boxplots_variables_categoricas.png", dpi=200)
plt.close()

# Scenario comparison with GroupKFold
categorical_features = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
]


def make_preprocessor(features):
    numeric_features = [f for f in ["reading score", "writing score"] if f in features]
    transformers = [
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


models = {
    "DummyRegressor": DummyRegressor(strategy="mean"),
    "RandomForestRegressor": RandomForestRegressor(
        n_estimators=400, random_state=42, n_jobs=-1
    ),
}


def evaluate_scenario(features, scenario_name):
    X = analysis_df[features].copy()
    y = analysis_df[target]
    groups = analysis_df["race/ethnicity"]
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X, y, groups, test_size=0.2, random_state=42
    )
    gkf = GroupKFold(n_splits=5)
    rows = []
    for model_name, base_model in models.items():
        fold_scores = []
        for train_idx, val_idx in gkf.split(X_train, y_train, groups_train):
            model = clone(base_model)
            preprocessor = make_preprocessor(features)
            pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
            X_tr = X_train.iloc[train_idx]
            X_va = X_train.iloc[val_idx]
            y_tr = y_train.iloc[train_idx]
            y_va = y_train.iloc[val_idx]
            pipeline.fit(X_tr, y_tr)
            pred = pipeline.predict(X_va)
            fold_scores.append(
                {
                    "rmse": np.sqrt(mean_squared_error(y_va, pred)),
                    "mae": mean_absolute_error(y_va, pred),
                    "r2": r2_score(y_va, pred),
                }
            )
        fold_df = pd.DataFrame(fold_scores)
        rows.append(
            {
                "scenario": scenario_name,
                "model": model_name,
                "rmse_mean": fold_df["rmse"].mean(),
                "rmse_std": fold_df["rmse"].std(),
                "mae_mean": fold_df["mae"].mean(),
                "mae_std": fold_df["mae"].std(),
                "r2_mean": fold_df["r2"].mean(),
                "r2_std": fold_df["r2"].std(),
            }
        )
    return pd.DataFrame(rows)


results_df = pd.concat(
    [
        evaluate_scenario(scenario_a_features, "Escenario A"),
        evaluate_scenario(scenario_b_features, "Escenario B"),
    ],
    ignore_index=True,
)
results_df = results_df.sort_values(["scenario", "rmse_mean"]).reset_index(drop=True)

# Plot scenario comparison
plt.figure(figsize=(10, 5))
ax = sns.barplot(
    data=results_df, x="scenario", y="rmse_mean", hue="model", palette="viridis"
)
ax.set_title("Comparación de RMSE por escenario y modelo")
ax.set_ylabel("RMSE promedio (CV)")
ax.set_xlabel("Escenario")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "comparacion_scenarios_rmse.png", dpi=200)
plt.close()

plt.figure(figsize=(10, 5))
ax = sns.barplot(
    data=results_df, x="scenario", y="r2_mean", hue="model", palette="magma"
)
ax.set_title("Comparación de R² por escenario y modelo")
ax.set_ylabel("R² promedio (CV)")
ax.set_xlabel("Escenario")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "comparacion_scenarios_r2.png", dpi=200)
plt.close()

# Train selected model on Scenario B
best_scenario = "Escenario B"
X = analysis_df[scenario_b_features].copy()
y = analysis_df[target]
groups = analysis_df["race/ethnicity"]
X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X, y, groups, test_size=0.2, random_state=42
)

final_pipeline = Pipeline(
    [
        ("preprocess", make_preprocessor(scenario_b_features)),
        ("model", RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)),
    ]
)
final_pipeline.fit(X_train, y_train)
pred_test = final_pipeline.predict(X_test)

holdout_metrics = {
    "rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
    "mae": float(mean_absolute_error(y_test, pred_test)),
    "r2": float(r2_score(y_test, pred_test)),
}

# Fairness analysis
fairness_rows = []
for group_col in ["gender", "race/ethnicity", "lunch"]:
    group_values = analysis_df.loc[X_test.index, group_col]
    for group in sorted(group_values.unique()):
        mask = group_values == group
        y_true = y_test[mask]
        y_pred = pred_test[mask]
        fairness_rows.append(
            {
                "group_col": group_col,
                "group": group,
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "count": int(mask.sum()),
            }
        )
fairness_df = pd.DataFrame(fairness_rows)

for group_col in ["gender", "race/ethnicity", "lunch"]:
    plot_df = fairness_df[fairness_df["group_col"] == group_col].copy()
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=plot_df, x="group", y="rmse", palette="coolwarm")
    ax.set_title(f"RMSE por {group_col}")
    ax.set_ylabel("RMSE")
    ax.set_xlabel(group_col)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"fairness_rmse_{group_col.replace('/', '_')}.png", dpi=200)
    plt.close()

# Feature importance, permutation importance, SHAP, PDP
preprocessed_train = final_pipeline.named_steps["preprocess"].fit_transform(X_train)
preprocessed_test = final_pipeline.named_steps["preprocess"].transform(X_test)
feature_names = final_pipeline.named_steps["preprocess"].get_feature_names_out()
rf_model = final_pipeline.named_steps["model"]
rf_model.fit(preprocessed_train, y_train)

importances = (
    pd.Series(rf_model.feature_importances_, index=feature_names)
    .sort_values(ascending=False)
    .head(15)
)
importances.to_frame("importance").to_csv(
    PLOTS_DIR / "feature_importance.csv", index=True
)
plt.figure(figsize=(10, 5))
importances.plot(kind="bar", color="steelblue")
plt.title("Feature Importance (Random Forest)")
plt.ylabel("Importancia")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=200)
plt.close()

perm = permutation_importance(
    rf_model, preprocessed_test, y_test, n_repeats=20, random_state=42, n_jobs=-1
)
perm_df = (
    pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    )
    .sort_values("importance_mean", ascending=False)
    .head(15)
)
perm_df.to_csv(PLOTS_DIR / "permutation_importance.csv", index=False)
plt.figure(figsize=(10, 5))
plt.barh(
    perm_df["feature"].iloc[::-1],
    perm_df["importance_mean"].iloc[::-1],
    xerr=perm_df["importance_std"].iloc[::-1],
    color="seagreen",
)
plt.title("Permutation Importance")
plt.xlabel("Disminución del R²")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "permutation_importance.png", dpi=200)
plt.close()

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(preprocessed_test)
shap.summary_plot(
    shap_values, preprocessed_test, feature_names=feature_names, show=False
)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "shap_summary.png", dpi=200)
plt.close()

try:
    feature_name = (
        "num__reading score"
        if "num__reading score" in feature_names
        else feature_names[0]
    )
    pdp_iso = pdp.pdp_isolate(
        model=rf_model,
        dataset=preprocessed_test,
        model_features=feature_names,
        feature=feature_name,
        num_grid_points=20,
    )
    pdp.pdp_plot(
        pdp_iso,
        feature_name="reading score",
        save_fig=True,
        fname=str(PLOTS_DIR / "pdp_reading_score.png"),
    )
    plt.close("all")
except Exception:
    pass

# Estimate effect of test prep course
X_effect = X_test.copy()
for col in X_effect.columns:
    if col == "test preparation course":
        continue
    X_effect[col] = (
        X_effect[col].median()
        if pd.api.types.is_numeric_dtype(X_effect[col])
        else X_effect[col].mode().iloc[0]
    )
X_completed = X_effect.copy()
X_none = X_effect.copy()
X_completed["test preparation course"] = "completed"
X_none["test preparation course"] = "none"
impact_points = float(
    final_pipeline.predict(X_completed).mean() - final_pipeline.predict(X_none).mean()
)

# Prediction helper


def predecir_estudiante(
    model,
    gender,
    race_ethnicity,
    parental_level_of_education,
    lunch,
    test_prep_course,
    reading_score,
    writing_score,
):
    if not 0 <= reading_score <= 100:
        raise ValueError("reading_score debe estar entre 0 y 100")
    if not 0 <= writing_score <= 100:
        raise ValueError("writing_score debe estar entre 0 y 100")
    input_df = pd.DataFrame(
        [
            {
                "gender": gender,
                "race/ethnicity": race_ethnicity,
                "parental level of education": parental_level_of_education,
                "lunch": lunch,
                "test preparation course": test_prep_course,
                "reading score": reading_score,
                "writing score": writing_score,
            }
        ]
    )
    return float(model.predict(input_df)[0])


# Save metadata
metadata = {
    "setup_df": {
        "dataset_rows": int(len(analysis_df)),
        "dataset_columns": list(analysis_df.columns),
        "target": target,
        "scenario_a_features": scenario_a_features,
        "scenario_b_features": scenario_b_features,
    },
    "results_df": results_df.to_dict(orient="records"),
    "holdout_metrics": holdout_metrics,
    "fairness_df": fairness_df.to_dict(orient="records"),
    "interpretability": {
        "feature_importance_top_10": importances.head(10).to_dict(),
        "permutation_importance_top_10": perm_df.head(10).to_dict(orient="records"),
        "test_prep_course_effect_points": impact_points,
        "sample_prediction": predecir_estudiante(
            final_pipeline,
            "female",
            "group B",
            "bachelor's degree",
            "standard",
            "completed",
            78,
            80,
        ),
    },
    "parameters": {
        "cv_splits": 5,
        "group_column": "race/ethnicity",
        "random_state": 42,
        "remove_outliers": False,
        "baseline_model": "DummyRegressor",
        "final_model": "RandomForestRegressor",
    },
}
with (ROOT / "metadata_experimento.json").open("w", encoding="utf-8") as fh:
    json.dump(metadata, fh, indent=2, ensure_ascii=False)

joblib.dump(final_pipeline, MODELS_DIR / "modelo_escenario_b.joblib")
joblib.dump(models["DummyRegressor"], MODELS_DIR / "baseline_dummy.joblib")

# Build the notebook JSON
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def add_markdown(text):
    notebook["cells"].append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": text.splitlines(keepends=True),
        }
    )


def add_code(code):
    notebook["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code.splitlines(keepends=True),
        }
    )


add_markdown(
    "# Predicción de Rendimiento Académico\n\nEste notebook resume los resultados y artefactos generados por el experimento."
)
add_code(
    """import json
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / 'metadata_experimento.json').exists():
    ROOT = ROOT.parent

metadata = json.loads((ROOT / 'metadata_experimento.json').read_text('utf-8'))

print('Setup:')
print(metadata['setup_df'])
print('\nHoldout metrics:')
print(metadata['holdout_metrics'])
print('\nSample prediction:')
print(metadata['interpretability']['sample_prediction'])
"""
)
add_markdown(
    "## Artefactos generados\n\n- `metadata_experimento.json`\n- `models/modelo_escenario_b.joblib`\n- `models/baseline_dummy.joblib`\n- `plots/` con gráficos de evaluación e interpretabilidad\n"
)
with (ROOT / "jupyter" / "Prediccion_Rendimiento_Academico_PyCaret-2.ipynb").open(
    "w", encoding="utf-8"
) as fh:
    import json

    json.dump(notebook, fh, indent=1)

print(
    "Notebook written to",
    ROOT / "jupyter" / "Prediccion_Rendimiento_Academico_PyCaret-2.ipynb",
)
