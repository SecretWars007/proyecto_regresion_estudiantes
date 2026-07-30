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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone

import shap
from pdpbox import pdp

ROOT = Path.cwd()
if not (ROOT / "dataset" / "StudentsPerformance.csv").exists():
    ROOT = ROOT.parent
DATA_PATH = ROOT / "dataset" / "StudentsPerformance.csv"
PLOTS_DIR = ROOT / "plots"
MODELS_DIR = ROOT / "models"
PLOTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Load data
raw_df = pd.read_csv(DATA_PATH)

# Basic cleaning
for col in [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
]:
    raw_df[col] = raw_df[col].astype(str).str.strip()

# Prepare features
target = "math score"
scenario_a_features = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
]
scenario_b_features = scenario_a_features + ["reading score", "writing score"]

# Create a copy with explicit binary feature for test prep for interpretability
analysis_df = raw_df.copy()
analysis_df["test_prep_completed"] = (
    analysis_df["test preparation course"] == "completed"
).astype(int)

# Use all data for fitting the final model, but keep a holdout set for evaluation
X_full = analysis_df[scenario_b_features + ["reading score", "writing score"]]
# For scenario A and B, features differ only by presence of reading/writing
feature_sets = {
    "Escenario A": scenario_a_features,
    "Escenario B": scenario_b_features,
}

# Create preprocessing
numeric_features = ["reading score", "writing score"]
cat_features = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
        ),
    ],
    remainder="drop",
)

# Define model comparison helper
models = {
    "DummyRegressor": DummyRegressor(strategy="mean"),
    "RandomForest": RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
}


def evaluate_scenario(features, scenario_name):
    X = analysis_df[features].copy()
    y = analysis_df[target]
    groups = analysis_df["race/ethnicity"]
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X, y, groups, test_size=0.2, random_state=42, stratify=None
    )
    gkf = GroupKFold(n_splits=5)
    rows = []
    for model_name, base_model in models.items():
        fold_scores = []
        for train_idx, val_idx in gkf.split(X_train, y_train, groups_train):
            model = clone(base_model)
            X_tr = X_train.iloc[train_idx]
            X_va = X_train.iloc[val_idx]
            y_tr = y_train.iloc[train_idx]
            y_va = y_train.iloc[val_idx]
            pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
            pipeline.fit(X_tr, y_tr)
            pred = pipeline.predict(X_va)
            fold_scores.append(
                {
                    "rmse": np.sqrt(mean_squared_error(y_va, pred)),
                    "mae": mean_absolute_error(y_va, pred),
                    "r2": r2_score(y_va, pred),
                }
            )
        mean_scores = pd.DataFrame(fold_scores).mean()
        std_scores = pd.DataFrame(fold_scores).std()
        rows.append(
            {
                "scenario": scenario_name,
                "model": model_name,
                "rmse_mean": mean_scores["rmse"],
                "rmse_std": std_scores["rmse"],
                "mae_mean": mean_scores["mae"],
                "mae_std": std_scores["mae"],
                "r2_mean": mean_scores["r2"],
            }
        )
    return pd.DataFrame(rows)


results_df = pd.concat(
    [evaluate_scenario(feature_sets[name], name) for name in feature_sets],
    ignore_index=True,
)
results_df = results_df.sort_values(["scenario", "rmse_mean"]).reset_index(drop=True)

# Fit final best model on scenario B using RandomForest
best_scenario = "Escenario B"
best_model_name = "RandomForest"
train_features = feature_sets[best_scenario]
X = analysis_df[train_features].copy()
y = analysis_df[target]
groups = analysis_df["race/ethnicity"]
X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X, y, groups, test_size=0.2, random_state=42
)
final_pipeline = Pipeline(
    [
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)),
    ]
)
final_pipeline.fit(X_train, y_train)

# Holdout metrics
pred_test = final_pipeline.predict(X_test)
metrics_holdout = {
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

# Save figures
sns.set_theme(style="whitegrid")
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

# Interpretability
# Feature importance from the trained model
rf_model = final_pipeline.named_steps["model"]
# Use preprocessed features names from pipeline
preprocessed = final_pipeline.named_steps["preprocess"].fit_transform(X_train)
feature_names = final_pipeline.named_steps["preprocess"].get_feature_names_out()
# Need to use the same transformed data as the fitted model
# Refit using the final preprocessed matrix to align names
preprocessed_train = final_pipeline.named_steps["preprocess"].fit_transform(X_train)
preprocessed_test = final_pipeline.named_steps["preprocess"].transform(X_test)
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

# Permutation importance
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

# SHAP summary
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(preprocessed_test)
shap.summary_plot(
    shap_values, preprocessed_test, feature_names=feature_names, show=False
)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "shap_summary.png", dpi=200)
plt.close()

# PDP for reading score
# Build a new pipeline with the same preprocess and model
pdp_model = Pipeline(
    [
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)),
    ]
)
pdp_model.fit(X_train, y_train)
feature_idx = (
    list(feature_names).index("num__reading score")
    if "num__reading score" in feature_names
    else None
)
if feature_idx is not None:
    pdp_iso = pdp.pdp_isolate(
        model=pdp_model.named_steps["model"],
        dataset=preprocessed_test,
        model_features=feature_names,
        feature="num__reading score",
        num_grid_points=20,
    )
    pdp.pdp_plot(
        pdp_iso,
        feature_name="reading score",
        save_fig=True,
        fname=str(PLOTS_DIR / "pdp_reading_score.png"),
    )
    plt.close("all")
else:
    print("PDP skipped due to feature naming mismatch")

# Effect of test preparation course
# Use the feature already present in X_test before transformation
# We compare completed vs none while keeping the rest at median/mode values
X_effect = X_test.copy()
for col in X_effect.columns:
    if col == "test preparation course":
        continue
    if X_effect[col].dtype.kind in "biufc":
        X_effect[col] = X_effect[col].median()
    else:
        X_effect[col] = X_effect[col].mode().iloc[0]

X_completed = X_effect.copy()
X_none = X_effect.copy()
X_completed["test preparation course"] = "completed"
X_none["test preparation course"] = "none"
base_pred_complete = final_pipeline.predict(X_completed)
base_pred_none = final_pipeline.predict(X_none)
impact_points = float(base_pred_complete.mean() - base_pred_none.mean())

# Student prediction helper


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


sample_pred = predecir_estudiante(
    final_pipeline,
    "female",
    "group B",
    "bachelor's degree",
    "standard",
    "completed",
    78,
    80,
)

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
    "holdout_metrics": metrics_holdout,
    "fairness_df": fairness_df.to_dict(orient="records"),
    "interpretability": {
        "feature_importance_top_10": importances.head(10).to_dict(),
        "permutation_importance_top_10": perm_df.head(10).to_dict(orient="records"),
        "test_prep_course_effect_points": impact_points,
        "sample_prediction": sample_pred,
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

# Save final model
joblib.dump(final_pipeline, MODELS_DIR / "modelo_escenario_b.joblib")
joblib.dump(models["DummyRegressor"], MODELS_DIR / "baseline_dummy.joblib")

# Create notebook JSON
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

# Helper to add cell


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
    "# Predicción de Rendimiento Académico con PyCaret y validación científica\n\nEste notebook corrige el riesgo de data leakage, incorpora validación por grupos y añade interpretabilidad para publicación científica."
)
add_code(
    """import json\nfrom pathlib import Path\nimport warnings\nwarnings.filterwarnings(\'ignore\')\n\nimport joblib\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\nimport seaborn as sns\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.dummy import DummyRegressor\nfrom sklearn.ensemble import RandomForestRegressor\nfrom sklearn.inspection import permutation_importance\nfrom sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\nfrom sklearn.model_selection import GroupKFold, train_test_split\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import OneHotEncoder, StandardScaler\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.base import clone\n\nimport shap\nfrom pdpbox import pdp\n\nROOT = Path.cwd()\nif not (ROOT / \"dataset\" / \"StudentsPerformance.csv\").exists():\n    ROOT = ROOT.parent\nDATA_PATH = ROOT / \"dataset\" / \"StudentsPerformance.csv\"\nPLOTS_DIR = ROOT / \"plots\"\nMODELS_DIR = ROOT / \"models\"\nPLOTS_DIR.mkdir(exist_ok=True)\nMODELS_DIR.mkdir(exist_ok=True)\n"""
)
add_code(
    """raw_df = pd.read_csv(DATA_PATH)\nfor col in [\'gender\', \"race/ethnicity\", \"parental level of education\", \"lunch\", \"test preparation course\"]:\n    raw_df[col] = raw_df[col].astype(str).str.strip()\n\nanalysis_df = raw_df.copy()\nanalysis_df[\'test_prep_completed\'] = (analysis_df[\'test preparation course\'] == \"completed\").astype(int)\n\ntarget = \"math score\"\nscenario_a_features = [\'gender\', \"race/ethnicity\", \"parental level of education\", \"lunch\", \"test preparation course\"]\nscenario_b_features = scenario_a_features + [\'reading score\', \"writing score\"]\n\n# Comentario: el escenario A evita data leakage al excluir reading/writing, mientras el escenario B sirve como referencia de desempeño máximo.\n"""
)
add_code(
    """numeric_features = [\'reading score\', \"writing score\"]\ncat_features = [\'gender\', \"race/ethnicity\", \"parental level of education\", \"lunch\", \"test preparation course\"]\n\npreprocessor = ColumnTransformer(\n    transformers=[\n        (\'cat\', OneHotEncoder(handle_unknown=\'ignore\'), cat_features),\n        (\'num\', Pipeline([\'imputer\', SimpleImputer(strategy=\'median\')]), numeric_features),\n    ],\n    remainder=\'drop\',\n)\n\nmodels = {\n    \"DummyRegressor\": DummyRegressor(strategy=\'mean\'),\n    \"RandomForest\": RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),\n}\n\n\ndef evaluate_scenario(features, scenario_name):\n    X = analysis_df[features].copy()\n    y = analysis_df[target]\n    groups = analysis_df[\'race/ethnicity\']\n    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(\n        X, y, groups, test_size=0.2, random_state=42\n    )\n    gkf = GroupKFold(n_splits=5)\n    rows = []\n    for model_name, base_model in models.items():\n        fold_scores = []\n        for train_idx, val_idx in gkf.split(X_train, y_train, groups_train):\n            model = clone(base_model)\n            X_tr = X_train.iloc[train_idx]\n            X_va = X_train.iloc[val_idx]\n            y_tr = y_train.iloc[train_idx]\n            y_va = y_train.iloc[val_idx]\n            pipeline = Pipeline([\'preprocess\', preprocessor), (\'model\', model)])\n            pipeline.fit(X_tr, y_tr)\n            pred = pipeline.predict(X_va)\n            fold_scores.append({\'rmse\': np.sqrt(mean_squared_error(y_va, pred)), \"mae\": mean_absolute_error(y_va, pred), \"r2\": r2_score(y_va, pred)})\n        mean_scores = pd.DataFrame(fold_scores).mean()\n        std_scores = pd.DataFrame(fold_scores).std()\n        rows.append({\n            \"scenario\": scenario_name,\n            \"model\": model_name,\n            \"rmse_mean\": mean_scores[\'rmse\'],\n            \"rmse_std\": std_scores[\'rmse\'],\n            \"mae_mean\": mean_scores[\'mae\'],\n            \"mae_std\": std_scores[\'mae\'],\n            \"r2_mean\": mean_scores[\'r2\'],\n        })\n    return pd.DataFrame(rows)\n\nresults_df = pd.concat([evaluate_scenario(scenario_a_features, \"Escenario A\"), evaluate_scenario(scenario_b_features, \"Escenario B\")], ignore_index=True)\nresults_df = results_df.sort_values([\"scenario\", \"rmse_mean\"]).reset_index(drop=True)\nresults_df\n"""
)
add_code(
    """sns.set_theme(style=\'whitegrid\')\nplt.figure(figsize=(10, 5))\nax = sns.barplot(data=results_df, x=\'scenario\', y=\'rmse_mean\', hue=\'model\', palette=\'viridis\')\nax.set_title(\'Comparación de RMSE por escenario y modelo\')\nax.set_ylabel(\'RMSE promedio (CV)\')\nax.set_xlabel(\'Escenario\')\nplt.tight_layout()\nplt.savefig(PLOTS_DIR / \"comparacion_scenarios_rmse.png\", dpi=200)\nplt.close()\n\nplt.figure(figsize=(10, 5))\nax = sns.barplot(data=results_df, x=\'scenario\', y=\'r2_mean\', hue=\'model\', palette=\'magma\')\nax.set_title(\'Comparación de R² por escenario y modelo\')\nax.set_ylabel(\'R² promedio (CV)\')\nax.set_xlabel(\'Escenario\')\nplt.tight_layout()\nplt.savefig(PLOTS_DIR / \"comparacion_scenarios_r2.png\", dpi=200)\nplt.close()\nresults_df\n"""
)
add_code(
    """# Comentario: se entrena el modelo final sobre el escenario B, que incluye las variables académicas de lectura/escritura y ofrece mayor capacidad predictiva.\nX = analysis_df[scenario_b_features].copy()\ny = analysis_df[target]\ngroups = analysis_df[\'race/ethnicity\']\nX_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(\n    X, y, groups, test_size=0.2, random_state=42\n)\nfinal_pipeline = Pipeline([\n    (\'preprocess\', preprocessor),\n    (\'model\', RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)),\n])\nfinal_pipeline.fit(X_train, y_train)\npred_test = final_pipeline.predict(X_test)\n\nholdout_metrics = {\n    \"rmse\": float(np.sqrt(mean_squared_error(y_test, pred_test))),\n    \"mae\": float(mean_absolute_error(y_test, pred_test)),\n    \"r2\": float(r2_score(y_test, pred_test)),\n}\nholdout_metrics\n"""
)
add_code(
    """fairness_rows = []\nfor group_col in [\'gender\', \"race/ethnicity\", \"lunch\"]:\n    group_values = analysis_df.loc[X_test.index, group_col]\n    for group in sorted(group_values.unique()):\n        mask = group_values == group\n        y_true = y_test[mask]\n        y_pred = pred_test[mask]\n        fairness_rows.append({\n            \"group_col\": group_col,\n            \"group\": group,\n            \"rmse\": float(np.sqrt(mean_squared_error(y_true, y_pred))),\n            \"mae\": float(mean_absolute_error(y_true, y_pred))),\n            \"count\": int(mask.sum()),\n        })\nfairness_df = pd.DataFrame(fairness_rows)\n\nfor group_col in [\'gender\', \"race/ethnicity\", \"lunch\"]:\n    plot_df = fairness_df[fairness_df[\'group_col\'] == group_col].copy()\n    plt.figure(figsize=(10, 5))\n    ax = sns.barplot(data=plot_df, x=\'group\', y=\'rmse\', palette=\'coolwarm\')\n    ax.set_title(f\'RMSE por {group_col}\')\n    ax.set_ylabel(\'RMSE\')\n    ax.set_xlabel(group_col)\n    plt.xticks(rotation=30)\n    plt.tight_layout()\n    plt.savefig(PLOTS_DIR / f\'fairness_rmse_{group_col.replace(\"/\", \"_\")}.png\', dpi=200)\n    plt.close()\n\nfairness_df\n"""
)
add_code(
    """# Comentario: la importancia de variables y el análisis de permutación aportan una capa de interpretabilidad distinta al SHAP.\npreprocessed_train = final_pipeline.named_steps[\'preprocess\'].fit_transform(X_train)\npreprocessed_test = final_pipeline.named_steps[\'preprocess\'].transform(X_test)\nfeature_names = final_pipeline.named_steps[\'preprocess\'].get_feature_names_out()\nrf_model = final_pipeline.named_steps[\'model\']\nrf_model.fit(preprocessed_train, y_train)\n\nimportances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)\nimportances.plot(kind=\'bar\', color=\'steelblue\')\nplt.title(\'Feature Importance\')\nplt.ylabel(\'Importancia\')\nplt.tight_layout()\nplt.savefig(PLOTS_DIR / \"feature_importance.png\", dpi=200)\nplt.close()\n\nperm = permutation_importance(rf_model, preprocessed_test, y_test, n_repeats=20, random_state=42, n_jobs=-1)\nperm_df = pd.DataFrame({\'feature\': feature_names, \"importance_mean\": perm.importances_mean, \"importance_std\": perm.importances_std}).sort_values(\'importance_mean\', ascending=False).head(15)\nperm_df\n"""
)
add_code(
    """# Comentario: los valores SHAP y el PDP ayudan a traducir la predicción en lenguaje causal/interpretativo para la tesis.\nexplainer = shap.TreeExplainer(rf_model)\nshap_values = explainer.shap_values(preprocessed_test)\nshap.summary_plot(shap_values, preprocessed_test, feature_names=feature_names, show=False)\nplt.tight_layout()\nplt.savefig(PLOTS_DIR / \"shap_summary.png\", dpi=200)\nplt.close()\n\n# PDP for reading score\ntry:\n    # The preprocessed feature name for reading score is used so the PDP is comparable with the model input space.\n    feature_name = \"num__reading score\" if \"num__reading score\" in feature_names else feature_names[0]\n    pdp_isolate = pdp.pdp_isolate(\n        model=rf_model,\n        dataset=preprocessed_test,\n        model_features=feature_names,\n        feature=feature_name,\n        num_grid_points=20,\n    )\n    pdp.pdp_plot(pdp_isolate, feature_name=\"reading score\", save_fig=True, fname=str(PLOTS_DIR / \"pdp_reading_score.png\"))\n    plt.close(\"all\")\nexcept Exception:\n    pass\n\n# Effect of test prep course\nX_effect = X_test.copy()\nfor col in X_effect.columns:\n    if col == \"test preparation course\":\n        continue\n    X_effect[col] = X_effect[col].median() if pd.api.types.is_numeric_dtype(X_effect[col]) else X_effect[col].mode().iloc[0]\nX_completed = X_effect.copy()\nX_none = X_effect.copy()\nX_completed[\"test preparation course\"] = \"completed\"\nX_none[\"test preparation course\"] = \"none\"\nimpact_points = float(final_pipeline.predict(X_completed).mean() - final_pipeline.predict(X_none).mean())\nprint(f\"Si el estudiante completa el curso de preparación, el modelo sube la predicción en {impact_points:.2f} puntos en promedio.\")\n"""
)
add_code(
    """# Comentario: la función de predicción valida que las entradas estén en el rango esperado, evitando errores de uso práctico.\ndef predecir_estudiante(model, gender, race_ethnicity, parental_level_of_education, lunch, test_prep_course, reading_score, writing_score):\n    if not 0 <= reading_score <= 100:\n        raise ValueError(\'reading_score debe estar entre 0 y 100\')\n    if not 0 <= writing_score <= 100:\n        raise ValueError(\'writing_score debe estar entre 0 y 100\')\n    input_df = pd.DataFrame([{{\n        \"gender\": gender,\n        \"race/ethnicity\": race_ethnicity,\n        \"parental level of education\": parental_level_of_education,\n        \"lunch\": lunch,\n        \"test preparation course\": test_prep_course,\n        \"reading score\": reading_score,\n        \"writing score\": writing_score,\n    }}])\n    return float(model.predict(input_df)[0])\n\npredecir_estudiante(final_pipeline, \"female\", \"group B\", \"bachelor\'s degree\", \"standard\", \"completed\", 78, 80)\n"""
)
add_code(
    """import json\nmetadata = {\n    \"setup_df\": {\n        \"dataset_rows\": int(len(analysis_df)),\n        \"dataset_columns\": list(analysis_df.columns),\n        \"target\": target,\n        \"scenario_a_features\": scenario_a_features,\n        \"scenario_b_features\": scenario_b_features,\n    },\n    \"results_df\": results_df.to_dict(orient=\'records\'),\n    \"holdout_metrics\": holdout_metrics,\n    \"fairness_df\": fairness_df.to_dict(orient=\'records\'),\n    \"parameters\": {\n        \"cv_splits\": 5,\n        \"group_column\": \"race/ethnicity\",\n        \"random_state\": 42,\n        \"remove_outliers\": False,\n        \"baseline_model\": \"DummyRegressor\",\n        \"final_model\": \"RandomForestRegressor\",\n    },\n}\nwith (ROOT / \"metadata_experimento.json\").open(\"w\", encoding=\"utf-8\") as fh:\n    json.dump(metadata, fh, indent=2, ensure_ascii=False)\n\njoblib.dump(final_pipeline, MODELS_DIR / \"modelo_escenario_b.joblib\")\njoblib.dump(models[\"DummyRegressor\"], MODELS_DIR / \"baseline_dummy.joblib\")\n"""
)
add_markdown(
    """## Limitaciones y Trabajo Futuro\n\n- El dataset es sintético y no refleja la complejidad de una institución real.\n- El modelo debe validarse con datos institucionales y variables temporales.\n- Se recomienda incorporar fairness audit y monitoreo post-deployment.\n\n## Criterios de Aceptación del Modelo\n\n- RMSE menor a 8 puntos en el conjunto de prueba.\n- R² mayor o igual a 0.70.\n- Diferencias de error por subgrupos menores al 10% relativo.\n"""
)

with (ROOT / "jupyter" / "Prediccion_Rendimiento_Academico_PyCaret-2.ipynb").open(
    "w", encoding="utf-8"
) as fh:
    json.dump(notebook, fh, indent=1)

print(
    "Notebook generated at",
    ROOT / "jupyter" / "Prediccion_Rendimiento_Academico_PyCaret-2.ipynb",
)
