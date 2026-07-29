import pandas as pd
from pycaret.regression import setup, create_model, tune_model, pull, predict_model

# Load dataset
path = "dataset/StudentsPerformance.csv"
df = pd.read_csv(path)

# PyCaret setup matching the notebook
s = setup(
    data=df,
    target="math score",
    train_size=0.8,
    normalize=True,
    normalize_method="zscore",
    transformation=False,
    remove_outliers=True,
    outliers_threshold=0.05,
    categorical_features=[
        "gender",
        "race/ethnicity",
        "parental level of education",
        "lunch",
        "test preparation course",
    ],
    numeric_features=["reading score", "writing score"],
    fold=10,
    session_id=123,
    silent=True,
    verbose=False,
)

results = {}
for model_name in ["ridge", "lasso", "elasticnet"]:
    base_model = create_model(model_name, fold=10, verbose=False)
    tuned_model = tune_model(
        base_model, optimize="RMSE", n_iter=30, fold=10, verbose=False
    )
    pred = predict_model(tuned_model)
    metrics = pull()
    params = tuned_model.get_params()
    alpha = params.get("alpha")
    if alpha is None:
        alpha = params.get("lambda")
    results[model_name] = {
        "alpha": alpha,
        "train_r2": metrics.iloc[0].get("R2", None),
        "test_r2": metrics.iloc[0].get("R2", None),
        "rmse": metrics.iloc[0].get("RMSE", None),
        "metrics": metrics,
    }
    print(
        f"[{model_name}] alpha={alpha} rmse={metrics.iloc[0].get('RMSE'):.4f} r2={metrics.iloc[0].get('R2'):.4f}"
    )

print("\nRESULTS SUMMARY")
for name, item in results.items():
    print(name, item["alpha"], item["rmse"], item["train_r2"])
