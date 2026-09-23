from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor


def build_models(task: str, seed: int):
    if task == "regression":
        return {
            "Ridge_desc": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            "RF_desc": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=seed),
            "ExtraTrees_desc": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.8, n_jobs=-1, random_state=seed),
            "XGBoost_desc": XGBRegressor(n_estimators=350, max_depth=5, learning_rate=0.035, subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0, objective="reg:squarederror", n_jobs=4, random_state=seed),
            "RF_Morgan": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=seed),
        }
    return {
        "Logistic_desc": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=seed)),
        "RF_desc": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=seed),
        "ExtraTrees_desc": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, max_features=0.8, class_weight="balanced", n_jobs=-1, random_state=seed),
        "XGBoost_desc": XGBClassifier(n_estimators=350, max_depth=5, learning_rate=0.035, subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0, eval_metric="logloss", n_jobs=4, random_state=seed),
        "RF_Morgan": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=seed),
    }


def fit_predict(model, x_train, y_train, x_test, task: str):
    model.fit(x_train, y_train)
    if task == "classification":
        return model.predict_proba(x_test)[:, 1]
    return model.predict(x_test)
