import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score

from app.config import RANDOM_SEED


def train_xgb(X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model: xgb.XGBClassifier, X_test: np.ndarray, y_test: np.ndarray) -> float:
    preds = model.predict(X_test)
    return float(accuracy_score(y_test, preds))
