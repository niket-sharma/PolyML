"""Gradient boosting regression example using scikit-learn.

This script demonstrates a modern boosting approach for predicting the HDPE
melt index using the ``HDPE_LG_Plant_Data.csv`` dataset.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent / "HDPE_LG_Plant_Data.csv"


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y


def main() -> None:
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    train_rmse = mean_squared_error(y_train, pred_train, squared=False)
    test_rmse = mean_squared_error(y_test, pred_test, squared=False)

    print(f"Gradient Boosting Train RMSE: {train_rmse:.3f}")
    print(f"Gradient Boosting Test RMSE: {test_rmse:.3f}")


if __name__ == "__main__":
    main()
