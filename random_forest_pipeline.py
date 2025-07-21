"""Random forest regression pipeline for HDPE melt index prediction.

This script provides a cleaned-up example that loads the
``HDPE_LG_Plant_Data.csv`` data, trains a ``RandomForestRegressor`` and reports
RMSE on the train and test sets.  It demonstrates a typical workflow for the
data in this repository.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent / "HDPE_LG_Plant_Data.csv"


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load features ``X`` and target ``y`` from the plant data CSV."""
    df = pd.read_csv(path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y


def main() -> None:
    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    scaler = StandardScaler().fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    model = RandomForestRegressor(random_state=0)
    model.fit(X_train_std, y_train)

    train_rmse = mean_squared_error(y_train, model.predict(X_train_std), squared=False)
    test_rmse = mean_squared_error(y_test, model.predict(X_test_std), squared=False)

    print(f"Train RMSE: {train_rmse:.3f}")
    print(f"Test RMSE: {test_rmse:.3f}")


if __name__ == "__main__":
    main()
