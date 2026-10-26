"""
smartml.core.forecaster
=======================
Tabular time-series lag feature extraction, rolling statistics, and forecasting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score


@dataclass
class ForecastResult:
    target_col: str
    date_col: str
    horizon: int
    predictions: list[float]
    model_r2: float
    model_rmse: float
    naive_rmse: float
    outperformed_naive: bool
    narrative: str


class TabularForecaster:
    """Automatic lag feature generator and time-series forecaster."""

    def __init__(
        self,
        lags: tuple[int, ...] = (1, 2, 3),
        rolling_windows: tuple[int, ...] = (3,),
    ) -> None:
        self.lags = lags
        self.rolling_windows = rolling_windows

    def forecast(
        self,
        df: pd.DataFrame,
        target_col: str,
        date_col: Optional[str] = None,
        horizon: int = 5,
    ) -> ForecastResult:
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found.")

        work = df.copy()

        # Find date column if not specified
        if date_col is None:
            for c in work.columns:
                if pd.api.types.is_datetime64_any_dtype(work[c]):
                    date_col = c
                    break
        if date_col is None:
            # Check string columns that can be converted
            for c in work.columns:
                if c != target_col:
                    try:
                        s_dt = pd.to_datetime(work[c], errors="coerce")
                        if s_dt.notna().sum() > len(work) * 0.7:
                            date_col = c
                            work[date_col] = s_dt
                            break
                    except Exception:
                        pass

        if date_col is None:
            raise ValueError(
                "Could not find a valid date or datetime column in the dataset for time-series forecasting."
            )

        # Sort by date
        work = work.sort_values(by=date_col).reset_index(drop=True)
        y = pd.to_numeric(work[target_col], errors="coerce").ffill().bfill()

        # Create lag features
        feature_df = pd.DataFrame(index=work.index)
        for lag in self.lags:
            feature_df[f"lag_{lag}"] = y.shift(lag)

        # Create rolling mean and rolling std
        for w in self.rolling_windows:
            feature_df[f"rolling_mean_{w}"] = y.shift(1).rolling(w).mean()
            feature_df[f"rolling_std_{w}"] = y.shift(1).rolling(w).std().fillna(0)

        # Drop NaNs created by lagging
        valid_idx = feature_df.dropna().index
        X_train = feature_df.loc[valid_idx]
        y_train = y.loc[valid_idx]

        if len(X_train) < 10:
            raise ValueError("Not enough time points to build reliable lag features (need at least 10).")

        # Train/Test split chronologically (last 20%)
        split_idx = int(len(X_train) * 0.8)
        X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
        y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

        model = Ridge()
        model.fit(X_tr, y_tr)
        val_preds = model.predict(X_val)

        r2 = round(float(r2_score(y_val, val_preds)), 4)
        rmse = round(float(np.sqrt(mean_squared_error(y_val, val_preds))), 4)

        # Naive baseline (y_t = y_{t-1})
        naive_preds = y_val.shift(1).bfill()
        naive_rmse = round(float(np.sqrt(mean_squared_error(y_val, naive_preds))), 4)
        outperformed = rmse < naive_rmse

        # Multi-step ahead recursive forecasting
        history = list(y.values)
        future_preds = []
        for _ in range(horizon):
            row_dict = {}
            for lag in self.lags:
                val = history[-lag] if len(history) >= lag else history[-1]
                row_dict[f"lag_{lag}"] = val
            for w in self.rolling_windows:
                vals = history[-w:] if len(history) >= w else history
                row_dict[f"rolling_mean_{w}"] = float(np.mean(vals))
                row_dict[f"rolling_std_{w}"] = float(np.std(vals))

            next_x = pd.DataFrame([row_dict])
            next_pred = float(model.predict(next_x)[0])
            future_preds.append(round(next_pred, 4))
            history.append(next_pred)

        narrative = (
            f"Forecasted {horizon} steps ahead for '{target_col}'. Model validation RMSE: {rmse:.4f} "
            f"(vs naive baseline RMSE: {naive_rmse:.4f}). "
            f"{'Outperformed naive persistence.' if outperformed else 'Model performance is close to naive persistence.'}"
        )

        return ForecastResult(
            target_col=target_col,
            date_col=date_col,
            horizon=horizon,
            predictions=future_preds,
            model_r2=r2,
            model_rmse=rmse,
            naive_rmse=naive_rmse,
            outperformed_naive=outperformed,
            narrative=narrative,
        )
