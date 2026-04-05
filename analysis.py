"""Analysis tools for multi-inverter solar power forecasting."""
import numpy as np
import pandas as pd
from typing import Dict

def compare_inverters(data: pd.DataFrame, col: str = "inverter_id") -> pd.DataFrame:
    return data.groupby(col).agg(mean_power=("ac_power","mean"), max_power=("ac_power","max"), std_power=("ac_power","std")).reset_index().sort_values("mean_power", ascending=False)

def detect_anomalies(series: pd.Series, threshold: float = 2.5) -> pd.Series:
    mean, std = series.mean(), series.std()
    return (np.abs((series - mean) / std) > threshold) if std != 0 else pd.Series([False]*len(series), index=series.index)

def forecast_accuracy(actuals: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    mape = np.mean(np.abs((actuals - preds) / np.where(actuals==0,1,actuals))) * 100
    rmse = np.sqrt(np.mean((actuals - preds)**2))
    return {"mape": round(mape,2), "rmse": round(rmse,4)}
