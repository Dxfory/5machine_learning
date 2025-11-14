"""Minimal smoke test for the MonthlyRolling5ModelStrategy."""
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_model_strategy import MonthlyRolling5ModelStrategy


def main() -> None:
    strategy = MonthlyRolling5ModelStrategy()
    strategy.training_months = 6  # keep the smoke test lightweight
    test_date = pd.Timestamp(datetime.today().strftime("%Y-%m-%d"))
    print(f"Running smoke test on {test_date.date()} with training_months={strategy.training_months}")

    try:
        trained = strategy.monthly_retrain(test_date, stock_pool="HS300")
        print(f"Smoke test retrain finished. Trained: {trained}, Models ready: {strategy.is_trained}")
        if strategy.is_trained:
            print(f"Selected features: {len(strategy.selected_features)}")
            print(f"Latest ensemble weights: {[(cfg['name'], cfg['weight']) for cfg in strategy.models_config.values()]}")
            if strategy.validation_history:
                latest_metrics = strategy.validation_history[-1]
                print(
                    "Latest validation metrics: "
                    f"RMSE={latest_metrics.get('rmse')}, R2={latest_metrics.get('r2')}"
                )
    except Exception as exc:
        print(f"Smoke test encountered an error: {exc}")


if __name__ == "__main__":
    main()
