"""Quick environment check for required machine learning dependencies."""
from importlib import import_module
from typing import Dict

REQUIRED_PACKAGES: Dict[str, str] = {
    "tushare": "ts",  # alias used in code
    "lightgbm": "lgb",
    "xgboost": "xgb",
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "numpy": "numpy",
}


def main() -> None:
    print("Checking required Python packages...")
    for package, alias in REQUIRED_PACKAGES.items():
        try:
            module = import_module(package)
            version = getattr(module, "__version__", "unknown")
            print(f"✔ {package} ({alias}) available – version: {version}")
        except Exception as exc:
            print(f"✖ {package} import failed: {exc}")


if __name__ == "__main__":
    main()
