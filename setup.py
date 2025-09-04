import importlib

packages = {
    "pandas": "pandas",
    "numpy": "numpy",
    "scikit-learn": "sklearn",
    "nltk": "nltk",
    "xgboost": "xgboost",
    "joblib": "joblib",
    "gradio": "gradio",
    "lime": "lime"
}

for name, module_name in packages.items():
    try:
        pkg = importlib.import_module(module_name)
        version = getattr(pkg, "__version__", "Version info not available")
        print(f"{name}: ✅ Installed — Version {version}")
    except ImportError:
        print(f"{name}: ❌ Not Installed")
