from setuptools import setup, find_packages

setup(
    name="predictive-alerting",
    version="0.1.0",
    description="Predictive alerting system for cloud metrics using ML anomaly detection",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "boto3>=1.26.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "xgboost>=1.7.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "matplotlib>=3.6.0",
            "seaborn>=0.12.0",
            "jupyter>=1.0.0",
        ]
    },
)
