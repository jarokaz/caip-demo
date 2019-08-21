
from setuptools import setup

setup(
    name='custom-predictor-2',
    description='Custom prediction routine.',
    version='0.1',
    install_requires=[
      'scikit-learn>=0.21.2',
      'numpy>=1.11.0',
      'scipy>=0.17.0',
      'joblib>=0.11'
    ],
    scripts=['predict.py']
)
