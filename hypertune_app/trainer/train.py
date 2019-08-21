
import logging
import os
import subprocess
import sys

import fire
import numpy as np
import pandas as pd

import hypertune

from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train(job_dir, data_path, n_components, alpha):
    
    
  # Load data from GCS
  df_train = pd.read_csv(data_path)

  y = df_train.octane
  X = df_train.drop('octane', axis=1)
    
  # Configure a training pipeline
  pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('reduce_dim', PCA(n_components=n_components)),
    ('regress', Ridge(alpha=alpha))
  ])

  # Calculate the performance metric
  scores = cross_val_score(pipeline, X, y, cv=10, scoring='neg_mean_squared_error')
    
  # Log it with hypertune
  hpt = hypertune.HyperTune()
  hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag='neg_mean_squared_error',
    metric_value=scores.mean()
    )

  # Fit the model on a full dataset
  pipeline.fit(X, y)

  # Save the model
  model_filename = 'model.joblib'
  joblib.dump(value=pipeline, filename=model_filename)
  gcs_model_path = "{}/{}".format(job_dir, model_filename)
  subprocess.check_call(['gsutil', 'cp', model_filename, gcs_model_path], stderr=sys.stdout)
  logging.info("Saved model in: {}".format(gcs_model_path)) 
    
if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  fire.Fire(train)
