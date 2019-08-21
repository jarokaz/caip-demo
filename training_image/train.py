
import logging
import os
import subprocess
import sys
import joblib
import fire
import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train(job_dir, data_path, n_features_options, l2_reg_options):
    
  # Load data from GCS
  df_train = pd.read_csv(data_path)

  y = df_train.octane
  X = df_train.drop('octane', axis=1)
    
  # Configure a training pipeline
  pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('reduce_dim', PCA()),
    ('regress', Ridge())
  ])

  # Configure a parameter grid
  param_grid = [
    {
      'reduce_dim__n_components': n_features_options,
      'regress__alpha': l2_reg_options
    }
  ]

  # Tune hyperparameters
  grid = GridSearchCV(pipeline, cv=10, n_jobs=None, param_grid=param_grid, scoring='neg_mean_squared_error', iid=False)
  grid.fit(X, y)

  logging.info("Best estimator: {}".format(grid.best_params_))
  logging.info("Best score: {}".format(grid.best_score_))
    
  # Retrain the best model on a full dataset
  best_estimator = grid.best_estimator_
  trained_pipeline = best_estimator.fit(X, y)

  # Save the model
  model_filename = 'model.joblib'
  joblib.dump(value=trained_pipeline, filename=model_filename)
  gcs_model_path = "{}/{}".format(job_dir, model_filename)
  subprocess.check_call(['gsutil', 'cp', model_filename, gcs_model_path], stderr=sys.stdout)
  logging.info("Saved model in: {}".format(gcs_model_path)) 
    
if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  fire.Fire(train)
