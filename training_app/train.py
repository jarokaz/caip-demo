
import logging
import os
import subprocess
import sys

import fire
import joblib
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

def select_model(X, y, n_features_options, l2_reg_options):
    
  # Set up grid search
  pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('reduce_dim', PCA()),
    ('regress', Ridge())
  ])

  param_grid = [
    {
        'reduce_dim': [PCA()],
        'reduce_dim__n_components': n_features_options,
        'regress': [Ridge()],
        'regress__alpha': l2_reg_options
    },
    {
        'reduce_dim': ['passthrough'],
        'regress': [PLSRegression(scale=False)],
        'regress__n_components': n_features_options
    }
  ]

  grid = GridSearchCV(pipeline, cv=10, n_jobs=None, param_grid=param_grid, scoring='neg_mean_squared_error', iid=False)

  grid.fit(X, y)

  return grid

def train(job_dir, data_path, n_features_options, l2_reg_options):
    
  # Load data from GCS
  df_train = pd.read_csv(data_path, index_col=0)
  y = df_train.octane
  X = df_train.drop('octane', axis=1)
    
  # Find the best model
  y = df_train.octane
  X = df_train.drop('octane', axis=1)
  grid = select_model(X, y, n_features_options, l2_reg_options)

  logging.info("Best estimator: {}".format(grid.best_params_))
  logging.info("Best score: {}".format(grid.best_score_))
    
  # Retrain the best model on a full dataset
  best_estimator = grid.best_estimator_
  trained_pipeline = best_estimator.fit(X, y)

  # Save the model
  model_filename = 'model.joblib'
  joblib.dump(value=trained_pipeline, filename=model_filename)
  gcs_model_path = "{}/trained_model/{}".format(job_dir, model_filename)
  subprocess.check_call(['gsutil', 'cp', model_filename, gcs_model_path], stderr=sys.stdout)
  logging.info("Saved model in: {}".format(gcs_model_path))
  
    
if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  fire.Fire(train)
