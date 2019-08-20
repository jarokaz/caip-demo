
import os
import numpy as np

from sklearn.externals import joblib

class OctaneRegressor(object):
    """A custom prediction routine for Octane regressor"""
    
    def __init__(self, model):
        """Stores the model loaded in from_path"""
        self._model = model
        
    def predict(self, instances, **kwargs):
        """Runs inference"""
    
        outputs = self._model.predict(np.asarray(instances))
        
        return outputs.tolist()

        
    @classmethod
    def from_path(cls, model_dir):
        """Loads the model from the joblib file"""
        model_path = os.path.join(model_dir, 'model.joblib')
        model = joblib.load(model_path)
        
        
        return cls(model)
    
