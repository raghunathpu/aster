"""
ASTER — LightGBM Wrapper
=========================
Wrapper for LightGBM classification and regression models.
Used for deserializing pre-trained model pickles from Gridvision.
"""
import lightgbm as lgb
import numpy as np
import pickle
from pathlib import Path

class LightGBMModel:
    def __init__(self, model_type="classification", params=None):
        self.model_type = model_type
        self.params = params or {}
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None, use_early_stopping=False):
        X_train_arr = np.array(X_train)
        y_train_arr = np.array(y_train)
        
        if self.model_type == "classification":
            self.model = lgb.LGBMClassifier(**self.params)
        else:
            self.model = lgb.LGBMRegressor(**self.params)
            
        eval_set = None
        callbacks = None
        if X_val is not None and y_val is not None:
            eval_set = [(np.array(X_val), np.array(y_val))]
            if use_early_stopping:
                callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]
            
        self.model.fit(
            X_train_arr,
            y_train_arr,
            eval_set=eval_set,
            callbacks=callbacks
        )
        return self

    def predict(self, X):
        X_arr = np.array(X)
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
        return self.model.predict(X_arr)

    def predict_proba(self, X):
        X_arr = np.array(X)
        if self.model_type != "classification":
            raise ValueError("predict_proba is only available for classification models.")
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
        return self.model.predict_proba(X_arr)

    def get_feature_importance(self, feature_names=None):
        if self.model is None:
            raise ValueError("Model is not fitted yet.")
        importances = self.model.feature_importances_
        if feature_names is not None:
            return dict(sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True))
        return importances

    def save(self, filepath):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)
