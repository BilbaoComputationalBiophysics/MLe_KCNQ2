import numpy as np
from sklearn.base import BaseEstimator

class FeaturizedModel:
    
    def __init__(self, sk_model: BaseEstimator, feature_idx: np.array, name: str='model', feature_names: list=None) -> None:
        """Wrapper class for scikit-learn models that trains the model and makes predictions using
        only a set of predefined features, rather than all features available in the input data.

        Args:
            sk_model (BaseEstimator): Base scikit-learn model used to make predictions.
            feature_idx (np.array): Array of indices that determine which of the available features will be used for training/predictions, and which will be ignored.
            name (str, optional): Name of the model. Defaults to 'model'.
            feature_names (list, optional): List of names associated to each feature used for training/prediction. Defaults to None.
        """

        self.model = sk_model
        self.feature_idx = feature_idx
        self.name = name
        self.feature_names = feature_names
        
    def fit(self, X: np.array, y: np.array) -> None:
        """Fit the base scikit-learn model using only the selected set of features.

        Args:
            X (np.array): Array of training data containing all features.
            y (np.array): Array of labels for training data.
        """
        
        X_feat = X[:, self.feature_idx]
        self.model.fit(X_feat, y)
        
    def predict(self, X: np.array) -> np.array:
        """Make predictions for a set of samples using only the selected set of features.

        Args:
            X (np.array): Array of samples whose labels will be predicted.

        Returns:
            np.array: Array of predictions for earch sample.
        """
        
        X_feat = X[:, self.feature_idx]
        y_pred = self.model.predict(X_feat)
        
        return y_pred
    
    def predict_proba(self, X: np.array) -> np.array:
        """Predict the probabilities for each label for a set of samples using only the selected set of features.

        Args:
            X (np.array): Array of samples whose labels will be predicted.

        Returns:
            np.array: Array of probabilities for each label for earch sample.
        """
        
        X_feat = X[:, self.feature_idx]
        y_pred = self.model.predict_proba(X_feat)
        
        return y_pred
