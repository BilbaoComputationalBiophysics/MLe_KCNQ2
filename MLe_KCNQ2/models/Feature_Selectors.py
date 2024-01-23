import numpy as np
from copy import deepcopy
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator

class ForwardGreedySearch:
    
    def __init__(self, base_model: BaseEstimator, cv_splitter, metric: str, tol: int=3, n_jobs: int=4) -> None:
        """Performs feature selection by following a forward greedy procedure. Starting with
       a model using zero features, the algorithm follows these steps:

            1.- For all available features:
                1.1.- Add feature to the model, and evaluate its performance with cross-validation.
            2.- Permanently add the feature with the best cross-validation score to the model.
            3.- If the best CV score has not improved in the last $tol iterations, end the procedure,
                otherwise go to step 1.

        Args:
            base_model (BaseEstimator): Base model for which optimal features will be searched.
            cv_splitter (_type_): _description_
            metric (str): Metric to optimize during the feature search procedure.
            tol (int, optional): Number of iterations after which the algorithm will terminate if no improvement in the CV score is obtained. Defaults to 3.
            n_jobs (int, optional): Number of cores used to run the algorithm. Defaults to 4.
        """

        self.cv_splitter = cv_splitter
        self.metric = metric
        self.tol = tol
        self.n_jobs = n_jobs
        self.model = base_model
        
    def fit(self, X: np.array, y: np.array, feature_names: list=None, feature_mask: list=[]) -> None:
        """Find the set of features that optimizes the cross-validation score of the model
        for the input training data.

        Args:
            X (np.array): Array of featurized training samples used to find the best performing features.
            y (np.array): Array of labels for each training sample.
            feature_names (list, optional): List of names for each feature used to characterize the samples. Defaults to None.
            feature_mask (list, optional): Mask used to indicate the indices of features that should not be considered in the selection process. Defaults to [].
        """
        
        avail_features = [i for i in range(X.shape[1]) if i not in feature_mask]
        best_features = []
        self.feature_history = {}
        iters_no_improve = 0
        best_score_all_iters = 0
        iter = 0
        
        while iters_no_improve < self.tol and avail_features:
            
            iter += 1
            best_score = 0
            
            for feature in avail_features:
                
                test_features = best_features + [feature]
                
                scores = cross_val_score(self.model, X[:, test_features], y, cv=self.cv_splitter, scoring=self.metric, n_jobs=self.n_jobs)
                mean_score = scores.mean()
                
                if mean_score > best_score:
                    
                    best_score = mean_score
                    best_new_feature = feature
            
            if best_score <= best_score_all_iters:
                iters_no_improve += 1
            else:
                best_score_all_iters = best_score
                self.best_results = {'score': best_score, 'features': deepcopy(best_features)}
                iters_no_improve = 0
                        
            best_features.append(best_new_feature)
            avail_features.remove(best_new_feature)
            
            if feature_names is None:
                self.feature_history[iter] = {'score': best_score,
                                              'features': deepcopy(best_features)}
            else:
                self.feature_history[iter] = {'score': best_score,
                                              'features': deepcopy(best_features),
                                              'feature_names': feature_names[best_features]}                

class BackwardGreedySearch:
    
    def __init__(self, base_model: BaseEstimator, cv_splitter, metric: str, tol: int=3, n_jobs: int=4) -> None:
        """Performs feature selection by following a backward greedy procedure. Starting with
       a model using all available features, the algorithm follows these steps:

            1.- For all available features:
                1.1.- Remove a feature from the model, and evaluate its performance with cross-validation.
            2.- Permanently remove the feature with the best cross-validation score from the model.
            3.- If the best CV score has not improved in the last $tol iterations, end the procedure,
                otherwise go to step 1.

        Args:
            base_model (BaseEstimator): Base model for which optimal features will be searched.
            cv_splitter (_type_): _description_
            metric (str): Metric to optimize during the feature search procedure.
            tol (int, optional): Number of iterations after which the algorithm will terminate if no improvement in the CV score is obtained. Defaults to 3.
            n_jobs (int, optional): Number of cores used to run the algorithm. Defaults to 4.
        """

        self.cv_splitter = cv_splitter
        self.metric = metric
        self.tol = tol
        self.n_jobs = n_jobs
        self.model = base_model
        
    def fit(self, X, y, feature_names=None, feature_mask=[]):
        """Find the set of features that optimizes the cross-validation score of the model
        for the input training data.

        Args:
            X (np.array): Array of featurized training samples used to find the best performing features.
            y (np.array): Array of labels for each training sample.
            feature_names (list, optional): List of names for each feature used to characterize the samples. Defaults to None.
            feature_mask (list, optional): Mask used to indicate the indices of features that should not be considered in the selection process. Defaults to [].
        """

        avail_features = [i for i in range(X.shape[1]) if i not in feature_mask]
        best_features = deepcopy(avail_features)
        self.feature_history = {}
        iters_no_improve = 0
        best_score_all_iters = 0
        iter = 0
        
        while iters_no_improve < self.tol and len(best_features)>1:
            
            iter += 1
            best_score = 0
            
            for feature in avail_features:
                
                test_features = deepcopy(avail_features)
                test_features.remove(feature)
                scores = cross_val_score(self.model, X[:, test_features], y, cv=self.cv_splitter, scoring=self.metric, n_jobs=self.n_jobs)
                mean_score = scores.mean()
                
                if mean_score > best_score:
                    
                    best_score = mean_score
                    best_feature_remove = feature
            
            if best_score <= best_score_all_iters:
                iters_no_improve += 1
            else:
                best_score_all_iters = best_score
                self.best_results = {'score': best_score, 'features': deepcopy(best_features)}
                iters_no_improve = 0
            
            best_features.remove(best_feature_remove)
            avail_features.remove(best_feature_remove)
            
            if feature_names is None:
                self.feature_history[iter] = {'score': best_score,
                                              'features': deepcopy(best_features)}
            else:
                self.feature_history[iter] = {'score': best_score,
                                              'features': deepcopy(best_features),
                                              'feature_names': feature_names[best_features]}
