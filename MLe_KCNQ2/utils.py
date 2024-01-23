import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Union, Tuple
from sklearn.base import BaseEstimator
from sklearn.model_selection import validation_curve
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from imblearn.over_sampling import RandomOverSampler
from models.Feature_Selectors import ForwardGreedySearch, BackwardGreedySearch


def sensitivity_score(y_true: np.array, y_pred: np.array) -> float:
    """Calculates the sensitivity score for a set of predicted labels. The sensitivity score is given by:

        Sensitivity = True Positives / (True Positive + False Negatives)

    Args:
        y_true (np.array): Array of correct labels.
        y_pred (np.array): Array of predicted labels

    Returns:
        float: Sensitivity score for the given predictions.
    """
    
    true_positives = np.where(y_true==1)[0]
    
    return (y_pred[true_positives] == y_true[true_positives]).sum() / true_positives.size


def specificity_score(y_true: np.array, y_pred: np.array) -> float:
    """Calculates the specificity score for a set of predicted labels. The specificity score is given by:

        Specificity = True Negatives / (True Negatives + False Positives)

    Args:
        y_true (np.array): Array of correct labels.
        y_pred (np.array): Array of predicted labels

    Returns:
        float: Specificity score for the given predictions.
    """
    
    true_negatives = np.where(y_true==0)[0]
    
    return (y_pred[true_negatives] == y_true[true_negatives]).sum() / true_negatives.size


def prep_df(df: pd.DataFrame) -> Tuple[np.array, np.array, np.array]:
    """From a dataframe of samples, generates an array of features, labels and feature names,
    which can be used for training/prediction.

    Args:
        df (pd.DataFrame): Dataframe containing the samples and their features.

    Returns:
        Tuple[np.array, np.array, np.array]: Array of features, labels and feature names.
    """
    
    y = pd.Categorical(df['MyLabel']).codes
    
    drop_cols = ['MyLabel', 'Variant', 'Position']
        
    df = df.drop(columns=drop_cols)
    X = df.to_numpy()
    feature_names = np.array(df.columns)
    
    return X, y, feature_names


def run_cv_analysis(model: BaseEstimator, X: np.array, y: np.array, mut_names: list, cv_splitter,
                    oversampling: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Performs repeated cross-validation to estimate the performance of a models based on multiple metrics.

    Args:
        model (BaseEstimator): Model whose performance will be assessed.
        X (np.array): Array of features for each sample in the training set.
        y (np.array): Array of labels for each sample in the test set.
        mut_names (list): List of names for each variant in the training set, following and
            initial amino-acid/positions/final amino-acid format (e.g. A125E)
        cv_splitter (_type_): Repeated splitter used to generate different train and test sets
            during cross-validation.
        oversampling (bool, optional): Whether to use oversampling to balance the number of
            positive and negative samples. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Dataframe of performance metrics obtained during
            cross-validation and number of times each variant was incorrectly predicted during
            the CV procedure.
    """
    
    # Dataframes containing counts of errors commited for each variant, and scores during cross-validation iterations.
    mutant_errors_df = pd.DataFrame(np.zeros((len(mut_names), 2), dtype=int), index=mut_names, columns=['Total Predictions', 'Error Counts'])
    scores_df = pd.DataFrame(columns=['AUC-ROC', 'F1-Score', 'Balanced-Accuracy', 'Sensitivity', 'Specificity'])
    
    # Dictionary with metrics that will be assessed during CV, and the functions used to calculate them.
    metric_dict = {'AUC-ROC': roc_auc_score, 'F1-Score': f1_score, 'Balanced-Accuracy': balanced_accuracy_score, 'Sensitivity': sensitivity_score,
                   'Specificity': specificity_score}
    
    if oversampling:
        ros = RandomOverSampler()
    
    for n, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y)):

        # Use indices created by cv_splitter to split the training set into multiple
        # sub-train and sub-test sets.
        X_train = X[train_idx, :]
        y_train = y[train_idx]
        X_test = X[test_idx, :]
        y_test = y[test_idx]
        test_mutants = mut_names[test_idx]
        
        if oversampling:

            # Perform oversampling to balance training subset.
            X_train, y_train = ros.fit_resample(X_train, y_train)
        
        # Fit model using subset of training samples for this iteration, then predict subset of test samples.
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Find variants in the test subset that were incorrectly predicted.
        error_idx = np.where(y_test != y_pred)[0]
        mutant_errors_df.loc[test_mutants, 'Total Predictions'] += 1
        mutant_errors_df.loc[test_mutants[error_idx], 'Error Counts'] += 1
        
        # Measure all metrics for this CV iteration.
        for metric_name, metric in metric_dict.items():
            
            if metric_name == 'AUC-ROC':
                scores_df.loc[n, metric_name] = metric(y_test, y_proba[:, 1])
            else:
                scores_df.loc[n, metric_name] = metric(y_test, y_pred)
        
    return scores_df, mutant_errors_df
    

def find_best_model(X: np.array, y: np.array, feature_selector: Union[ForwardGreedySearch, BackwardGreedySearch],
                    parameter_selector, feature_names: list=None, feature_mask: list=[]) -> Tuple[BaseEstimator, np.array, dict]:
    """Using a feature and hyperparameter selector, and a set of training samples, find optimal features
    and hyperparamters.

    Args:
        X (np.array): Array of features for training samples.
        y (np.array): Array of labels for training.
        feature_selector (Union[ForwardGreedySearch, BackwardGreedySearch]): Algorithm used to find optimal features.
        parameter_selector (_type_): Algorithm used to find optimal hyperparamters.
        feature_names (list, optional): List of names for all available features. Defaults to None.
        feature_mask (list, optional): Mask of features that should not be considered during feature selection. Defaults to [].

    Returns:
        Tuple[BaseEstimator, np.array, dict]: Final model with optimal features and hyperparamters, array of indices for selected
            features and dictionay with summary of results from the feature and hyperparamter search.
    """

    # Find optimal features using feature selector.
    feature_selector.fit(X, y, feature_names=feature_names, feature_mask=feature_mask)
    best_feat_score = feature_selector.best_results['score']
    feature_idx = feature_selector.best_results['features']
    X_feat = X[:, feature_idx]
    
    # Find optimal hyperparameters using parameter selector.
    parameter_selector.fit(X_feat, y)
    best_model = parameter_selector.best_estimator_
    best_param_score = parameter_selector.best_score_
    best_params = parameter_selector.best_params_
    
    # Save summary of results in dictionary.
    summary = {'features_score': best_feat_score, 'hyperparameters_score': best_param_score, 'best_hyperparamters': best_params}
    
    if feature_names is not None:
        summary['best_feature_names'] = feature_names[feature_idx]
    
    return best_model, feature_idx, summary


def hp_val_test(model: BaseEstimator, X_train: np.array, y_train: np.array, params: dict,
                cv_splitter, metric: str, n_jobs: int=4) -> Tuple[Figure, Axes]:
    
    ncols = 2 if len(params) >= 2 else 1
    nrows = len(params)//2 + (1 if len(params)%2==1 else 0)
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 8))
    
    for n, (hp, search_space) in enumerate(params.items()):
        
        j = n//2
        i = n%2
        
        _, test_scores = validation_curve(model, X_train, y_train, param_name=hp, param_range=search_space, scoring=metric, cv=cv_splitter, n_jobs=n_jobs)
        
        mean_scores = test_scores.mean(axis=1)
        std_scores = test_scores.std(axis=1)
        
        if ncols==1 and nrows==1:
            ax.plot(search_space, mean_scores, color='darkblue')
            ax.fill_between(search_space, mean_scores, mean_scores - std_scores, color='blue', alpha=0.4)
            ax.fill_between(search_space, mean_scores, mean_scores + std_scores, color='blue', alpha=0.4)
            ax.set_xlabel(hp, fontweight='bold')
            ax.set_ylabel(metric, fontweight='bold')
        
        elif nrows==1 and ncols==2:
            ax[i].plot(search_space, mean_scores, color='darkblue')
            ax[i].fill_between(search_space, mean_scores, mean_scores - std_scores, color='blue', alpha=0.4)
            ax[i].fill_between(search_space, mean_scores, mean_scores + std_scores, color='blue', alpha=0.4)
            ax[i].set_xlabel(hp, fontweight='bold')
            ax[i].set_ylabel(metric, fontweight='bold')
        
        else:
            ax[j, i].plot(search_space, mean_scores, color='darkblue')
            ax[j, i].fill_between(search_space, mean_scores, mean_scores - std_scores, color='blue', alpha=0.4)
            ax[j, i].fill_between(search_space, mean_scores, mean_scores + std_scores, color='blue', alpha=0.4)
            ax[j, i].set_xlabel(hp, fontweight='bold')
            ax[j, i].set_ylabel(metric, fontweight='bold')

    fig.tight_layout()

    return fig, ax

def scores_confidence_ranges(scores_dict: dict, metrics: list) -> Tuple[Figure, Axes]:
    """Plot the mean for each metric from samples obtained during cross-validation, along with
    70 and 95% quantiles.

    Args:
        scores_dict (dict): Dictionary containing Dataframes with CV results for a set of algorithms.
        metrics (list): List of metrics whose results will be shown.

    Returns:
        Tuple[Figure, Axes]: Figure and Axes objects for the resulting figure.
    """
    
    colors = ['green', 'red', 'blue', 'cyan', 'violet', 'darkorange', 'black', 'mediumpurple', 'lightsalmon', 'gold', 'grey']

    ncols = 2 if len(metrics) >= 2 else 1
    nrows = len(metrics) // 2 + (1 if len(metrics)%2 == 1 else 0)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(14, 8))

    for metric, ax in zip(metrics, axes.ravel()):
        
        for k, df in enumerate(reversed(scores_dict.values())):
            
            ax.plot([df[metric].quantile(0.025), df[metric].quantile(0.975)], [k, k], linewidth=1.2, color=colors[k])
            ax.plot([df[metric].quantile(0.15), df[metric].quantile(0.85)], [k, k], linewidth=2.5, color=colors[k])
            ax.scatter(df[metric].mean(), k, color=colors[k], s=50)
        
        ax.set_xlabel(metric, fontweight='bold', fontsize=13, labelpad=15)
        ax.set_yticks(np.arange(len(scores_dict)))
        ax.set_yticklabels(reversed(list(scores_dict.keys())))
        ax.set_xlim([0.0, 1.0])
        
    fig.tight_layout()

    return fig, axes

    