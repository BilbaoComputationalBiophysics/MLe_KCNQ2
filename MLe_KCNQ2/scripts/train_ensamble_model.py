import warnings
warnings.filterwarnings('ignore')

import os
import argparse
from typing import Tuple
from copy import deepcopy
import pickle
import numpy as np
import pandas as pd
from utils import prep_df, run_cv_analysis
from ..models.Ensemble_Models import SoftStackedClassifier, SoftVotingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

def load_pathogenic_data(data_dir: str, scaler: StandardScaler) -> Tuple[np.array, np.array, np.array, np.array]:
    """Function that loads data for pathogenicity prediction into numpy arrays
    and standardizes it.

    Args:
        data_dir (str): Directory with source data.
        scaler (StandardScaler): Standardizer for input data.

    Returns:
        Tuple[np.array, np.array, np.array, np.array]: Arrays with training features, labels, variant names and feature names.
    """

    # Load train data.
    train_data_benign = pd.read_csv(os.path.join(data_dir, 'ClinVar_Train_Benign_Featurized.csv'), index_col=0)
    train_data_pathogenic = pd.read_csv(os.path.join(data_dir, 'ClinVar_Train_Pathogenic_Featurized.csv'), index_col=0)
    train_data = pd.concat([train_data_benign, train_data_pathogenic])

    # Load supplementary data.
    primates_train_df = pd.read_csv(os.path.join(data_dir, 'PrimateAD_Featurized.csv'), index_col=0)
    train_data = pd.concat([train_data, primates_train_df])
    train_data.drop_duplicates('Variant', inplace=True)

    # Standardize data.
    X_train, y_train, feature_names = prep_df(train_data)
    train_variants = train_data['Variant'].to_numpy()

    X_train_std = scaler.transform(X_train)

    return X_train_std, y_train, train_variants, feature_names


def load_phenotype_data(data_dir: str, scaler: StandardScaler) -> Tuple[np.array, np.array, np.array, np.array]:
    """Function that loads data for phenotype prediction into numpy arrays
    and standardizes it.

    Args:
        data_dir (str): Directory with source data.
        scaler (StandardScaler): Standardizer for input data.

    Returns:
        Tuple[np.array, np.array, np.array, np.array]: Arrays with training features, labels, variant names and feature names.
    """

    # Load train data.
    train_data = pd.read_csv(os.path.join(data_dir, 'BFNC_EIEE_Train_Featurized.csv'), index_col=0)
    train_data.drop_duplicates('Variant', inplace=True)

    # Map labels
    train_data['MyLabel'] = train_data['MyLabel'].replace('BFNC/EIEE', 'BFNC')

    # Standardize data.
    X_train, y_train, feature_names = prep_df(train_data)
    train_variants = train_data['Variant']
    X_train = X_train[:, 40:]

    X_train_std = scaler.transform(X_train)

    return X_train_std, y_train, train_variants, feature_names

def train_ensemble(ensemble: BaseEstimator, available_models: list, n_splits: int, n_repeats: int,
                   X_train: np.array, y_train: np.array, variants: np.array) -> Tuple[BaseEstimator, pd.DataFrame]:
    """Given a set of trained models, trains an ensemble model by selecting the subset of trained
    models that, when aggregated according to this ensemble model, maximize the AUC-ROC of the ensemble.

    Args:
        ensemble (BaseEstimator): Ensemble model used to aggregate the trained models.
        available_models (list): Trained models to be aggregated.
        num_splits (int): Number of splits used for cross-validation analysis.
        num_repeats (int): Number of times the cross-validation analysis will be repeated.
        X_train (np.array): Array of features for training samples.
        y_train (np.array): Array of labels for training samples.
        variants (np.array): Array of variants that belong to the training set.

    Returns:
        Tuple[BaseEstimator, pd.DataFrame]: Optimal ensemble model and dataframe with performance metrics from CV analysis.
    """

    remaining_models = deepcopy(list(available_models.values()))
    selected_models = []
    scores_hist = np.empty((len(available_models), 5))
    models_hist = []

    for n in range(len(available_models)):

        best_score = 0

        for model in remaining_models:

            test_models = deepcopy(selected_models)
            test_models.append(model)

            if ensemble is None:
                ensemble_model = SoftVotingClassifier(test_models)
            else:
                ensemble_model = SoftStackedClassifier(test_models, ensemble)
            
            splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
            scores_df, _ = run_cv_analysis(ensemble_model, X_train, y_train, variants, splitter)
            score = scores_df['AUC-ROC'].mean(axis=0)
            
            if score > best_score:
                best_score = score
                best_new_model = model

        models_hist.append(best_new_model)
        selected_models.append(best_new_model)
        remaining_models.remove(best_new_model)
        
        scores_hist[n, :] = scores_df.mean(axis=0).to_numpy()
        
        print(f'Added Model: {best_new_model.name}, Score: {best_score:.4f}')

    scores_hist_df = pd.DataFrame(scores_hist, columns=['AUC-ROC', 'F1-Score', 'Balanced-Accuracy', 'Sensitivity', 'Specificity'],
                                  index=[model.name for model in models_hist])
    
    if ensemble is None:
        best_models_set = selected_models[:scores_hist[:, 2].argmax() + 1]
        final_model = SoftVotingClassifier(best_models_set)
    else:
        best_models_set = selected_models[:scores_hist[:, 2].argmax() + 1]
        final_model = SoftStackedClassifier(best_models_set, ensemble)

    return final_model, scores_hist_df

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--models_dir', type=str, required=True, help='Directory that contains the models used to construct the ensemble.')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Directory where ensemble models will be stored.')
    parser.add_argument('-t', '--train_type', type=str, required=True, choices={'pathogenicity', 'phenotype'}, help='Whether the training is for a pathogeneity or phenotype model.')
    parser.add_argument('-p', '--model_type', type=str, default='estimator', choices={'estimator', 'voting'},
                        help='Whether to train an estimator that takes the predictions of other models as features, or a voting estimator that averages the predictions of other models.')
    parser.add_argument('-s', '--num_splits', type=int, default=5, help='Number of splits used for the cross-validation analysis.')
    parser.add_argument('-r', '--num_repeats', type=int, default=25, help='Number of repeats used for the cross-validation analysis.')

    args = parser.parse_args()

    assert os.path.isdir(args.models_dir)

    data_dir = os.path.join(args.models_dir, 'featurized_data')

    assert os.path.isdir(data_dir), 'Model directory does not contain input data directory.'

    # Load models
    base_models = {}

    for file in os.listdir(args.models_dir):
        
        if file.endswith('.sav') and not file == 'Standardizer.sav':

            model = pickle.load(open(os.path.join(args.models_dir, file), 'rb'))
            base_models[model.name] = model
    
    scaler = pickle.load(open(os.path.join(args.models_dir, 'Standardizer.sav'), 'rb'))

    if args.model_type == 'estimator':
        # Define ensembles to test
        ensembles = {
            'Logistic Regression': LogisticRegression(max_iter=10000),
            'KNN 3 Neighbors': KNeighborsClassifier(n_neighbors=3),
            'KNN 5 Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Gaussian Naive Bayes': GaussianNB()
        }
    else:
        ensembles = {'Soft Voting': None}

    # Load Data
    num_splits = args.num_splits
    num_repeats = args.num_repeats
    all_scores_dict = {}
    all_errors_dict = {}
    summary_scores_dict = {}

    if args.train_type == 'pathogenicity':
        X_train_std, y_train, train_variants, feature_names = load_pathogenic_data(data_dir, args.data_type, args.benign_train, scaler)
    else:
        X_train_std, y_train, train_variants, feature_names = load_phenotype_data(data_dir, scaler)
    
    train_variants = train_variants.to_numpy()

    for ensemble_name, ensemble_model in ensembles.items():

        model, scores_hist_df = train_ensemble(ensemble_model, base_models, num_splits, num_repeats, X_train_std, y_train, train_variants)
        scores_hist = scores_hist_df['AUC-ROC']

        splitter = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=num_repeats)
        scores_df, errors_df = run_cv_analysis(model, X_train_std, y_train, train_variants, splitter)
        
        all_scores_dict[ensemble_name] = scores_df
        summary_scores_dict[ensemble_name, 'Mean'] = scores_df.mean(axis=0)
        summary_scores_dict[ensemble_name, 'Std'] = scores_df.std(axis=0)

        all_errors_dict[ensemble_name] = errors_df.loc[:, 'Error Counts']

        model.fit(X_train_std, y_train)
        pickle.dump(model, open(os.path.join(args.output_dir, f'{ensemble_name}.sav'), 'wb'))

        scores_hist_df.to_csv(os.path.join(args.output_dir, f'{ensemble_name}_Scores.csv'), index=None)

        print(f'Finished {ensemble_name}')
    
    summary_scores_df = pd.DataFrame(summary_scores_dict, index=['AUC-ROC', 'F1-Score', 'Balanced-Accuracy', 'Sensitivity', 'Specificity'])
    summary_scores_df.to_csv(os.path.join(args.output_dir, 'Scores.csv'))

