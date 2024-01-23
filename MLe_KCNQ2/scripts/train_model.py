import warnings
warnings.filterwarnings('ignore')

import pickle
from featurize import *
import os
import argparse
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from utils import prep_df, find_best_model, run_cv_analysis, scores_confidence_ranges
from ..models.Feature_Selectors import ForwardGreedySearch, BackwardGreedySearch
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from ..models.Featurized_Model import FeaturizedModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Models to train 
lr_model = LogisticRegression(max_iter=10000)
svc_model = SVC(probability=True)
rf_model = RandomForestClassifier()
knn_model = KNeighborsClassifier()
lda_model = LinearDiscriminantAnalysis()
gnb_model = GaussianNB()
lgb_model = LGBMClassifier()
xg_model = XGBClassifier(verbosity = 0, use_label_encoder=False)
gp_model = GaussianProcessClassifier()
cat_model = CatBoostClassifier(silent=True)

# Hyperparameter search space for each model type.
lr_params = [{'solver': ['lbfgs'], 'C': np.linspace(0.1, 10, 25), 'penalty': ['l2']},
            {'solver': ['saga'], 'penalty': ['l1'], 'C': np.linspace(0.1, 10, 25)},
            {'solver': ['saga'], 'penalty': ['elasticnet'], 'C': np.linspace(0.1, 10, 25), 'l1_ratio': np.linspace(00.1, 0.9, 8)}]
svc_params = {'C': np.linspace(0.1, 5, 12), 'kernel': ['rbf', 'linear', 'sigmoid']}
rf_params = {'n_estimators': np.arange(10, 250, 10), 'criterion': ['gini', 'entropy'], 'max_depth': np.arange(1, 11),
            'max_features': np.linspace(0.1, 1, 9)}
knn_params = {'n_neighbors': np.arange(1, 11), 'weights': ['uniform', 'distance'], 'p': [1, 2]}
lda_params = {'solver': ['lsqr']}
gnb_params = {'var_smoothing': [1e-10, 1e-9, 1e-8]}
lgb_params = {'num_leaves': np.arange(20, 32, 2), 'max_depth': np.array([-1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            'n_estimators': np.arange(30, 60, 10), 'reg_alpha': np.linspace(0, 1, 5), 'reg_lambda': np.linspace(0, 1, 5)}
cat_params = {'iterations': np.arange(40, 100, 10), 'learning_rate': np.linspace(0.01, 0.5, 10), 'l2_leaf_reg': np.linspace(0.01, 1, 10),
            'depth': np.arange(2, 8)}
xg_params = {'learning_rate': np.linspace(0.01, 0.2, 5), 'reg_alpha' : np.linspace(1, 25, 10), 'reg_lambda' : np.linspace(0, 5, 5),
            'subsample': np.linspace(0.8, 1, 5), 'n_estimators': np.arange(10, 90, 20)}
gp_params = {'kernel': [RBF(l) for l in np.logspace(-1, 1, 10)]}


def run(source_data: str, out_dir: str, model_type: str, num_splits: int, num_repeats: int,
        vfi_sigma: float, feature_selector: str, hyperparameter_selector: str,
        feature_search_iters: str, position_col: str, variant_col: str, label_col: str,
        gene_info_file: str, save_scaler: bool, num_cpu: int):
    """Train a set of algorithms for the specified samples of KCNQ2 variants. Optimal features
    and hyperparameters will be automatically generated and the optimized and trained model, as
    well as performance analysis saved at the output directory.

    Args:
        source_data (str): File containing the variants that will be used for training.
        out_dir (str): Directory where trained model and analysis files will be stored.
        model_type (str): Whether the model to be trained is for pathogenicity of phenotype prediction.
        num_splits (int): Number of splits used for cross-validation analysis.
        num_repeats (int): Number of times the cross-validation analysis will be repeated.
        vfi_sigma (float): Sigma value used for convolution used to create the VFI score.
            feature_selector (str): Algorithm used to select an optimal subset of features.
        Either forward of backward greedy search can be used.
        hyperparameter_selector (str): Algorithm used to select optimal hyperparameters. Either exhaustive or random search can be used.
        feature_search_iters (str): Number of iterations after which the greedy search will be stopped if not further performance gained is observed.
        position_col (str): Name of the column that contains the position of the variant in the source data file.
        variant_col (str): Name of the column that contains the name of the variant in the source data file.
        label_col (str): Name of the column that contains the label of the variant in the source data file.
        gene_info_file (str): File that contains allele frequency, residue conservation and pLDDT for KCNQ2.
        save_scaler (bool): Whether to save the standardizer used to standardize the features of the training dataset.
        num_cpu (int): Number of cores used.
    """

    # Crete output directory if it doesn't exist
    data_directory = os.path.join(out_dir, 'featurized_data')
    if not os.path.isdir(data_directory):
        os.makedirs(data_directory, exist_ok=True)

    # Validate sigma value.
    sigma = None if vfi_sigma == 'none' else float(vfi_sigma)
    assert sigma is None or sigma > 0, 'Invalid sigma value. Must be either "none" (to use only pLDDT) or a number bigger than 0.'

    # Featurize data.
    featurize(source_data, data_directory, sigma, position_col, variant_col,
              label_col, gene_info_file)

    # Load train data.
    if model_type == 'pathogenicity':
        train_data_benign = pd.read_csv(os.path.join(data_directory, 'ClinVar_Train_Benign_Featurized.csv'), index_col=0)
        train_data_pathogenic = pd.read_csv(os.path.join(data_directory, 'ClinVar_Train_Pathogenic_Featurized.csv'), index_col=0)
        train_data = pd.concat([train_data_benign, train_data_pathogenic])

    elif model_type == 'phenotype':
        train_data = pd.read_csv(os.path.join(data_directory, 'BFNC_EIEE_Train_Featurized.csv'), index_col=0)
        train_data['MyLabel'] = train_data['MyLabel'].replace('BFNC/EIEE', 'BFNC')

    # Load supplementary data.
    primates_train_df = pd.read_csv(os.path.join(data_directory, 'PrimateAD_Featurized.csv'), index_col=0)
    train_data = pd.concat([train_data, primates_train_df])

    # Prepare data for training
    train_data.drop_duplicates('Variant', inplace=True)
    X_train, y_train, feature_names = prep_df(train_data)
    variants = train_data['Variant']

    # List of models to train with their name and hyperparamter search space. To iterate easier.
    models_list = [
        ['Logistic Regression', lr_model, lr_params],
        ['K Nearest Neighbors', knn_model, knn_params],
        ['Support Vector Machine', svc_model, svc_params],
        ['Random Forest', rf_model, rf_params],
        ['Light GBM', lgb_model, lgb_params],
        ['CatBoost', cat_model, cat_params],
        ['XGBoost', xg_model, xg_params],
        ['Gaussian Process', gp_model, gp_params]
    ]
    models_lst = []
    all_scores_dict = {}
    all_errors_dict = {}
    summary_scores_dict = {}

    # Define standardizer.
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)

    if bool(save_scaler):
        pickle.dump(scaler, open(os.path.join(out_dir, 'Standardizer.sav'), 'wb'))

    # Training loop.
    perm_mutants_arr = variants.to_numpy()

    for model_name, base_model, params in models_list:
        
        selection_splitter = StratifiedKFold(n_splits=5, shuffle=True)
        analysis_splitter = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=num_repeats)
        if feature_selector == 'forward':
            feature_selector = ForwardGreedySearch(base_model, cv_splitter=selection_splitter, metric='roc_auc', tol=feature_search_iters, n_jobs=num_cpu)
        elif feature_selector == 'backward':
            feature_selector = BackwardGreedySearch(base_model, cv_splitter=selection_splitter, metric='roc_auc', tol=feature_search_iters, n_jobs=num_cpu)
        
        if hyperparameter_selector == 'exhastive':
            parameter_selector = GridSearchCV(base_model, params, scoring='roc_auc', n_jobs=num_cpu, cv=selection_splitter, verbose=0)
        elif hyperparameter_selector == 'random':
            parameter_selector = RandomizedSearchCV(base_model, params, n_iter=100, scoring='roc_auc', n_jobs=num_cpu, cv=selection_splitter, verbose=0)

        best_model, feature_idx, _ = find_best_model(X_train_std, y_train, feature_selector, parameter_selector,
                                                        feature_names=feature_names)
        best_model = FeaturizedModel(best_model, feature_idx, model_name, feature_names[feature_idx])

        scores_df, mutant_errors_df = run_cv_analysis(best_model, X_train_std, y_train, mut_names=perm_mutants_arr, cv_splitter=analysis_splitter)
        
        best_model.fit(X_train_std, y_train)
        
        models_lst.append(best_model)
        all_scores_dict[model_name] = scores_df
        summary_scores_dict[model_name, 'Mean'] = scores_df.mean(axis=0)
        summary_scores_dict[model_name, 'Std'] = scores_df.std(axis=0)
        
        all_errors_dict[model_name] = mutant_errors_df.loc[:, 'Error Counts']
        
        pickle.dump(best_model, open(os.path.join(out_dir, f'{model_name}.sav'), 'wb'))
        
        print(f'Finished {model_name}')
    
    # Summarize results.
    summary_scores_df = pd.DataFrame(summary_scores_dict, index=['AUC-ROC', 'F1-Score', 'Balanced-Accuracy', 'Sensitivity', 'Specificity'])
    summary_scores_df.round(decimals=3)
    summary_scores_df.to_csv(os.path.join(out_dir, 'Scores.csv'))

    # Create and save plot of score distributions.
    metrics = ['AUC-ROC', 'Balanced-Accuracy', 'Sensitivity', 'Specificity']
    fig, _ = scores_confidence_ranges(all_scores_dict, metrics)
    fig.savefig(os.path.join(out_dir, 'confidence_intervals.png'))

    # Check most commonly used features.
    features_count_list = []

    for model in models_lst:
        
        features_count_list.extend(model.feature_names)

    counter = Counter(features_count_list)
    features_counts = counter.most_common()

    # Create and save plot.
    used_features = [name for name, _ in features_counts]
    counts = [count for _, count in features_counts]

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.bar(np.arange(len(counts)), counts)
    ax.set_xticks(np.arange(len(counts)))
    ax.set_xticklabels(used_features, rotation=90)

    ax.set_ylabel('Counts', fontsize=14, fontweight='bold', labelpad=15)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'Feature_Uses.png'))

    # Check which variants are most difficult to predict when they are in the test set
    errors_summary = pd.concat(all_errors_dict.values(), axis=1)
    errors_summary.columns = all_errors_dict.keys()
    errors_summary['Total'] = errors_summary.sum(axis=1)
    errors_summary['Ratio'] = errors_summary['Total'] / (num_repeats * (len(models_lst)))

    errors_summary.to_csv(os.path.join(out_dir, 'Errors_Summary.csv'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--source_data', required=True, type=str, help='Data file containing the variants used for training.')
    parser.add_argument('-vs', '--vfi_sigma', required=True, type=str, help='Value of sigma used to define the VFI feature.')
    parser.add_argument('-o', '--out_dir', required=True, type=str, help='Directory where results will be stored.')
    parser.add_argument('-m', '--model_type', required=True, type=str, choices={'pathogenicity', 'phenotype'}, help='Wheter to train a pathogenicity or phenotype prediction model.')
    parser.add_argument('-s', '--num_splits', type=int, default=5, help='Number of splits used for the cross-validation analysis.')
    parser.add_argument('-r', '--num_repeats', type=int, default=25, help='Number of repeats used for the cross-validation analysis.')
    parser.add_argument('-fs', '--feature_selector', default='forward', type=str, choices={'forward', 'backward'}, help='Whether to perform forward or backward greedy search for feature selection.')
    parser.add_argument('-tol', '--feature_search_iters', type=int, default=10, help='Number of iterations without training improvement that the feature selection algorithm will keep searching before stopping.')
    parser.add_argument('-hp', '--hyperparameter_selector', default='random', type=str, choices={'exhaustive', 'random'}, help='Whether to perform exhaustive or random grid search for hyperparameter selection.')
    parser.add_argument('-ss', '--save_scaler', type=int, default=0, choices={0, 1}, help='Whether to save the standardizer.')
    parser.add_argument('-vc', '--variant_col', type=str, default='Variant', help='Name of the column in the input file that contains the names of the variants to featurize.')
    parser.add_argument('-pc', '--position_col', type=str, default='Position', help='Name of the column in the input file that contains the positions of the variants to featurize.')
    parser.add_argument('-lc', '--label_col', type=str, default='My_Label', help='Name of the column in the input file that contains the labels of the variants to featurize.')
    parser.add_argument('-gi', '--gene_info_file', type=str, required=True, help='File containing allele frequency, residue conservation and pLDDT information for KCNQ2.')
    parser.add_argument('-np', '--num_cpu', type=int, default=8, help='Number of cores used for the training.')

    args = parser.parse_args()

    run(args.source_data, args.out_dir, args.model_type, args.num_splits, args.num_repeats, args.vfi_sigma,
        args.feature_selector, args.hyperparameter_selector, args.feature_search_iters, args.position_col,
        args.variant_col, args.label_col, args.gene_info_file, args.save_scaler, args.num_cpu)
    