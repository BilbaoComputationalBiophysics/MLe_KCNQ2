import os
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
from utils import sensitivity_score, specificity_score, prep_df
from ..models.Ensemble_Models import SoftStackedClassifier

def test_model(model: BaseEstimator, scaler: StandardScaler, X_test: np.array,
               y_test: np.array) -> dict:
    """Given a model and a set of featurized samples, makes predictions for those
    samples and computes performance metrics for those predictions based on the true
    labels given for those samples.

    Args:
        model (BaseModel): Model to test.
        scaler (StandardScaler): Standardizer of input features.
        X_test (np.array): Array of features for test samples.
        y_test (np.array): Array of true labels for test samples.

    Returns:
        dict: Dictionary of performance metrics for input model.
    """
    # Standardize data
    X_test_std = scaler.transform(X_test)

    # Make predictions
    y_pred = model.predict(X_test_std)
    y_prob = model.predict_proba(X_test_std)[:, 1]

    # Compute metrics
    metrics = {
        'AUC-ROC': roc_auc_score(y_test, y_prob),
        'F1-Score': f1_score(y_test, y_pred),
        'Balanced-Accuracy': balanced_accuracy_score(y_test, y_pred),
        'Sensitivity': sensitivity_score(y_test, y_pred),
        'Specificity': specificity_score(y_test, y_pred),
    }

    return metrics

def get_predictions(model: BaseEstimator, scaler: StandardScaler, X_train: np.array,
                    y_train: np.array, train_variants: np.array, X_test: np.array,
                    y_test: np.array, test_variants: np.array) -> pd.DataFrame:
    """Produces a dataframe of predicted labels and probabilities for a all available data,
    both train and test samples.

    Args:
        model (BaseEstimator): Model to use to make predictions.
        scaler (StandardScaler): Standardizer used to standardize the input data.
        X_train (np.array): Array of featurized training samples.
        y_train (np.array): Array of labels for training samples.
        train_variants (np.array): Array of names for variants in the training set.
        X_test (np.array): Array of featurized test samples.
        y_test (np.array): Array of labels for test samples.
        test_variants (np.array): Array of names for variants in the test set.

    Returns:
        pd.DataFrame: Dataframe containing all predictions made by the model.
    """

    # Standardize data
    X_train_std = scaler.transform(X_train)
    X_test_std = scaler.transform(X_test)

    # Make predictions
    y_pred_train = model.predict(X_train_std)
    y_prob_train = model.predict_proba(X_train_std)[:, 1]
    y_pred_test = model.predict(X_test_std)
    y_prob_test = model.predict_proba(X_test_std)[:, 1]

    # Assemble dataframes with the names of the predicted variants, the predicted labels, and probabilities, and the true labels.
    train_preds = pd.DataFrame(np.vstack([train_variants, y_pred_train, y_prob_train, y_train]).T,
                               columns=['Variant', 'Prediction', 'Score', 'Label'])
    train_preds['Group'] = 'Train'
    test_preds = pd.DataFrame(np.vstack([test_variants, y_pred_test, y_prob_test, y_test]).T,
                              columns=['Variant', 'Prediction', 'Score', 'Label'])
    test_preds['Group'] = 'Test'

    all_preds = pd.concat([train_preds, test_preds])

    return all_preds

def test(in_dir: str, out_dir: str, data_dir: str, data_type: str) -> None:
    """Collects all data in data_dir directory, from both the train and test sets, loads models saved in
    in_dir directory, and predicts all variants with said models. Then, computes performance metrics for
    the test set specifically. Finally, saves all predicted labels and probabilities, as well as a csv with
    the performance metrics of all models.

    Args:
        in_dir (str): Directory were the models to be tested as stored.
        out_dir (str): Directory were the outputs will be stored.
        data_dir (str): Directory were the variants from the train and test set are stored.
        data_type (str): Whether the models to be tested are pathogenicity or phenotype prediction models.
    """

    # Load all variants belonging to both the train and test sets.
    if data_type == 'pathogenicity':
        train_clinvar_pathogenic = pd.read_csv(os.path.join(data_dir, 'ClinVar_Train_Pathogenic_Featurized.csv'), index_col=0)
        train_clinvar_benign = pd.read_csv(os.path.join(data_dir, 'ClinVar_Train_Benign_Featurized.csv'), index_col=0)
        train_primate = pd.read_csv(os.path.join(data_dir, 'PrimateAD_Featurized.csv'), index_col=0)
        train_data = pd.concat([train_clinvar_pathogenic, train_clinvar_benign, train_primate])
        test_data = pd.read_csv(os.path.join(data_dir, 'ClinVar_Test_Featurized.csv'), index_col=0)
    
    else:
        train_data = pd.read_csv(os.path.join(data_dir, 'BFNC_EIEE_Train_Featurized.csv'), index_col=0)
        test_data = pd.read_csv(os.path.join(data_dir, 'BFNC_EIEE_Test_Featurized.csv'), index_col=0)
        train_data['MyLabel'] = train_data['MyLabel'].replace('BFNC/EIEE', 'BFNC')
        test_data['MyLabel'] = test_data['MyLabel'].replace('BFNC/EIEE', 'BFNC')
    
    train_variants = train_data['Variant'].to_numpy()
    test_variants = test_data['Variant'].to_numpy()

    X_train, y_train, _ = prep_df(train_data)
    X_test, y_test, _ = prep_df(test_data)

    # If a phenotype prediction model is to be tested, eliminate amino-acid code features.
    if data_type == 'phenotype':
        X_train = X_train[:, 40:]
        X_test = X_test[:, 40:]

    # Load standardizer.
    standard_scaler = pickle.load(open(os.path.join(in_dir, 'Standardizer.sav'), 'rb'))

    # Load all models in input directory and predict variants with them.
    metrics = {}
    excel_writer = pd.ExcelWriter(os.path.join(out_dir, 'Predictions.xlsx'), engine='xlsxwriter')

    for file in os.listdir(in_dir):

        if file.endswith('.sav') and file != 'Standardizer.sav':

            model = pickle.load(open(os.path.join(in_dir, file), 'rb'))

            if isinstance(model, SoftStackedClassifier):
                metrics[file.split('.')[0]] = test_model(model, standard_scaler, X_test, y_test)
            else:
                metrics[model.name] = test_model(model, standard_scaler, X_test, y_test)

            all_preds = get_predictions(model, standard_scaler, X_train, y_train, train_variants, X_test, y_test, test_variants)
            all_preds.to_excel(excel_writer, sheet_name=file.split('.')[0], index=False)
    
    excel_writer.save()
    results_df = pd.DataFrame(metrics)
    results_df.to_csv(os.path.join(out_dir, 'Test_Scores.csv'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--in_dir', '-i', type=str, required=True, help='Directory where models to be tested are located.')
    parser.add_argument('--out_dir', '-o', type=str, required=True, help='Directory where test scores will be saved.')
    parser.add_argument('--data_dir', '-d', type=str, required=True, help='Directory where test data is stored.')
    parser.add_argument('--data_type', '-dt', type=str, required=True, choices={'pathogenicity', 'phenotype'}, help='Which type of data to test.')
    parser.add_argument('-c', '--classification', default='binary', type=str, choices={'binary', 'multi_class'},
                        help='Whether to treat the encephalopathy classification problem as binary or multi-class.')

    args = parser.parse_args()

    assert os.path.isdir(args.in_dir), 'Directory does not exist or cannot be found.'
    assert os.path.isdir(args.data_dir), 'Directory does not contain data for testing.'

    test(args.in_dir, args.out_dir, args.data_dir, args.data_type)
