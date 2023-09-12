import json
import os
import sys
import time

import pandas as pd
from pytictoc import TicToc
from sklearn.datasets import fetch_openml
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

from data_augmentation.pipeline import synthesize_data


EXEC_TIME_MINUTES = 10
EXEC_TIME_SECONDS = EXEC_TIME_MINUTES*60
SEED = 42
TIMER = TicToc()

def get_dataset_ref():
    dataset_ref = None
    if len(sys.argv) > 4:
        print('usage: python common.py dataset_ref')
    else:
        try:
            dataset_ref = int(sys.argv[1])
        except:
            dataset_ref = str(sys.argv[1])
    return dataset_ref

def get_synthesizer_ref():
    synthesizer_ref = None
    if len(sys.argv) > 4:
        print('usage: python common.py synthesizer_ref')
    elif len(sys.argv) == 2:
        synthesizer_ref = "none"
    else:    
        synthesizer_ref = str(sys.argv[2])
    return synthesizer_ref

def get_synthesizer_limit():
    if len(sys.argv) > 4:
        print('usage: python common.py dataset_ref')
    else:
        try:
            synthesizer_limit = int(sys.argv[3])
        except:
            synthesizer_limit = str(sys.argv[3])
    return synthesizer_limit

def infer_task_type(y_test):
    num_classes = len(set(y_test))
    if num_classes == 1:
        raise Exception('Malformed data set; num_classes == 1')
    elif num_classes == 2:
        task_type = 'binary'
    else:
        task_type = 'multiclass'
    return task_type

def load_data_delegate():
    if isinstance(get_dataset_ref(), int):
        return load_openml()
    elif isinstance(get_dataset_ref(), str):
        return load_csv()
    else:
        raise Exception('dataset_ref must be int (OpenML) or str (local CSV)')

def load_csv():
    base_folder = os.path.join(os.path.dirname(__file__), 'datasets', get_dataset_ref())
    dataset = pd.read_csv(base_folder+".csv").infer_objects()
    # The target column must be the last one
    target_name = dataset.columns[-1]
    X = dataset.drop(columns=target_name)
    y = dataset[target_name]
    y.name = target_name

    return X, y

def load_openml():
    dataset = fetch_openml(data_id=get_dataset_ref(), return_X_y=False)

    X, y = dataset.data, dataset.target
    target_name = dataset.target_names[0]
    for col in X.columns.values:
        if X[col].dtype.name == 'category':
            X[col] = pd.Series(pd.factorize(X[col])[0])
    y = pd.Series(pd.factorize(y)[0])
    y.name = target_name

    return X, y

def generate_synthetic_dataset(X: pd.DataFrame, y: pd.Series):
    synthesizer_name = get_synthesizer_ref()
    synthesizer_limit = get_synthesizer_limit()

    print(synthesizer_name)
    print(synthesizer_limit)

    if synthesizer_name == "none" and synthesizer_limit != "none":
        return None, None, None, None, None, None

    target_name = y.name

    dataset = X.copy()
    dataset[target_name] = y

    synthetic_data = pd.DataFrame()
    if synthesizer_name != "none":
        synthetic_data = synthesize_data(dataset, target_name, synthesizer_name, synthesizer_limit)
    complete_data = pd.concat([dataset, synthetic_data], ignore_index=True)

    X = complete_data.drop(columns=[target_name])
    y = complete_data[target_name]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    return X_train, X_test, y_train, y_test, complete_data, synthetic_data

def calculate_score(metric, y_true, y_pred, **kwargs):
	try:
		return metric(y_true, y_pred, **kwargs)
	except:
		return -1.0

def collect_and_persist_results(
        y_test, y_pred, training_time, test_time, framework="unknown",
        complete_data=None, synthetic_data=None, synthesizer_time=0):
    results = {}
    results.update({"framework":                                framework})
    results.update({"accuracy_score":                           calculate_score(accuracy_score, y_test, y_pred)})
    results.update({"average_precision_score":                  calculate_score(average_precision_score, y_test, y_pred)})
    results.update({"balanced_accuracy_score":                  calculate_score(balanced_accuracy_score, y_test, y_pred)})
    results.update({"cohen_kappa_score":                        calculate_score(cohen_kappa_score, y_test, y_pred)})
    results.update({"f1_score_macro":                           calculate_score(f1_score, y_test, y_pred, average="macro")})
    results.update({"f1_score_micro":                           calculate_score(f1_score, y_test, y_pred, average="micro")})
    results.update({"f1_score_weighted":                        calculate_score(f1_score, y_test, y_pred, average="weighted")})
    results.update({"matthews_corrcoef":                        calculate_score(matthews_corrcoef, y_test, y_pred)})
    results.update({"precision_score":                          calculate_score(precision_score, y_test, y_pred)})
    results.update({"recall_score":                             calculate_score(recall_score, y_test, y_pred)})
    results.update({"roc_auc_score":                            calculate_score(roc_auc_score, y_test, y_pred)})
    results.update({"coverage_error":                           calculate_score(coverage_error, y_test, y_pred)})
    results.update({"label_ranking_average_precision_score":    calculate_score(label_ranking_average_precision_score, y_test, y_pred)})
    results.update({"label_ranking_loss":                       calculate_score(label_ranking_loss, y_test, y_pred)})
    results.update({"training_time":                            time.strftime("%H:%M:%S", time.gmtime(training_time))})
    results.update({"test_time":                                time.strftime("%H:%M:%S", time.gmtime(test_time))})
    results.update({"synthesizer_time":                         time.strftime("%H:%M:%S", time.gmtime(synthesizer_time))})
    print(results)
    if not os.path.exists(f'./results/{get_dataset_ref()}'):
        os.makedirs(f'./results/{get_dataset_ref()}')
    with open(f"./results/{get_dataset_ref()}/automl_{framework}_{get_synthesizer_ref()}_thresh_{get_synthesizer_limit()}.json", "w") as outfile:
        json.dump(results, outfile)

    
    if complete_data is None:
        complete_data = pd.DataFrame()
    if synthetic_data is None:
        synthetic_data = pd.DataFrame()

    print(complete_data.head())
    print(complete_data.shape)
    print(synthetic_data.head())
    print(synthetic_data.shape)

    complete_data.to_csv(f"./results/{get_dataset_ref()}/automl_{framework}_{get_synthesizer_ref()}_thresh_{get_synthesizer_limit()}complete.csv", index=False)
    synthetic_data.to_csv(f"./results/{get_dataset_ref()}/automl_{framework}_{get_synthesizer_ref()}_thresh_{get_synthesizer_limit()}generated.csv", index=False)
