from autosklearn.classification import AutoSklearnClassifier

from common import *

try:

    # X_train, X_test, y_train, y_test = load_data_delegate()
    X, y = load_data_delegate()


    TIMER.tic()    
    X_train, X_test, y_train, y_test, complete_data, synthetic_data = \
        generate_synthetic_dataset(X, y)
    synthesizer_time = TIMER.tocvalue()

    if X_train is None or X_test is None or y_train is None or y_test is None:
        raise ValueError("Empty data")

    clf = AutoSklearnClassifier(time_left_for_this_task=EXEC_TIME_SECONDS, resampling_strategy="cv", resampling_strategy_arguments={"folds": 5}, seed=SEED)

    TIMER.tic()
    clf.fit(X_train, y_train)
    training_time = TIMER.tocvalue()

    TIMER.tic()
    y_pred = clf.predict(X_test)
    test_time = TIMER.tocvalue()

    collect_and_persist_results(y_test, y_pred, training_time, test_time, "autosklearn",
                                complete_data, synthetic_data, synthesizer_time)

except Exception as e:
    print(f'Cannot run autosklearn for dataset {get_dataset_ref()}. Reason: {str(e)}')
