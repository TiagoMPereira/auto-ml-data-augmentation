from autoPyTorch.api.tabular_classification import TabularClassificationTask

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

    clf = TabularClassificationTask(seed=SEED)

    TIMER.tic()
    clf.search(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        optimize_metric='accuracy',
        budget_type='runtime',
        total_walltime_limit=EXEC_TIME_SECONDS,
        func_eval_time_limit_secs=EXEC_TIME_SECONDS/10,
        memory_limit=8192
    )
    training_time = TIMER.tocvalue()

    TIMER.tic()
    y_pred = clf.predict(X_test)
    test_time = TIMER.tocvalue()

    collect_and_persist_results(y_test, y_pred, training_time, test_time, "autopytorch",
                                complete_data, synthetic_data, synthesizer_time)

except Exception as e:
    print(f'Cannot run autopytorch for dataset {get_dataset_ref()}. Reason: {str(e)}')
