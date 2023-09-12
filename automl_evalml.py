from evalml.automl import AutoMLSearch

from common import *

try:

    # X_train, X_test, y_train, y_test = load_data_delegate()
    X, y = load_data_delegate()


    TIMER.tic()    
    X_train, X_test, y_train, y_test, complete_data, synthetic_data = \
        generate_synthetic_dataset(X, y)
    synthesizer_time = TIMER.tocvalue()

    clf = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type=infer_task_type(y_test), random_seed=SEED, max_time=EXEC_TIME_SECONDS)

    TIMER.tic()
    clf.search()
    best = clf.best_pipeline.fit(X_train, y_train)
    training_time = TIMER.tocvalue()

    TIMER.tic()
    y_pred = best.predict(X_test)
    test_time = TIMER.tocvalue()

    collect_and_persist_results(y_test, y_pred, training_time, test_time, "evalml",
                                complete_data, synthetic_data, synthesizer_time)

except Exception as e:
    print(f'Cannot run evalml for dataset {get_dataset_ref()}. Reason: {str(e)}')
