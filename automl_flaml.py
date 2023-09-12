from flaml import AutoML

from common import *

try:

    # X_train, X_test, y_train, y_test = load_data_delegate()
    X, y = load_data_delegate()


    TIMER.tic()    
    X_train, X_test, y_train, y_test, complete_data, synthetic_data = \
        generate_synthetic_dataset(X, y)
    synthesizer_time = TIMER.tocvalue()
    clf = AutoML()

    TIMER.tic()
    clf.fit(X_train, y_train, metric="accuracy", task="classification", time_budget=EXEC_TIME_SECONDS)
    training_time = TIMER.tocvalue()

    TIMER.tic()
    y_pred = clf.predict(X_test)
    test_time = TIMER.tocvalue()

    collect_and_persist_results(y_test, y_pred, training_time, test_time, "flaml",
                                complete_data, synthetic_data, synthesizer_time)

except Exception as e:
    print(f'Cannot run flaml for dataset {get_dataset_ref()}. Reason: {str(e)}')
