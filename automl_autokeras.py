import autokeras as ak

from common import *

try:

    # X_train, X_test, y_train, y_test = load_data_delegate()
    X, y = load_data_delegate()


    TIMER.tic()    
    X_train, X_test, y_train, y_test, complete_data, synthetic_data = \
        generate_synthetic_dataset(X, y)
    synthesizer_time = TIMER.tocvalue()

    multi_label = False
    autokeras = ak.StructuredDataClassifier(multi_label=multi_label, max_trials=3, overwrite=True, seed=SEED)

    TIMER.tic()
    autokeras.fit(X_train, y_train, epochs=1000)
    training_time = TIMER.tocvalue()

    TIMER.tic()
    if multi_label:
        y_pred = autokeras.predict(X_test).astype(int)
    else:
        y_pred = autokeras.predict(X_test).astype(int).flatten()
    test_time = TIMER.tocvalue()

    collect_and_persist_results(
        y_test, y_pred, training_time, test_time, "autokeras",
        complete_data, synthetic_data, synthesizer_time)

except Exception as e:
    print(f'Cannot run autokeras for dataset {get_dataset_ref()}. Reason: {str(e)}')
