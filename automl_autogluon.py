import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

from common import *

try:
    # X_train, X_test, y_train, y_test = load_data_delegate()
    X, y = load_data_delegate()


    TIMER.tic()    
    X_train, X_test, y_train, y_test, complete_data, synthetic_data = \
        generate_synthetic_dataset(X, y)
    synthesizer_time = TIMER.tocvalue()



    train_df = pd.DataFrame(X_train).assign(**{'class': pd.Series(y_train)}).dropna()
    test_df = pd.DataFrame(X_test).assign(**{'class': pd.Series(y_test)}).dropna()

    clf = TabularPredictor(eval_metric='accuracy', label='class')

    TIMER.tic()
    clf = clf.fit(time_limit=EXEC_TIME_SECONDS, train_data=train_df)
    training_time = TIMER.tocvalue()

    TIMER.tic()
    y_test = test_df['class'].values
    y_pred = clf.predict(test_df)
    test_time = TIMER.tocvalue()

    collect_and_persist_results(
        y_test, y_pred, training_time, test_time, "autogluon",
        complete_data, synthetic_data, synthesizer_time)

except Exception as e:
    print(f'Cannot run autogluon for dataset {get_dataset_ref()}. Reason: {str(e)}')
