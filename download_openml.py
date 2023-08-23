import pandas as pd
from sklearn.datasets import fetch_openml

ids = [37, 44, 1462, 1479, 1510, 23, 181, 1466, 40691, 40975]

for id_ in ids:
    print(id_)
    dataset = fetch_openml(data_id=id_, return_X_y=False)
    target_name = dataset.target_names[0]

    X, y = dataset.data, dataset.target
    y = pd.Series(pd.factorize(y)[0])

    complete_data = X.copy()
    complete_data[target_name] = y
    complete_data.to_csv(f"./openml_datasets/{id_}.csv", index=False)
