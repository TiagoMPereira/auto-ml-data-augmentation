import pandas as pd
import json

ids = [37, 44, 1462, 1479, 1510, 23, 181, 1466, 40691, 40975]

total_infos = []
for id_ in ids:
    print(id_)
    dataset = pd.read_csv(f"./openml_datasets/{id_}.csv")
    target = dataset.columns[-1]
    infos = {
        "id": id_,
        "dataset_shape": dataset.shape,
        "target": target,
        "value_counts": {k: int(v) for k, v in dict(dataset[target].value_counts(0)).items()},
        "proportions": {k: float(v) for k, v in dict(dataset[target].value_counts(1)).items()},
    }
    total_infos.append(infos)

print(total_infos)
with open(f"./openml_datasets/descriptions.json", "w") as fp:
    json.dump(total_infos, fp)