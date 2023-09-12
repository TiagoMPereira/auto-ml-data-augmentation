import pandas as pd
from sklearn.datasets import make_classification
from scipy.stats import norm

SEED = 42
N_ROWS = 1599
N_COLS = 10
N_CLASSES = 6

if __name__ == "__main__":

    proportion_factor = 3200 / 63    # x + x/2 + x/4 + x/8 + x/16 + x/32 = 100

    proportions = {
        i: (proportion_factor / (2**i)) / 100 for i in range(N_CLASSES)
    }
    
    X, y = make_classification(
        n_samples=N_ROWS,
        n_features=N_COLS,
        n_classes=N_CLASSES,
        weights=proportions,
        n_informative=3,
        n_clusters_per_class=1,
        random_state=SEED
    )

    dataset = pd.DataFrame(X)
    dataset.columns = [f"col_{c}" for c in range(N_COLS)]
    dataset["class"] = y

    new_dataset = pd.DataFrame()
    # 3 colunas gaussianas
    new_dataset["col_0"] = dataset["col_0"]
    new_dataset["col_1"] = dataset["col_1"]
    new_dataset["col_2"] = dataset["col_2"]

    # 3 distribuições uniformes
    new_dataset["col_3"] = norm.cdf(dataset["col_3"], dataset['col_3'].mean(), dataset['col_3'].std())
    new_dataset["col_4"] = norm.cdf(dataset["col_4"], dataset['col_4'].mean(), dataset['col_4'].std())
    new_dataset["col_5"] = norm.cdf(dataset["col_5"], dataset['col_5'].mean(), dataset['col_5'].std())

    # 4 colunas com valores categóricos numéricos
    new_dataset["col_6"] = dataset["col_6"].astype("int")
    new_dataset["col_7"] = dataset["col_7"].astype("int")
    new_dataset["col_8"] = dataset["col_8"].astype("int")
    new_dataset["col_9"] = dataset["col_9"].astype("int")

    new_dataset['class'] = dataset['class']

    new_dataset.to_csv("openml_datasets/synthetic_dataset2.csv", index=False)
