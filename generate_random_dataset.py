import pandas as pd
from sklearn.datasets import make_classification

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
    # 2 colunas originais
    new_dataset["col_0"] = dataset["col_0"]
    new_dataset["col_1"] = dataset["col_1"]

    # 1 coluna apenas com valores positivos
    new_dataset["col_2"] = dataset["col_2"] + abs(min(dataset["col_2"]))

    # 1 coluna apenas com valores negativos
    new_dataset["col_3"] = dataset["col_3"] - max(dataset["col_3"])

    # 2 colunas com valores escalados * 100
    new_dataset["col_4"] = dataset["col_4"] * 100
    new_dataset["col_5"] = dataset["col_5"] * 100

    # 2 colunas com valores categóricos numéricos
    new_dataset["col_6"] = dataset["col_6"].astype("int")
    new_dataset["col_7"] = dataset["col_7"].astype("int")

    # 2 colunas com valores categóricos literais
    replace_8 = {-3: "earth", -2: "water", -1: "fire", 0: "air"}
    new_dataset["col_8"] = dataset["col_8"].astype("int").replace(replace_8)
    replace_9 = {-4: "sun", -3: "mon", -2: "tue", -1: "wed", 0: "thu", 1: "fri", 2: "sat"}
    new_dataset["col_9"] = dataset["col_9"].astype("int").replace(replace_9)

    new_dataset['class'] = dataset['class']

    new_dataset.to_csv("openml_datasets/synthetic_dataset.csv", index=False)
