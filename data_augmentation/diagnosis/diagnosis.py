import pandas as pd

from data_augmentation.utils.sampling_strategies import (all_strategy,
                                                   minority_strategy,
                                                   not_majority_strategy,
                                                   not_minority_strategy,
                                                   threshold_strategy)


def diagnostic(
    data: pd.DataFrame,
    target: str,
    sampling_strategy: str = "auto",
    sampling_strategy_thresh: int = None
):
    
    available_strategies = [
        "all", "auto", "minority", "not majority", "not minority", "threshold"]
    if sampling_strategy not in available_strategies:
        raise ValueError(f"The strategy {sampling_strategy} is not available."
                         "The available strategies are "
                         f"{', '.join(available_strategies)}")

    if not target in data.columns:
        raise ValueError(f"Target '{target}' is not in the dataset")
    
    dataset_length = len(data)
    classes_proportion = {k: float(v) for k, v
                          in data[target].value_counts(1).items()}
    classes_occurences = {k: int(v) for k, v
                          in data[target].value_counts(0).items()}
    dataset_memo = data.memory_usage().sum() / 1000 / 1000

    # Calculates the number of rows to generate
    if sampling_strategy == "all":
        rows_to_generate = all_strategy(classes_occurences)
    elif sampling_strategy in ["auto", "not majority"]:
        rows_to_generate = not_majority_strategy(classes_occurences)
    elif sampling_strategy == "minority":
        rows_to_generate = minority_strategy(classes_occurences)
    elif sampling_strategy == "not minority":
        rows_to_generate = not_minority_strategy(classes_occurences)
    elif sampling_strategy == "threshold":
        rows_to_generate = threshold_strategy(classes_occurences,
                                              sampling_strategy_thresh)

    target_type = data[target].dtypes.name

    diagnosis = {
        "target": str(target),
        "dataset_length": int(dataset_length),
        "memory_usage": float(dataset_memo),
        "classes_occurences": classes_occurences,
        "classes_proportions": classes_proportion,
        "rows_to_generate": rows_to_generate,
        "target_type": target_type
    }

    return diagnosis