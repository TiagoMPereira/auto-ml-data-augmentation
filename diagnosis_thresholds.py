import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sdmetrics.column_pairs import ContingencySimilarity, CorrelationSimilarity
from sdmetrics.single_column import KSComplement, TVComplement

from data_augmentation.pipeline import synthesize_data


if __name__ == "__main__":
    SAVE_PATH = "results/class_thresholds/"
    DATASET_NAME = "synthetic_dataset2"

    dataset = pd.read_csv("openml_datasets/synthetic_dataset2.csv")

    synthesizers = ["ctgan", "fastml", "tvae", "copulagan", "gaussiancopula"]
    # synthesizers = ["fastml", "gaussiancopula"]
    thresholds = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

    continuous_cols = ["col_0", "col_1", "col_2", "col_3", "col_4", "col_5"]
    discrete_cols = ["col_6", "col_7", "col_8", "col_9"]

    classes = [1, 2, 3, 4, 5]
    original_class_values = dict(dataset["class"].value_counts())
    default_classes_generate = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    for class_ in classes:
        class_metrics = {}

        synt_metrics = {}
        for synt in synthesizers:
            synt_ks_mean = []
            synt_tv_mean = []
            synt_corr_mean = []
            synt_cont_mean = []
            synt_ks_median = []
            synt_tv_median = []
            synt_corr_median = []
            synt_cont_median = []


            for thresh in thresholds:
                print(str(class_)+"  "+synt+"  "+str(thresh))
                n_generate = default_classes_generate.copy()
                n_generate[class_] = int(original_class_values[class_] * thresh)
                diagnosis = {"rows_to_generate": n_generate}

                synthetic_data = synthesize_data(dataset, "class", synt, diagnosis)

                _cummulative_ks = []
                for col in continuous_cols:
                    try:
                        _ks = KSComplement.compute(
                            real_data=dataset.loc[dataset['class'] == class_, col],
                            synthetic_data=synthetic_data.loc[synthetic_data['class'] == class_, col]
                        )
                    except:
                        _ks = 0
                    
                    _cummulative_ks.append(_ks)
                synt_ks_mean.append(np.mean(_cummulative_ks))
                synt_ks_median.append(np.median(_cummulative_ks))

                _cummulative_tv = []
                for col in discrete_cols:
                    try:
                        _tv = TVComplement.compute(
                            real_data=dataset.loc[dataset['class'] == class_, col],
                            synthetic_data=synthetic_data.loc[synthetic_data['class'] == class_, col]
                        )
                    except:
                        _tv = 0
                    _cummulative_tv.append(_tv)
                synt_tv_mean.append(np.mean(_cummulative_tv))
                synt_tv_median.append(np.median(_cummulative_tv))

                current_correlation = []
                for i in range(len(continuous_cols)):
                    for j in range(i, len(continuous_cols)):
                        if i == j:
                            continue
                        try:
                            _current_correlation = CorrelationSimilarity.compute(
                                real_data=dataset.loc[dataset['class'] == class_, [continuous_cols[i], continuous_cols[j]]],
                                synthetic_data=synthetic_data.loc[synthetic_data['class'] == class_, [continuous_cols[i], continuous_cols[j]]],
                                coefficient='Pearson'
                            )
                        except:
                            _current_correlation = 0
                        current_correlation.append(_current_correlation)
                synt_corr_mean.append(np.mean(current_correlation))
                synt_corr_median.append(np.median(current_correlation))
                
                current_contingency = []
                for i in range(len(discrete_cols)):
                    for j in range(i, len(discrete_cols)):
                        if i == j:
                            continue
                        try:
                            _current_contingency = ContingencySimilarity.compute(
                                real_data=dataset.loc[dataset['class'] == class_, [discrete_cols[i], discrete_cols[j]]],
                                synthetic_data=synthetic_data.loc[synthetic_data['class'] == class_, [discrete_cols[i], discrete_cols[j]]],
                            )
                        except:
                            _current_contingency = 0
                        current_contingency.append(_current_contingency)
                synt_cont_mean.append(np.mean(current_contingency))
                synt_cont_median.append(np.median(current_contingency))
            synt_metrics[synt] = {
                "KS_mean": synt_ks_mean, "KS_median": synt_ks_median,
                "TV_mean": synt_tv_mean, "TV_median": synt_tv_median,
                "Correlation_mean": synt_corr_mean, "Correlation_median": synt_corr_median,
                "Contingency_mean": synt_cont_mean, "Contingency_median": synt_cont_median
            }
        class_metrics[class_] = synt_metrics.copy()

        with open(SAVE_PATH+DATASET_NAME+"_"+str(class_)+".json", "w") as fp:
            json.dump(class_metrics, fp)

