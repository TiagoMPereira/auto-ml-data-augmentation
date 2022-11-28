| framework   |   accuracy_score |   average_precision_score |   balanced_accuracy_score |   cohen_kappa_score |   f1_score_macro |   f1_score_micro |   f1_score_weighted |   matthews_corrcoef |   precision_score |   recall_score |   roc_auc_score | training_time   | test_time   |
|:------------|-----------------:|--------------------------:|--------------------------:|--------------------:|-----------------:|-----------------:|--------------------:|--------------------:|------------------:|---------------:|----------------:|:----------------|:------------|
| autogluon   |            0.889 |                     0.856 |                     0.886 |               0.775 |            0.887 |            0.889 |               0.889 |               0.776 |             0.884 |          0.917 |           0.886 | 00:00:14        | 00:00:00    |
| autokeras   |            0.675 |                     0.72  |                     0.701 |               0.38  |            0.661 |            0.675 |               0.654 |               0.463 |             0.95  |          0.429 |           0.701 | 00:00:57        | 00:00:00    |
| autopytorch |            0.823 |                     0.817 |                     0.83  |               0.649 |            0.823 |            0.823 |               0.823 |               0.658 |             0.902 |          0.759 |           0.83  | 00:10:52        | 00:00:03    |
| autosklearn |            1     |                     1     |                     1     |               1     |            1     |            1     |               1     |               1     |             1     |          1     |           1     | 00:47:52        | 00:00:02    |
| evalml      |            0.897 |                     0.877 |                     0.897 |               0.793 |            0.896 |            0.897 |               0.897 |               0.793 |             0.915 |          0.895 |           0.897 | 00:10:00        | 00:00:00    |
| flaml       |            0.918 |                     0.903 |                     0.919 |               0.834 |            0.917 |            0.918 |               0.918 |               0.835 |             0.938 |          0.91  |           0.919 | 00:10:00        | 00:00:00    |
| gama        |            1     |                     1     |                     1     |               1     |            1     |            1     |               1     |               1     |             1     |          1     |           1     | 00:09:00        | 00:00:00    |
| h2o         |            0.527 |                     0.54  |                     0.484 |              -0.034 |            0.374 |            0.527 |               0.403 |              -0.069 |             0.539 |          0.932 |           0.484 | 00:09:54        | 00:00:00    |
| lightautoml |            0.453 |                     0.547 |                     0.5   |               0     |            0.312 |            0.453 |               0.282 |               0     |             0     |          0     |           0.5   | 00:01:00        | 00:00:00    |
| tpot        |            0.951 |                     0.944 |                     0.952 |               0.901 |            0.95  |            0.951 |               0.951 |               0.901 |             0.969 |          0.94  |           0.952 | 00:11:26        | 00:00:00    |