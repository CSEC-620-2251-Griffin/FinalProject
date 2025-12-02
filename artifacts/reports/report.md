# VPN Detector Evaluation

## Thresholds
- rf: 0.05
- xgb: 0.43571428571428567
- logreg: 0.95

## RF
- accuracy: 0.9919
- precision: 0.6719
- recall: 1.0000
- f1: 0.8038
- f1_macro: 0.8998
- f1_weighted: 0.9926
- roc_auc: 1.0000
- pr_auc: 0.9990
- recall_at_fpr_1pct: 1.0000

## XGB
- accuracy: 0.9991
- precision: 0.9548
- recall: 0.9941
- f1: 0.9741
- f1_macro: 0.9868
- f1_weighted: 0.9991
- roc_auc: 1.0000
- pr_auc: 0.9995
- recall_at_fpr_1pct: 1.0000

## LOGREG
- accuracy: 0.9759
- precision: 0.3000
- recall: 0.3353
- f1: 0.3167
- f1_macro: 0.6522
- f1_weighted: 0.9765
- roc_auc: 0.9875
- pr_auc: 0.4135
- recall_at_fpr_1pct: 0.3118

### Classification report (val for threshold reference)
**RandomForest**

              precision    recall  f1-score   support

           0       1.00      1.00      1.00      3664
           1       1.00      1.00      1.00        20

    accuracy                           1.00      3684
   macro avg       1.00      1.00      1.00      3684
weighted avg       1.00      1.00      1.00      3684

**XGBoost**

              precision    recall  f1-score   support

           0       1.00      1.00      1.00      3664
           1       1.00      1.00      1.00        20

    accuracy                           1.00      3684
   macro avg       1.00      1.00      1.00      3684
weighted avg       1.00      1.00      1.00      3684

**LogReg**

              precision    recall  f1-score   support

           0       1.00      1.00      1.00      3664
           1       0.74      0.85      0.79        20

    accuracy                           1.00      3684
   macro avg       0.87      0.92      0.89      3684
weighted avg       1.00      1.00      1.00      3684

