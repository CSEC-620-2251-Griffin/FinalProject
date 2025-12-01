# VPN Detector Evaluation

## Thresholds
- rf: 0.05
- xgb: 0.14183673469387753
- logreg: 0.8765306122448979

## RF
- accuracy: 0.9974
- precision: 0.8673
- recall: 1.0000
- f1: 0.9290
- f1_macro: 0.9638
- f1_weighted: 0.9975
- roc_auc: 1.0000
- pr_auc: 1.0000
- recall_at_fpr_1pct: 1.0000

## XGB
- accuracy: 0.9997
- precision: 0.9827
- recall: 1.0000
- f1: 0.9913
- f1_macro: 0.9956
- f1_weighted: 0.9997
- roc_auc: 1.0000
- pr_auc: 1.0000
- recall_at_fpr_1pct: 1.0000

## LOGREG
- accuracy: 0.9769
- precision: 0.4198
- recall: 1.0000
- f1: 0.5913
- f1_macro: 0.7897
- f1_weighted: 0.9815
- roc_auc: 0.9911
- pr_auc: 0.4631
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
           1       0.75      0.90      0.82        20

    accuracy                           1.00      3684
   macro avg       0.87      0.95      0.91      3684
weighted avg       1.00      1.00      1.00      3684

