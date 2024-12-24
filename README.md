# Machine Learning Model Performance and Insights

This project aimed to use multiple supervised and unsupervised machine learning algorithms to find the best-performing model for this dataset. This involved thorough EDA with the help of box-plot graphs, removing null values, correlation matrices, univariate-multivariate analysis, and target analysis. Dummy encoding was performed for the categorical variables, and label encoding was applied to the target variable.

This process allowed me to build a simpler model, reducing the feature count from 23 to 12 (including dummy encoding).

---

## Supervised Machine Learning Models

### Summary of Model Performance

| Model                      | Accuracy | F1   | Precision | Recall | AUC for ROC Curve |
|----------------------------|----------|------|-----------|--------|-------------------|
| Decision Tree              | 91.67%   | 0.90 | 0.91      | 0.90   | 0.98              |
| Random Forest              | 90.75%   | 0.89 | 0.91      | 0.88   | 0.98              |
| Bagging                    | 90.81%   | 0.89 | 0.91      | 0.88   | 0.97              |
| Boosting (AdaBoost)        | 90.80%   | 0.88 | 0.91      | 0.86   | 0.97              |
| Gaussian Naive Bayes (GNB) | 81.08%   | 0.78 | 0.79      | 0.76   | 0.88              |
| Linear Discriminant (LDA)  | 82.72%   | 0.80 | 0.80      | 0.79   | 0.89              |
| Quadratic Discriminant (QDA) | 82.85% | 0.80 | 0.80      | 0.79   | 0.90              |
| K-Nearest Neighbours (KNN) | 90.75%   | 0.89 | 0.91      | 0.88   | 0.96              |

The Decision Tree model, after careful tuning, achieved the highest accuracy (91.67%), outperforming all other models. It also maintained strong F1, precision, and recall scores, indicating that it successfully identifies satisfied customers with minimal misclassifications.

The ensemble methods (Random Forest, Bagging, Boosting) and the KNN model offered robust alternatives, but none matched the Decision Tree’s top accuracy. While they demonstrate resilience and a balanced trade-off between precision and recall, their slight drop in accuracy could mean missing opportunities to correctly classify customers—an essential factor for strategic business decisions.

---

## Unsupervised Machine Learning Models

### K-Means Clustering

- **Optimal Number of Clusters (k):** 4  
- The K-Means model was chosen to cluster features and derive potential business insights.  
- Detailed business insights from clustering analysis are provided in the report.
