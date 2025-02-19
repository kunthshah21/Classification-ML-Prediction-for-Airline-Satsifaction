# Airline Satisfaction – Classification

---

## 1. Problem Statement
The airline aims to identify factors influencing client satisfaction to predict customer sentiment accurately. By understanding which attributes most strongly affect satisfaction, the airline can tailor personalized offers, enhance customer retention, address specific needs, and foster loyalty.

---

## 2. Proposed Solution
1. **Supervised Learning (Classification)**  
   - Predict customer satisfaction as either “satisfied” or “dissatisfied.”  
   - Identify the best-performing model (and its tuned hyperparameters) for classification.

2. **Unsupervised Learning (Clustering)**  
   - Segment customers into four clusters based on key attributes.  
   - Gain deeper insights into customer profiles and preferences for targeted strategies.

---

## 3. Exploratory Data Analysis

### 3.1 Understanding the Data
- **Source:** [Kaggle Airline Passenger Satisfaction Dataset](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)  
- **Size:** ~26,000 instances  
- **Features:** 23 total (mixed numeric and categorical variables)  
- **Target Variable:** Satisfaction label (satisfied vs. dissatisfied/neutral)

### 3.2 Data Cleaning
- **Handling Missing Values:**  
  - Found 83 null values in the dataset (<0.1% of total).  
  - Rows containing null values were dropped due to their minimal proportion.

### 3.3 Data Exploration

#### 3.3.1 Target Analysis
- The dataset is reasonably balanced:  
  - ~12k “satisfied” passengers  
  - ~14k “dissatisfied/neutral” passengers  
- No advanced balancing (e.g., SMOTE) was required.

#### 3.3.2 Univariate Analysis
- **Age:** Approximately normal distribution, mostly between 25 and 60 years old.  
- **Flight Distance:** Skewed to the right; most passengers flew shorter distances (<1000 km).  
- **Departure Delays:** Over 99% had zero delay.  
- **Service Ratings:** Generally between 3 and 4, with some variation depending on the feature (inflight service, seat comfort, etc.).

#### 3.3.3 Multivariate Analysis
- Identified some pairs of highly correlated features (>40% correlation).  
- Used feature selection by examining box plots and scatter plots to drop those less predictive or highly redundant.  
- Resulted in a reduced, more relevant set of predictors.

**Figure 1**: Correlation Matrix (before and after feature selection and reduction)   
<img width="686" alt="Screenshot 2025-02-19 at 6 32 59 PM" src="https://github.com/user-attachments/assets/58a8500c-11c7-4cc6-9982-a1a0b6e7b5e0" />


#### 3.3.4 Feature Encoding
- **Dummy Encoding:**  
  - Categorical features were dummy-encoded.  
  - Dropped the first column of each category to avoid the dummy variable trap.  
- **Label Encoding:**  
  - Target variable was label-encoded (satisfied vs. dissatisfied).

---

## 4. Supervised Models Evaluation and Interpretation

### 4.1 Building Models
1. **Data Scaling:** All features scaled to a standard range.  
2. **Train-Test Split:** 80:20 ratio.  
3. **Cross-Validation & Grid Search:**  
   - 5-fold cross-validation within the training set.  
   - Grid search to find optimal hyperparameters.  
4. **Final Model Creation:** Models were refit on the entire training set using best hyperparameters and then evaluated on the test set.

### 4.2 Comparative Analysis and Model Selection

#### 4.2.1 Comparison Analysis of All Models

| Model                   | Accuracy |  F1  | Precision | Recall |  AUC  |
|-------------------------|---------:|-----:|----------:|-------:|------:|
| **Decision Tree**       |   91.67% | 0.90 |      0.91 |   0.90 |  0.98 |
| **Random Forest**       |   90.75% | 0.89 |      0.91 |   0.88 |  0.98 |
| **Bagging**             |   90.81% | 0.89 |      0.91 |   0.88 |  0.97 |
| **AdaBoost (Boosting)** |    90.8% | 0.88 |      0.91 |   0.86 |  0.97 |
| **Gaussian Naive Bayes**|   81.08% | 0.78 |      0.79 |   0.76 |  0.88 |
| **LDA**                 |   82.72% | 0.80 |      0.80 |   0.79 |  0.89 |
| **QDA**                 |   82.85% | 0.80 |      0.80 |   0.79 |  0.90 |
| **KNN**                 |   90.75% | 0.89 |      0.91 |   0.88 |  0.96 |

**Key Takeaways:**
- **Decision Tree** emerged as the top performer with the highest accuracy (91.67%) and robust F1, precision, and recall.  
- Ensemble methods (Random Forest, Bagging, AdaBoost) and KNN showed competitive performance but slightly lower accuracy.  
- Gaussian Naive Bayes and QDA had lower accuracy (~81–83%), making them less ideal for final selection.

#### 4.2.2 Choosing the Best Model
- **Model Selected:** Decision Tree  
- **Rationale:**  
  - Highest accuracy (91.67%).  
  - Balanced F1, precision, and recall.  
  - Strong AUC of 0.98, indicating excellent class separation.  
- **Business Impact:**  
  - Accurate classification helps target dissatisfied customers for interventions and reward satisfied customers with loyalty perks.  
  - Interpretable decision tree structure aids business stakeholders in understanding key drivers of satisfaction.

---

### 4.3 Unsupervised Model with K-Means Clustering

#### 4.3.1 Model Architecture
- **K-Means:** Partitions data into *k* clusters by minimizing the Within-Cluster Sum of Squares (WCSS).  
- **Initialization:** Random centroids.  
- **Iteration:** Data points assigned to nearest centroid; centroids updated until convergence.

#### 4.3.2 Finding Optimal Value for K
- Tested values of *k* from 1 to 10.  
- Used **Elbow Method** (WCSS) and **Silhouette Scores** to identify the best separation.  
- **Optimal K:** 4 clusters.

**Figure 2**: Elbow Diagram and Silhouette Scores for K-Means  
<img width="614" alt="Screenshot 2025-02-19 at 6 33 36 PM" src="https://github.com/user-attachments/assets/b3750e54-63de-4744-9744-72e1af45ec44" />


---

## 5. Business Application

### 5.1 Insights from Decision Tree Analysis on Customer Satisfaction
1. **Personal Travel Customers Often More Satisfied**  
   - **Implication:** Provide special promotions or personalized services for these travelers.

2. **Disloyal (Multi-Airline) Flyers Show High Satisfaction**  
   - **Implication:** Expand loyalty programs and highlight airline advantages to convert them into repeat customers.

3. **Check-in Service Is Critical**  
   - **Implication:** Streamline check-in via digital kiosks or better staffing to reduce wait times and frustrations.

4. **Younger Customers More Satisfied**  
   - **Implication:** Offer modern conveniences (Wi-Fi, eco-friendly initiatives) to maintain loyalty.

5. **Legroom Matters**  
   - **Implication:** Consider rearranging cabin layouts or upselling seats with extra legroom.

6. **Inflight Entertainment & Baggage Handling Less Influential**  
   - **Implication:** Keep these at acceptable levels but invest resources primarily into high-impact areas like check-in or seating comfort.

7. **Long-Distance Travelers Tend to Be Less Satisfied**  
   - **Implication:** Improve comfort (seating, meals, in-flight amenities) for long-haul flights to boost overall satisfaction.

### 5.2 Cluster Descriptions (K-Means)

1. **Cluster 0: Premium Frequent Flyers**  
   - **Profile:** Middle-aged (avg. 44.1), frequent long-distance flyers (~1888 km)  
   - **High Satisfaction** with entertainment, legroom, baggage handling, inflight service  
   - Likely business travelers; minimal arrival delays

2. **Cluster 1: Occasional Personal Travelers**  
   - **Profile:** Younger (avg. 37.1), shorter flight distances (~758 km), mainly economy class  
   - Predominantly personal travel (90% personal vs. 10% business)  
   - Moderate service ratings (~3.2) but slightly higher delays (~14.76 mins)

3. **Cluster 2: Disloyal Economy Travelers**  
   - **Profile:** Younger (avg. 29.9), 100% disloyal, mostly economy (57%)  
   - Short flights (~706 km) with average service ratings (3.1–3.7)  
   - Slightly higher delays (~15 mins)

4. **Cluster 3: Dissatisfied Middle-Aged Flyers**  
   - **Profile:** Middle-aged (avg. 44.1), lowest satisfaction scores across service categories (2.38–2.72)  
   - Medium-length flights (~1173 km) but highest arrival delays (~17.73 mins)  
   - Mixed purpose (personal/business) with overall dissatisfaction stemming from poor service experiences

### 5.3 Model Limitations
- **Similar Data Assumption:** The model performs best if new data follows a similar distribution to the training set (e.g., mostly shorter flights).  
- **Outliers/Edge Cases:** Not specifically tested on outlier-heavy or drastically different datasets.  
- **Continual Updates:** Periodic retraining is recommended to handle evolving customer profiles and airline operations.

