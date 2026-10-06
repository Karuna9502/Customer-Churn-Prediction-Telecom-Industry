# Customer Churn Analysis & Prediction | Telecom Industry

> End-to-end analysis of 7,043 telecom customers: data cleaning, statistical testing, churn prediction, and a 3-page Power BI dashboard that gives a retention team a prioritised call list.

![Python](https://img.shields.io/badge/Python-pandas%20%7C%20scikit--learn%20%7C%20scipy-blue)
![Power BI](https://img.shields.io/badge/Power%20BI-DAX%20%7C%20Power%20Query-yellow)
![Status](https://img.shields.io/badge/Status-Complete-green)

---

## 1. Business Problem

A telecom company is losing **26.5% of its customers** (1,869 of 7,043). Those customers represent **$139,131 in monthly revenue, which is 30.5% of total monthly revenue (about $1.67M per year)**.

The company does not know:
1. **Why** customers leave
2. **Which** current customers are most likely to leave next
3. **Whom** the retention team should contact first

Acquiring a new customer costs far more than retaining an existing one, so spending the retention budget on the right customers matters.

## 2. Objective

1. Identify the main drivers of churn using statistical tests
2. Build a model that predicts churn risk for every customer
3. Deliver a Power BI dashboard with a prioritised call list for the retention team

## 3. Dataset

| Item | Detail |
|---|---|
| Source | [Telco Customer Churn (Kaggle, IBM sample data)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Size | 7,043 customers, 21 columns |
| Features | 3 numeric (tenure, MonthlyCharges, TotalCharges), 16 categorical, 1 target (Churn), 1 ID |
| Target balance | 73.5% retained, 26.5% churned (imbalanced) |
| Type | Cross-sectional snapshot (no time dimension) |

## 4. Key Findings

| # | Finding | Evidence |
|---|---|---|
| 1 | **Contract type is the strongest churn driver** | Month-to-month 42.7% churn vs two-year 2.8% (Cramér's V = 0.41) |
| 2 | **New customers leave the most** | Month-to-month customers in their first 12 months churn at **51.4%** |
| 3 | **Fiber optic customers churn far more than DSL** | 41.9% vs 19.0% (logistic regression odds ratio 3.06) |
| 4 | **Electronic check users churn the most** | 45.3% vs 15 to 19% for other payment methods |
| 5 | **Support services protect against churn** | No TechSupport 31.2% vs 15.2% with TechSupport |
| 6 | **Churners are newer and pay more** | Tenure 18.0 vs 37.6 months, monthly bill $74.4 vs $61.3 (Welch t-test, p < 0.001) |
| 7 | **Gender and phone service have no effect** | p = 0.49 and p = 0.34 (chi-square) |

### Statistical tests (chi-square, top drivers)

| Feature | Cramér's V | Strength |
|---|---|---|
| Contract | 0.410 | Strong |
| Tenure group | 0.349 | Moderate |
| Internet service | 0.322 | Moderate |
| Payment method | 0.303 | Moderate |
| Paperless billing | 0.191 | Weak |
| Online security | 0.171 | Weak |
| Tech support | 0.164 | Weak |

15 of 17 categorical features are significantly related to churn (p < 0.05).

### Churn rate by contract and tenure

| Contract | 0-12m | 13-24m | 25-48m | 49-72m |
|---|---|---|---|---|
| Month-to-month | **51.4%** | 37.7% | 32.9% | 26.0% |
| One year | 10.5% | 8.1% | 10.6% | 12.9% |
| Two year | 0.0% | 0.0% | 2.2% | 3.3% |

Even long-standing month-to-month customers (49-72 months) churn at 26.0%, about 8 times the rate of two-year customers of the same tenure.

## 5. Methodology

```
Load -> Know -> Find faults -> Clean -> EDA & stats -> Preprocess -> Model -> Evaluate -> Dashboard
```

### Data quality (faults found and fixed)

| Fault | Fix | Reason |
|---|---|---|
| `TotalCharges` stored as text, 11 hidden blanks (spaces) | Converted to float, blanks set to 0 | All 11 rows had tenure = 0 (new customers, no bill yet) and none churned |
| `SeniorCitizen` as 0/1 | Mapped to No/Yes | Consistent with other columns |
| "No internet service" / "No phone service" in 7 columns | Merged into "No" | Redundant, already captured by `InternetService` and `PhoneService` |
| `Churn` as text | Added `Churn_flag` (1/0) | Needed for modelling |
| Duplicates, other nulls, outliers | None found | Checked with IQR, retained as genuine behaviour |

### Preprocessing

- Dropped `customerID`, `Churn` (target leakage), `tenure_group` (duplicate of tenure) and `TotalCharges` (VIF = 8.08, highly correlated with tenure, r = 0.83)
- One-hot encoding with `drop_first=True` (avoids the dummy variable trap), 22 model features
- Stratified 80/20 split (5,634 train, 1,409 test, 26.54% churn in both)
- `StandardScaler` fitted on the training set only, to prevent data leakage
- Class imbalance handled with `class_weight='balanced'`

### Modelling

| Model | Purpose |
|---|---|
| Dummy baseline (always "No") | Reference point to beat |
| Logistic Regression | Interpretable drivers through odds ratios |
| Random Forest (300 trees, max_depth 8, min_samples_leaf 20) | Churn risk scores |

## 6. Model Results (test set, 1,409 customers)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (always "No") | 0.735 | 0.000 | 0.000 | 0.000 | n/a |
| Logistic Regression | 0.741 | 0.508 | 0.775 | 0.614 | 0.839 |
| **Random Forest** | **0.752** | **0.521** | **0.781** | **0.625** | **0.844** |

5-fold cross-validation on the training set confirms stability:

| Model | Recall | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.793 ± 0.033 | 0.845 |
| Random Forest | 0.777 ± 0.023 | 0.846 |

**Why recall, not accuracy?** The baseline scores 73.5% accuracy while catching zero churners. A missed churner means lost revenue, while a false alarm only costs a retention offer. Random Forest caught **292 of 374 churners (78.1%)** and exceeded the 75% recall target.

**Confusion matrix (Random Forest):** 767 true negatives, 268 false positives, 82 false negatives, 292 true positives.

### Why customers churn: three methods agree

| Driver | Chi-square (Cramér's V) | Logistic regression odds ratio | Random Forest importance rank |
|---|---|---|---|
| Contract | 0.410 | Two year 0.245, One year 0.486 | 2nd, 4th |
| Tenure | 0.349 | 0.468 per +1 SD (about 25 months) | 1st |
| Fiber optic internet | 0.322 | 3.06 | 3rd |
| Electronic check | 0.303 | 1.49 | n/a |

A statistical test, a linear model and a tree model identify the same top drivers, which makes the findings robust.

> **Note on MonthlyCharges:** churners pay more in raw data, but the odds ratio is 0.77 after controlling for services. The higher bill is driven by Fiber and streaming services that are already in the model. This is confounding, not a contradiction.

### Threshold analysis (Random Forest)

| Threshold | Flagged | Churners caught | Recall | Precision |
|---|---|---|---|---|
| 0.3 | 836 | 350 | 0.936 | 0.419 |
| 0.4 | 697 | 325 | 0.869 | 0.466 |
| **0.5 (chosen)** | **560** | **292** | **0.781** | **0.521** |
| 0.6 | 439 | 251 | 0.671 | 0.572 |
| 0.7 | 279 | 185 | 0.495 | 0.663 |

The threshold is a business lever. With a larger retention budget, lowering it to 0.4 catches 33 more churners for 137 extra offers.

## 7. Business Impact

Churn probabilities were generated for all 7,043 customers using **out-of-fold predictions**, so no customer is scored by a model that trained on them.

| Risk band | Customers | Actual churn rate |
|---|---|---|
| Low (< 0.4) | 3,519 | 6.4% |
| Medium (0.4 to 0.7) | 2,113 | 33.4% |
| High (> 0.7) | 1,411 | 66.4% |

High-risk customers are about **10 times** more likely to churn than low-risk ones.

### Actionable output

- **474 current (active) customers are high risk**
- They represent **$37,238 in monthly revenue (about $447K per year)**
- **All 474 are on month-to-month contracts**, mostly Fiber optic with electronic check and very short tenure
- Average churn probability of this group: 77.9%

## 8. Recommendations

1. **Convert month-to-month customers to annual contracts**, starting with first-year customers, using a discounted first-year offer
2. **Investigate Fiber optic pricing and service quality**, since churn is more than double that of DSL despite the premium price
3. **Incentivise auto-pay** (bank transfer or credit card) over electronic check
4. **Bundle TechSupport and OnlineSecurity free** for the first few months for new customers
5. **Prioritise the 474-customer call list**, starting from the highest churn probability

## 9. Power BI Dashboard

| Page | Audience | Purpose |
|---|---|---|
| **Executive Overview** | Management | Size of the problem: 7,043 customers, 26.5% churn, $139K monthly revenue lost, with contract slicer |
| **Churn Drivers** | Analysts and marketing | Churn by internet service, payment method, tech support, and contract × tenure matrix |
| **Retention Action List** | Retention team | 474 high-risk active customers sorted by churn probability, filterable by contract, internet service and payment method |

**Executive Overview**

![Executive Overview](images/page1_overview.png)

**Churn Drivers**

![Churn Drivers](images/page2_drivers.png)

**Retention Action List**

![Retention Action List](images/page3_retention.png)

DAX measures used: `Total Customers`, `Churned Customers`, `Churn Rate %`, `Revenue Lost`, `Revenue at Risk`, `High Risk Active`, `Avg Churn Probability`. Dashboard numbers were cross-verified against the Python results.

## 10. Python Visuals

| | |
|---|---|
| ![Churn by features](images/churn_by_features.png) | ![Numeric vs churn](images/numeric_vs_churn.png) |
| ![Contract x tenure](images/contract_tenure_heatmap.png) | ![Feature importance](images/feature_importance.png) |
| ![Confusion matrix](images/confusion_matrix.png) | ![ROC curve](images/roc_curve.png) |

## 11. Limitations

- **Snapshot data with no time dimension**, so the model ranks risk but cannot predict churn "within the next 30 days"
- **No behavioural data** (usage, complaints, support calls, competitor offers), which would likely improve accuracy
- **Correlation is not causation:** the real ROI of a retention offer needs an A/B test
- Precision is about 52%, so roughly half of flagged customers would not have left; offers should be low-cost

## 12. Tools

- **Python:** pandas, numpy, scipy, statsmodels, scikit-learn, matplotlib, seaborn
- **Power BI:** Power Query, DAX, report design
- **Statistics:** chi-square, Cramér's V, Welch t-test, VIF, odds ratios, cross-validation
- **Git / GitHub**

## 13. Repository Structure

```
customer-churn-project/
├── README.md
├── data/
│   ├── raw/          WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/    telco_clean.csv, churn_dashboard_data.csv
├── notebooks/        Telco-Customer-Churn.ipynb
├── dashboard/        Customer_Churn_Dashboard.pbix
└── images/           charts and dashboard screenshots
```

## 14. How to Reproduce

1. Clone the repository
2. Install: `pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn jupyter`
3. Open `notebooks/Telco-Customer-Churn.ipynb` and run all cells (the raw CSV path may need adjusting)
4. Open `dashboard/Customer_Churn_Dashboard.pbix` in Power BI Desktop and point the data source to `data/processed/churn_dashboard_data.csv`

## 15. Author

**Karuna Kumari**
Aspiring Data / Business Analyst
[LinkedIn](ADD-YOUR-LINKEDIN-URL) · [Email](mailto:ADD-YOUR-EMAIL) · [GitHub](https://github.com/Karuna9502)
