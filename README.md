🏦 Predicting Corporate Bankruptcy Before It Happens
An End-to-End Machine Learning Pipeline for Financial Distress Detection
📌 Project Summary
Every year, thousands of companies collapse — destroying jobs, wiping out investor wealth, and leaving banks with unpaid loans. The tragedy is that most bankruptcies are predictable, if you know which financial signals to look for.
This project builds a machine learning pipeline that predicts whether a company will go bankrupt up to 3 years in advance, using 64 financial ratios extracted from real company balance sheets. The model is benchmarked against the Altman Z-Score — the formula used by JPMorgan, Goldman Sachs, and credit rating agencies since 1968.
Result: My XGBoost model achieves 0.98 AUC — beating the Altman Z-Score by 47%.

🎯 Objective

Detect early warning signs of corporate financial distress from publicly available financial ratios
Build a model that outperforms the Altman Z-Score — the 56-year-old industry benchmark
Explain why each prediction is made using SHAP — because banks and regulators require explainability, not just accuracy
Deliver findings in a format that non-technical stakeholders (CFOs, credit analysts, investors) can act on


📊 Dataset
PropertyDetailSourceUCI Machine Learning Repository (ID: 365)Companies43,405 real Polish company recordsTime Period2000 – 2012Features64 financial ratios (A1–A64)Target1 = Bankrupt, 0 = SurvivedBankrupt2,091 companies (4.8%)Healthy41,314 companies (95.2%)
The dataset contains 5 years of observations per company, allowing prediction windows of 1–5 years before bankruptcy. We focus on the 3-year prediction window — hard enough to be meaningful, achievable enough to be useful.

🔍 Key Insights & Analysis
1. The Class Imbalance Problem
95.2% of companies are healthy. A naive model that always predicts "healthy" would be 95% accurate — but completely useless. This is why accuracy is the wrong metric for this problem. I used AUC-ROC instead, which measures how well the model separates the two classes regardless of imbalance.
2. Missing Data Patterns

A37 had 43.7% missing values — too unreliable to use, dropped entirely
A21 had 13.5% missing — filled using group median (separately for bankrupt and healthy companies)
Filling missing values by group (not overall median) is critical — bankrupt and healthy companies have very different financial profiles

3. Top Bankruptcy Predictors (SHAP Analysis)
After training, SHAP explainability revealed the most important signals:
RankRatioWhat It Likely MeasuresSignal1A27Debt structure / leverageLow values = high bankruptcy risk2A34Profitability ratioDeclining values precede collapse3A21Liquidity measureCash flow deterioration4A46Asset efficiencyPoor asset utilization = warning sign5A24Coverage ratioInability to service debt
Key finding: A27 is the single strongest predictor — companies with low A27 values are at significantly higher risk of bankruptcy up to 3 years before collapse.
4. Outlier Impact
Removing extreme outliers (IQR method, 10x multiplier) reduced the dataset from 43,405 to 26,737 rows — but dramatically improved model stability. Real financial data contains extreme values from data entry errors and genuinely unusual companies.
5. What A5 Tells Us
Ratio A5 turns negative in bankrupt companies (median: -36.6) while healthy companies maintain positive values (median: 0.648). A negative A5 likely indicates negative profitability — the company is destroying value, not creating it.

📈 Model Results
ModelAUC ScoreNotesAltman Z-Score0.6679Industry standard since 1968Logistic Regression0.7276Simple baselineXGBoost (this project)0.9820Best model
Why XGBoost beats Altman Z-Score by 47%:
The Altman Z-Score uses only 5 hand-picked ratios combined in a linear formula designed in 1968. XGBoost learns from all 64 ratios simultaneously, captures non-linear relationships, and adapts to the data rather than relying on fixed coefficients. This is not a fair fight — but it proves that modern ML dramatically outperforms legacy methods on this problem.

🗂️ Project Structure
bankruptcy_prediction/
│
├── step1_collect_data.py      # Download dataset from UCI repository
├── step2_explore.py           # Shape, missing values, class balance
├── step3_visualize.py         # First look charts — distributions and balance
├── step4_key_ratios.py        # Find ratios that separate healthy vs bankrupt
├── step5_cleaning.py          # Handle missing values, outliers, save clean data
├── step6_model.py             # Train Logistic Regression + XGBoost, compare vs Altman
├── step7_shap.py              # SHAP explainability — why each prediction is made
│
├── bankruptcy_data.csv        # Raw downloaded data (43,405 rows)
├── bankruptcy_clean.csv       # Cleaned data (26,737 rows, ready for modeling)
│
└── README.md                  # This file

▶️ How to Run This Project
Prerequisites

Python 3.10 or higher
Anaconda (recommended) or standard Python installation

Step 1: Clone the repository
bashgit clone https://github.com/YOUR_USERNAME/bankruptcy-prediction.git
cd bankruptcy-prediction
Step 2: Install dependencies
bashpip install pandas numpy matplotlib seaborn scikit-learn xgboost shap scipy imbalanced-learn ucimlrepo
Step 3: Run the pipeline in order
bash# Download and save the data
python step1_collect_data.py

# Explore the data
python step2_explore.py

# Visualize distributions
python step3_visualize.py

# Find key ratios
python step4_key_ratios.py

# Clean the data
python step5_cleaning.py

# Train models and compare
python step6_model.py

# Generate SHAP explanations
python step7_shap.py

Note: Step 1 requires an internet connection to download the dataset from UCI. All subsequent steps run locally using the saved CSV files.

Expected Output
After running all steps you will have:

bankruptcy_data.csv — raw dataset
bankruptcy_clean.csv — processed dataset
first_look.png — class balance and distribution charts
key_ratios.png — top differentiating ratios
model_comparison.png — AUC comparison chart
shap_importance.png — feature importance chart
shap_impact.png — SHAP beeswarm plot


💼 Business Impact
This type of model has direct applications in:
Use CaseWho Uses ItValueCredit Risk AssessmentBanks before issuing loansAvoid lending to companies about to collapseInvestment Due DiligenceHedge funds, PE firmsDon't buy equity in a dying companyPortfolio MonitoringAsset managersFlag holdings showing early warning signsSupplier RiskLarge corporationsAvoid depending on financially fragile suppliersRegulatory ComplianceCredit rating agenciesExplainable AI required by regulators
A bank that lends $10M to a company 18 months before bankruptcy loses that money. A model that flags that company 3 years early saves it entirely.

🛠️ Tools & Technologies
ToolPurposePython 3.12Core programming languagePandasData manipulation and cleaningNumPyNumerical computationsMatplotlib & SeabornData visualizationScikit-learnLogistic Regression, train/test split, metricsXGBoostPrimary prediction modelSHAPModel explainabilityUCI ML RepositoryDataset source

🚀 What I Would Do Next

Rename columns — Map A1–A64 to actual ratio names (debt-to-equity, current ratio etc.) using the UCI documentation for clearer interpretability
Add risk tiers — Output High / Watch / Safe risk categories instead of just a probability score
Sector analysis — Do certain industries show warning signs earlier than others?
Time series analysis — Track the same company across multiple years to see how risk scores evolve
Deploy as an API — Wrap the model in a Flask API so analysts can input financials and get instant risk scores


👩‍💻 Author
Rimjhim Ghawry
Aspiring Data Analyst | Finance & ML
📧 your.email@gmail.com
🔗 linkedin.com/in/yourprofile

📄 Citation
Zieba, M., Tomczak, S., & Tomczak, J. (2016). Ensemble Boosted Trees with Synthetic Features Generation in Application to Bankruptcy Prediction. Expert Systems with Applications.
Dataset: https://archive.uci.edu/dataset/365/polish+companies+bankruptcy+data
EOFShareContent
