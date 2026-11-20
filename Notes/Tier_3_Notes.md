# C-DAC AIML: Data Analytics & Statistics — Tier 3 Study Notes

> **Tier Classification**: **TIER 3 (Read Once — Quick Reading & Specialized Domains)**  
> **Coverage**: Conceptual overviews, software architectures, specialized statistical methods, operations, Question Indicators & "Tellers" (How to Spot and Answer in Exams), and High-Yield Examination MCQs for Topics 36 to 50.  
> **Source Material**: Synthesized from C-DAC Data Analytics (DA) Modules 01–10 and Statistics Sessions 01–11.

---

## Table of Contents
36. [Topic 36: Analysis vs Reporting](#topic-36-analysis-vs-reporting)
37. [Topic 37: Modern Data Analytics Tools](#topic-37-modern-data-analytics-tools)
38. [Topic 38: Analytic Processes and Tools](#topic-38-analytic-processes-and-tools)
39. [Topic 39: Statistical Interfaces](#topic-39-statistical-interfaces)
40. [Topic 40: Benefits and Challenges of Predictive Modelling](#topic-40-benefits-and-challenges-of-predictive-modelling)
41. [Topic 41: Future and Limitations of Predictive Modelling](#topic-41-future-and-limitations-of-predictive-modelling)
42. [Topic 42: Predictive Modelling Tools](#topic-42-predictive-modelling-tools)
43. [Topic 43: Installation, Acceptance and Operation](#topic-43-installation-acceptance-and-operation)
44. [Topic 44: Quality Assurance and Documentation](#topic-44-quality-assurance-and-documentation)
45. [Topic 45: Factor Analysis](#topic-45-factor-analysis)
46. [Topic 46: Directional Data Analytics](#topic-46-directional-data-analytics)
47. [Topic 47: Functional Data Analysis](#topic-47-functional-data-analysis)
48. [Topic 48: Power BI Architecture and Interface](#topic-48-power-bi-architecture-and-interface)
49. [Topic 49: Power Query](#topic-49-power-query)
50. [Topic 50: Interactive Dashboards and Sharing Insights](#topic-50-interactive-dashboards-and-sharing-insights)

---

## Topic 36: Analysis vs Reporting

### 1. What the Topic Is
- **Reporting**: Passive aggregation, formatting, and delivery of historical operational data. Answers *"What happened?"*.
- **Analysis**: Active, hypothesis-driven interrogation of data to isolate causal mechanisms and prescribe business action. Answers *"Why did it happen?"* and *"What should we do?"*.

---

### 2. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)
- *"Automated daily PDF report of ticket counts"* $\to$ **Reporting** (delivers facts without diagnosis or strategy).

#### MCQ: Reporting vs Analysis
**Question**: An operational manager receives an automated daily PDF showing total call center tickets resolved by each agent. How is this artifact classified, and why?
- A) Predictive Analytics; It predicts agent tenure.
- B) Reporting; It organizes and delivers historical operational records without diagnosing root causes or prescribing strategy.
- C) Prescriptive Modeling; It optimizes shift schedules.
- D) Exploratory Data Analysis; It handles unlabelled clusters.
* **Spotting the Clue (The Teller)**: *"Automated daily PDF showing total resolved tickets"*.
* **Answering Strategy**: Delivering factual historical metrics = Reporting.
* **Correct Answer**: **B**  
* **Explanation**: Standardized daily aggregation and delivery of historical operational facts constitutes Reporting. Option B is correct.

---

## Topic 37: Modern Data Analytics Tools

### 1. What the Topic Is
Modern cloud data architectures (Snowflake, BigQuery, ClickHouse) decouple compute from storage and store data in **columnar format** to optimize analytical aggregate queries.

---

### 2. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)
- *"Why do cloud data warehouses utilize columnar storage?"* $\to$ Analytical queries aggregate specific columns across millions of rows; columnar reads only needed columns, reducing I/O.

#### MCQ: Columnar Storage Advantage
**Question**: Why do cloud data platforms like Snowflake and BigQuery utilize columnar storage instead of traditional row-oriented storage for analytical queries?
- A) Columnar storage speeds up single-row updates and transactional rollbacks.
- B) Analytical queries typically aggregate a few specific columns across millions of rows; columnar storage allows reading only the needed columns, reducing I/O and improving compression.
- C) Columnar storage eliminates the need for SQL queries.
- D) It guarantees that all machine learning models achieve zero bias.
* **Spotting the Clue (The Teller)**: *"Why cloud warehouses use columnar storage for analytical queries"*.
* **Answering Strategy**: Analytical aggregation scans columns, not whole rows $\implies$ columnar format reduces I/O.
* **Correct Answer**: **B**  
* **Explanation**: OLAP queries aggregate specific columns across millions of records. Columnar format reads only the requested columns and achieves high compression, reducing I/O overhead. Option B is correct.

---

## Topic 38: Analytic Processes and Tools

### 1. What the Topic Is
Transition from ETL (Extract-Transform-Load) to modern ELT (Extract-Load-Transform), loading raw data directly into the cloud warehouse and leveraging its distributed compute engine.

---

### 2. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)
- *"Primary operational advantage of ELT over ETL"* $\to$ Preserves untransformed raw data in warehouse and leverages scalable warehouse compute for transformations.

#### MCQ: ELT Architectural Advantage
**Question**: What is the primary operational advantage of transitioning from an ETL architecture to an ELT architecture in a modern enterprise analytics environment?
- A) It removes the need for data transformation entirely.
- B) Raw data is immediately preserved in the warehouse, and transformations leverage the scalable compute power of the cloud data warehouse without secondary staging bottlenecks.
- C) It guarantees that data is always stored in non-relational CSV files.
- D) It reduces network bandwidth requirements to zero.
* **Spotting the Clue (The Teller)**: *"Advantage of transitioning from ETL to ELT"*.
* **Answering Strategy**: ELT stores raw data first, transforms using warehouse compute.
* **Correct Answer**: **B**  
* **Explanation**: ELT loads untransformed raw data directly into the warehouse, preserving historical fidelity and leveraging parallel cloud query engines (e.g., BigQuery, Snowflake) to transform data on-demand. Hence Option B is correct.

---

## Topic 39: Statistical Interfaces

### 1. What the Topic Is
Statistical programming interfaces (Python `statsmodels`, `scipy.stats`, R formula syntax `y ~ x1 + x2`) provide validated implementations for regression, ANOVA, and hypothesis tests.

---

### 2. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)
- *"`C(Category)` operator in statsmodels / Patsy formulas"* $\to$ Automatically creates dummy indicator variables with one reference level omitted.

#### MCQ: Statsmodels Formula Operator
**Question**: In Python's `statsmodels` library, what does the formula operator `C(Category)` represent when fitting an OLS regression model?
- A) It computes the continuous Pearson correlation of the variable.
- B) It treats the variable as a categorical factor, automatically generating dummy (one-hot) indicator variables with one reference level omitted.
- C) It applies a Box-Cox logarithmic transformation.
- D) It clips outliers outside 3 standard deviations.
* **Spotting the Clue (The Teller)**: *"`C(Category)` operator in statsmodels"*.
* **Answering Strategy**: `C()` = Categorical factor conversion with reference level omitted.
* **Correct Answer**: **B**  
* **Explanation**: The `C()` operator in Patsy/`statsmodels` designates an attribute as a categorical factor, automatically creating $k - 1$ dummy indicator variables to prevent full rank collinearity with the intercept. Option B is correct.

---

## Topic 40: Benefits and Challenges of Predictive Modelling

### 1. What the Topic Is
Predictive modelling converts historical records into proactive forecasts. Key production challenges include Covariate Shift, Label Drift, Feedback Loops, and **Concept Drift** ($P(Y \mid \mathbf{X})$ changes over time).

---

### 2. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)
- *"Model trained during economic expansion begins failing during a recession despite input features having similar ranges"* $\to$ **Concept Drift**.

#### MCQ: Recognizing Concept Drift
**Question**: A credit scoring model trained during an economic expansion begins approving loans during a recession, causing a sudden spike in defaults despite input features having similar ranges. What has occurred?
- A) Multicollinearity between predictors
- B) Concept Drift ($P(Y \mid \mathbf{X})$ changed due to macroeconomic shifts)
- C) Underfitting due to excessive pruning
- D) High-cardinality bias in tree leaves
* **Spotting the Clue (The Teller)**: *"Trained during expansion, defaults spike during recession despite similar feature ranges"*.
* **Answering Strategy**: The conditional relationship between inputs and output changed = Concept Drift.
* **Correct Answer**: **B**  
* **Explanation**: The relationship between applicant attributes and default risk changed due to the economic recession. When $P(Y \mid \mathbf{X})$ shifts over time, this is Concept Drift. Option B is correct.

---

## Topic 41: Future and Limitations of Predictive Modelling

### 1. What the Topic Is
Predictive models capture observational associations ($P(Y \mid X)$), not causal relationships. Causal interventions require experimental A/B testing or Pearl's $do$-calculus ($P(Y \mid do(X))$).

---

### 2. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)
- *"Why is observational predictive modeling insufficient for setting business pricing policies?"* $\to$ Historical pricing is confounded with promotions/seasons; changing prices is an intervention ($do(X)$) requiring causal modeling.

#### MCQ: Observational vs Causal Prediction
**Question**: Why is a standard predictive model ($P(Y \mid X)$) generally insufficient for setting business pricing policies without causal analysis or experimental A/B testing?
- A) Predictive models cannot process continuous numeric features.
- B) Historical data reflects past policy rules and customer reactions (confounding); changing prices ($do(\text{Price} = p)$) alters the environment and can yield outcomes different from historical correlations.
- C) Prediction intervals are only valid when $n > 10,000$.
- D) Linear regression models cannot calculate derivatives.
* **Spotting the Clue (The Teller)**: *"Why standard predictive model insufficient for setting pricing policy"*.
* **Answering Strategy**: Confounding in observational data means correlation $\ne$ causation; interventions require causal $do$-calculus.
* **Correct Answer**: **B**  
* **Explanation**: Standard prediction conditions on historical observational data where price was tied to other variables (promotions, seasonality, customer segment). Intervening requires causal understanding ($do$-calculus or randomized experiments) to evaluate how customers will respond to a new pricing policy. Option B is correct.

---

## Topic 42: Predictive Modelling Tools

### 1. What the Topic Is
Production machine learning toolkits: `scikit-learn` (unified APIs), `XGBoost`, `LightGBM` (Gradient-based One-Side Sampling GOSS, Exclusive Feature Bundling EFB), and `CatBoost`.

---

### 2. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)
- *"Architectural innovation of LightGBM over traditional boosting"* $\to$ GOSS and EFB with leaf-wise tree growth.

#### MCQ: LightGBM Architectural Innovation
**Question**: What architectural innovation allows `LightGBM` to train significantly faster than traditional gradient boosting implementations on large tabular datasets?
- A) It replaces decision trees with linear perceptrons.
- B) Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) to reduce sample size and feature dimensionality while building trees leaf-wise.
- C) It eliminates the need for cross-validation.
- D) It avoids computing gradients during training.
* **Spotting the Clue (The Teller)**: *"Architectural innovation allowing LightGBM to train faster"*.
* **Answering Strategy**: GOSS (samples by gradients) + EFB (bundles sparse features) + leaf-wise growth.
* **Correct Answer**: **B**  
* **Explanation**: LightGBM achieves high speed through GOSS (keeping instances with large gradients and downsampling instances with small gradients) and EFB (bundling mutually exclusive sparse features), combined with a leaf-wise (best-first) tree growth strategy. Option B is correct.

---

## Topic 43: Installation, Acceptance and Operation

### 1. What the Topic Is
MLOps operationalization: Docker containers, ONNX runtimes, and deployment strategies (Shadow Deployment, Canary Deployment).

---

### 2. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)
- *"Shadow Deployment"* $\to$ Routes live traffic to new model in parallel; logs outputs without showing predictions to users.

#### MCQ: Purpose of Shadow Deployment
**Question**: What is the primary purpose of a "Shadow Deployment" when deploying a new machine learning model to production?
- A) To reduce server hosting costs by 50%.
- B) To evaluate real-world inference latency, stability, and prediction distributions on live traffic without exposing end-users to the new model's output.
- C) To automatically retrain the model weights using backpropagation.
- D) To bypass company security firewalls.
* **Spotting the Clue (The Teller)**: *"Primary purpose of Shadow Deployment"*.
* **Answering Strategy**: Shadow = receives real traffic in background, logs results without user impact.
* **Correct Answer**: **B**  
* **Explanation**: Shadow deployments route live traffic to the new model in parallel with the current production system. The new model's outputs are logged for monitoring and verification without impacting actual user experience. Option B is correct.

---

## Topic 44: Quality Assurance and Documentation

### 1. What the Topic Is
Model governance and documentation: Model Cards (Mitchell et al., 2019) detailing intended use, evaluation data, subgroup demographic metrics, and ethical limits.

---

### 2. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)
- *"Model Card primary role"* $\to$ Standardized document of architecture, performance across subgroups, limitations, and ethical considerations.

#### MCQ: Governance via Model Cards
**Question**: In enterprise machine learning governance, what is the primary role of a "Model Card"?
- A) A GPU hardware expansion board used to accelerate training.
- B) A standardized document that details a model's architecture, intended use, performance across demographic subgroups, limitations, and ethical considerations.
- C) A license key granting commercial access to a closed-source API.
- D) A visualization tool that displays the confusion matrix in 3D.
* **Spotting the Clue (The Teller)**: *"Primary role of a Model Card"*.
* **Answering Strategy**: Model Card = structured documentation of model details, limitations, and subgroup metrics.
* **Correct Answer**: **B**  
* **Explanation**: Introduced by Google Research, Model Cards provide structured documentation detailing model purpose, evaluation metrics, subgroup performance, and operational limitations to ensure transparency and accountability. Option B is correct.

---

## Topic 45: Factor Analysis

### 1. What the Topic Is
Exploratory Factor Analysis (EFA) uncovers latent common factors $\mathbf{f}$ explaining observed covariance: $\mathbf{x} = \boldsymbol{\Lambda}\mathbf{f} + \boldsymbol{\epsilon}$. Total variance decomposes into **Communality** ($h_i^2 = \sum \lambda_{ij}^2$) and **Unique Variance** ($\psi_i = 1 - h_i^2$).

---

### 2. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)
- *"Factor loadings $\lambda_1 = 0.60, \lambda_2 = 0.80$ on orthogonal factors. Find Communality and Unique Variance"*.
- *Answering Rule*: $h^2 = 0.60^2 + 0.80^2 = 0.36 + 0.64 = 1.00$. Unique variance $\psi = 1 - h^2 = 1 - 1.00 = 0.00$.

#### MCQ: Factor Loading & Communality Calculation
**Question**: In an Exploratory Factor Analysis of standardized survey items, an item has factor loadings $\lambda_1 = 0.60$ and $\lambda_2 = 0.80$ on two retained orthogonal factors. What is the Communality ($h^2$) and Unique Variance ($\psi$) of this item?
- A) $h^2 = 1.00$; $\psi = 0.00$
- B) $h^2 = 0.48$; $\psi = 0.52$
- C) $h^2 = 1.00$; $\psi = 0.20$
- D) $h^2 = 0.28$; $\psi = 0.72$
* **Spotting the Clue (The Teller)**: *"Factor loadings $\lambda_1 = 0.60, \lambda_2 = 0.80$ ... Communality and Unique Variance"*.
* **Answering Strategy**: Communality = sum of squared loadings: $0.36 + 0.64 = 1.00$. Unique variance = $1 - h^2 = 0.00$.
* **Correct Answer**: **A**  
* **Explanation**: Communality is the sum of squared factor loadings across orthogonal factors: $h^2 = 0.60^2 + 0.80^2 = 0.36 + 0.64 = 1.00$. Unique variance $\psi = 1 - h^2 = 0.00$. Option A is correct.

---

## Topic 46: Directional Data Analytics

### 1. What the Topic Is
Directional Data Analytics (Circular Statistics) handles angular data on the circle $\mathbb{S}^1$ where $0^\circ \equiv 360^\circ$. Standard arithmetic means fail ($10^\circ$ and $350^\circ$ average to $180^\circ$ Due South under arithmetic mean, but their true average heading is $0^\circ$ Due North).

---

### 2. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)
- *"Compass headings / angles / wind direction averaging"* $\to$ **Circular Statistics / Mean Direction $\bar{\theta} = \text{atan2}(\bar{S}, \bar{C})$**.

#### MCQ: Circular Directional Average
**Question**: An environmental sensor records two compass headings: $\theta_1 = 30^\circ$ and $\theta_2 = 330^\circ$. What is the correct circular mean direction $\bar{\theta}$?
- A) $180^\circ$
- B) $0^\circ$ (or $360^\circ$)
- C) $90^\circ$
- D) $270^\circ$
* **Spotting the Clue (The Teller)**: *"Compass headings $30^\circ$ and $330^\circ$ ... circular mean direction"*.
* **Answering Strategy**: $\bar{C} = (\cos 30 + \cos 330)/2 = \sqrt{3}/2$. $\bar{S} = (\sin 30 + \sin 330)/2 = 0$. $\text{atan2}(0, \sqrt{3}/2) = 0^\circ$ (North).
* **Correct Answer**: **B**  
* **Explanation**: $\bar{C} = \frac{\sqrt{3}/2 + \sqrt{3}/2}{2} \approx 0.866$. $\bar{S} = \frac{0.5 - 0.5}{2} = 0.0$. $\bar{\theta} = \text{atan2}(0.0, 0.866) = 0^\circ$ (Due North). Arithmetic averaging gives $180^\circ$ (Due South), which is completely wrong. Option B is correct.

---

## Topic 47: Functional Data Analysis

### 1. What the Topic Is
Functional Data Analysis (FDA) treats individual observations not as discrete vectors, but as continuous functions/curves $x_i(t)$ evaluated over a continuum using basis function expansions (e.g. B-splines, Fourier series).

---

### 2. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)
- *"Role of B-spline basis expansion in FDA"* $\to$ Converts discrete, noisy point observations over a continuum into a smooth, continuous, differentiable curve $x(t)$.

#### MCQ: B-Spline Expansion in FDA
**Question**: What is the primary role of a B-spline basis expansion in Functional Data Analysis?
- A) To convert continuous signals into discrete binary tokens.
- B) To convert discrete, noisy point observations sampled over a continuum into a smooth, continuous, differentiable mathematical function $x(t)$.
- C) To calculate the Pearson correlation matrix.
- D) To eliminate the need for cross-validation.
* **Spotting the Clue (The Teller)**: *"Role of B-spline basis expansion in Functional Data Analysis"*.
* **Answering Strategy**: Projects discrete noisy measurements into a smooth continuous curve $x(t)$.
* **Correct Answer**: **B**  
* **Explanation**: B-spline basis expansion projects discrete noisy measurements onto a linear combination of polynomial basis splines, yielding a continuous, differentiable function $x(t)$ for subsequent functional analysis. Option B is correct.

---

## Topic 48: Power BI Architecture and Interface

### 1. What the Topic Is
Power BI architecture: VertiPaq in-memory columnar engine, Import mode vs DirectQuery mode. Star Schema is recommended over Snowflake Schema to minimize join traversals and optimize VertiPaq compression.

---

### 2. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)
- *"Why is Star Schema recommended over Snowflake Schema in Power BI VertiPaq?"* $\to$ Minimizes table relationship traversals during filter propagation and optimizes in-memory columnar compression.

#### MCQ: Star Schema in Power BI
**Question**: Why is a Star Schema strongly recommended over a highly normalized Snowflake Schema when designing semantic data models for the Power BI VertiPaq engine?
- A) VertiPaq cannot process foreign keys.
- B) Star schemas minimize table relationship traversals, reducing relationship hops during filter propagation and improving memory compression in VertiPaq.
- C) Star schemas eliminate the need for DAX measures.
- D) DirectQuery mode only works with star schemas.
* **Spotting the Clue (The Teller)**: *"Why Star Schema recommended over Snowflake Schema for VertiPaq"*.
* **Answering Strategy**: Star schema minimizes join hops and optimizes VertiPaq columnar scans.
* **Correct Answer**: **B**  
* **Explanation**: Star schemas simplify filter context propagation by reducing relationship joins between dimension and fact tables, optimizing VertiPaq columnar query execution. Option B is correct.

---

## Topic 49: Power Query

### 1. What the Topic Is
Power Query ETL engine driven by the M formula language. **Query Folding** translates M transformation steps into native SQL executed on the source database server.

---

### 2. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)
- *"What happens when Query Folding breaks?"* $\to$ Transformations can no longer be pushed to source database; Power Query must download the raw dataset over network and process locally in RAM.

#### MCQ: Breaking Query Folding
**Question**: In Power Query ETL, what happens when an analyst inserts a custom step that breaks "Query Folding" early in a transformation pipeline?
- A) Power Query aborts execution and marks the dataset as corrupted.
- B) Subsequent transformations cannot be offloaded to the source database engine; Power Query must download the remaining raw dataset over the network and perform transformations locally in desktop memory.
- C) The M language automatically converts into DAX.
- D) Row-Level Security is disabled.
* **Spotting the Clue (The Teller)**: *"What happens when Query Folding breaks in Power Query"*.
* **Answering Strategy**: Breaking folding = source stops executing $\implies$ forces local download and processing in desktop RAM.
* **Correct Answer**: **B**  
* **Explanation**: When query folding breaks, the source engine can no longer execute subsequent steps. Power Query must pull all data up to that point over the network and process remaining steps using local memory, which can significantly degrade performance. Hence Option B is correct.

---

## Topic 50: Interactive Dashboards and Sharing Insights

### 1. What the Topic Is
Dashboard UX design, dynamic Row-Level Security (RLS) via DAX (`[Email] = USERPRINCIPALNAME()`), and distinguishing Calculated Columns (static in RAM) from DAX Measures (computed on-the-fly under active filter context).

---

### 2. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)
- *"Displaying Profit Margin ratio that recalculates dynamically as users filter by slicers"* $\to$ **DAX Measure**, NOT Calculated Column.

#### MCQ: Calculated Column vs DAX Measure
**Question**: An analyst needs to display Total Profit Margin in a Power BI dashboard so that it recalculates accurately whenever users slice by Region, Quarter, or Product Category. Should this be implemented as a Calculated Column or a DAX Measure, and why?
- A) Calculated Column; Because it stores values in RAM for faster performance.
- B) DAX Measure; Because measures recalculate dynamically based on the visual filter context of the active user session.
- C) Calculated Column; Because DAX measures cannot perform division.
- D) Power Query M Step; Because DAX cannot handle aggregations.
* **Spotting the Clue (The Teller)**: *"Profit margin ratio recalculating accurately across dynamic slicers"*.
* **Answering Strategy**: Dynamic aggregation of ratios = DAX Measure (calculated columns average row-level ratios incorrectly).
* **Correct Answer**: **B**  
* **Explanation**: Profit margin is a ratio ($\sum \text{Profit} / \sum \text{Revenue}$). Averaging a calculated column of row-level margins produces mathematically incorrect results. A DAX measure computes the sums dynamically within the active filter context, ensuring mathematical accuracy across slicers. Option B is correct.

---
