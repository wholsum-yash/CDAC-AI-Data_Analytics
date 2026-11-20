# C-DAC AIML: Data Analytics & Statistics — Tier 1 Study Notes

> **Tier Classification**: **TIER 1 (Highest Priority — Master These)**  
> **Coverage**: Foundational principles, mathematical definitions, derivations/formulas, intuition, Question Indicators & "Tellers" (How to Spot and Answer in Exams), and High-Yield Examination MCQs for Topics 1 to 15.  
> **Source Material**: Synthesized from C-DAC Data Analytics (DA) Modules 01–10 and Statistics Sessions 01–11.

---

## Table of Contents
1. [Topic 01: Descriptive Statistics](#topic-01-descriptive-statistics)
2. [Topic 02: Probability: Joint, Conditional and Marginal](#topic-02-probability-joint-conditional-and-marginal)
3. [Topic 03: Bayes' Theorem](#topic-03-bayes-theorem)
4. [Topic 04: Probability Distributions: Normal, Binomial, Poisson](#topic-04-probability-distributions-normal-binomial-poisson)
5. [Topic 05: Central Limit Theorem](#topic-05-central-limit-theorem)
6. [Topic 06: Correlation and Covariance](#topic-06-correlation-and-covariance)
7. [Topic 07: Pearson Correlation](#topic-07-pearson-correlation)
8. [Topic 08: Hypothesis Testing](#topic-08-hypothesis-testing)
9. [Topic 09: Z-Test and Chi-Square Test](#topic-09-z-test-and-chi-square-test)
10. [Topic 10: Regression Analysis](#topic-10-regression-analysis)
11. [Topic 11: Predictive Modelling](#topic-11-predictive-modelling)
12. [Topic 12: Supervised Segmentation and Decision Trees](#topic-12-supervised-segmentation-and-decision-trees)
13. [Topic 13: Overfitting, Generalization and Cross-Validation](#topic-13-overfitting-generalization-and-cross-validation)
14. [Topic 14: Evaluating Classifiers](#topic-14-evaluating-classifiers)
15. [Topic 15: Sampling and Estimation](#topic-15-sampling-and-estimation)

---

## Topic 01: Descriptive Statistics

### 1. What the Topic Is
Descriptive statistics encompasses numerical, tabular, and graphical procedures designed to organize, summarize, and elucidate the salient characteristics of a collected sample or entire population dataset without making inductive inferences beyond the observed data. It is partitioned into three core dimensions:
- **Measures of Central Tendency (Location)**: Arithmetic Mean, Trimmed Mean, Geometric Mean, Harmonic Mean, Median, Mode.
- **Measures of Dispersion (Spread/Variability)**: Range, Interquartile Range ($IQR$), Variance ($s^2$), Standard Deviation ($s$), Mean Absolute Deviation ($MAD$), Coefficient of Variation ($CV$).
- **Measures of Shape & Relative Position**: Percentiles, Quartiles, Skewness, Kurtosis.

### 2. Intuition
Imagine entering an unfamiliar country and asking, "What is the economic reality of the average citizen?" If a few billionaires live next to impoverished communities, reporting the arithmetic mean salary presents a misleadingly prosperous illusion because the mean acts as a physical center of gravity pulled violently by extreme outliers. The median, by contrast, identifies the exact citizen standing in the middle of a sorted line, unaffected by how excessively wealthy the richest citizen is. 
Simultaneously, location alone is insufficient. Knowing the average water depth of a river is 4 feet does not prevent you from drowning if the dispersion ranges from 1 foot to 15 feet. Dispersion gauges the risk, unpredictability, and spread behind any summary average.

### 3. Mathematical Formula

#### A. Measures of Central Tendency
- **Arithmetic Mean**: $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$.
- **Harmonic Mean (Averaging rates/speeds over equal distances)**: $HM = \frac{n}{\sum_{i=1}^n \frac{1}{x_i}}$.
- **Geometric Mean (Compounding growth rates, investment returns)**: $GM = (\prod_{i=1}^n x_i)^{1/n}$.
  *Inequality*: $\min(x) \le HM \le GM \le \bar{x} \le \max(x)$.
- **Median ($\tilde{x}$)**: Middle value in sorted array $x_{(1)} \le \dots \le x_{(n)}$.

#### B. Measures of Dispersion
- **Sample Variance (Bessel's Correction)**: $s^2 = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})^2$.
- **Interquartile Range**: $IQR = Q_3 - Q_1$.
- **Coefficient of Variation (Dimensionless relative risk)**: $CV = \left( \frac{s}{\bar{x}} \right) \times 100\%$.

---

### 4. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)

#### The "Teller" / Trigger Keywords
- *"Best measure of central tendency for skewed data / extreme income outliers"* $\to$ **Median**.
- *"Averaging speeds across equal distances / rates of work"* $\to$ **Harmonic Mean**.
- *"Compounding portfolio investment returns / growth rates over time"* $\to$ **Geometric Mean**.
- *"Bessel's correction / why divide by $n - 1$ instead of $n$"* $\to$ **Unbiasedness ($E[s^2] = \sigma^2$)**.
- *"Comparing risk or variability between two stocks with different price scales"* $\to$ **Coefficient of Variation ($CV$)**.
- *"Sum of deviations about the arithmetic mean"* $\to$ **Always equals 0**.

#### Question Archetypes & Answering Strategies
1. **Outlier-Resistant Location Archetype**:
   - *Teller Clue*: Dataset has positive or negative outliers (e.g. $[12, 15, 15, 18, 140]$). Question asks which location metric represents the typical value.
   - *Answering Rule*: Eliminate Mean immediately. Select **Median**.
2. **Rate/Speed Averaging Archetype**:
   - *Teller Clue*: A car travels to a city at 40 km/h and returns along the identical route at 60 km/h. What is the average speed?
   - *Answering Rule*: Do NOT take the arithmetic mean ($(40+60)/2 = 50$ is WRONG!). Equal distance averaging requires **Harmonic Mean**: $HM = \frac{2}{\frac{1}{40} + \frac{1}{60}} = \frac{2}{\frac{3+2}{120}} = \frac{240}{5} = 48\text{ km/h}$.
3. **Relative Spread Comparison Archetype**:
   - *Teller Clue*: Asset A has mean $\$100, s = \$10$; Asset B has mean $\$10, s = \$2$. Which asset has higher relative volatility?
   - *Answering Rule*: Compute $CV = s/\bar{x}$. Asset A: $10/100 = 10\%$. Asset B: $2/10 = 20\%$. Asset B has higher relative dispersion despite smaller absolute standard deviation.

---

### 5. Exam-Style MCQs Covering the Possibilities

#### MCQ 1: Robust Central Tendency with Outliers
**Question**: An analyst evaluates five server response times recorded in milliseconds: `[12, 18, 15, 15, 140]`. Which measure of central tendency provides the most representative latency, and what is the exact Interquartile Range ($IQR$) if quartiles are computed using standard positional percentiles $Q_1 = 13.5$ and $Q_3 = 77.5$?
- A) Arithmetic Mean (Latency = 40 ms); $IQR = 64.0\text{ ms}$
- B) Median (Latency = 15 ms); $IQR = 64.0\text{ ms}$
- C) Harmonic Mean (Latency = 140 ms); $IQR = 125.0\text{ ms}$
- D) Median (Latency = 15 ms); $IQR = 15.0\text{ ms}$

* **Spotting the Clue (The Teller)**: Latency data `[12, 18, 15, 15, 140]` contains an extreme outlier ($140\text{ ms}$).
* **Answering Strategy**: Outliers skew the mean ($200/5 = 40\text{ ms}$). The median of sorted `[12, 15, 15, 18, 140]` is $15\text{ ms}$. $IQR = Q_3 - Q_1 = 77.5 - 13.5 = 64.0\text{ ms}$.
* **Correct Answer**: **B**  
* **Explanation**: The arithmetic mean ($40\text{ ms}$) is inflated by the single outlier ($140\text{ ms}$). The median ($15\text{ ms}$) accurately captures typical latency. $IQR = 77.5 - 13.5 = 64.0\text{ ms}$. Option B is correct.

#### MCQ 2: Rate Averaging via Harmonic Mean
**Question**: A delivery truck travels from Warehouse A to Warehouse B at a constant speed of $40\text{ km/h}$, and returns along the identical route at a constant speed of $60\text{ km/h}$. What is the true average round-trip speed of the vehicle?
- A) $50.0\text{ km/h}$
- B) $48.0\text{ km/h}$
- C) $46.5\text{ km/h}$
- D) $52.0\text{ km/h}$

* **Spotting the Clue (The Teller)**: *"Travels at 40 km/h and returns along the identical route at 60 km/h ... average speed"*.
* **Answering Strategy**: Averaging speeds over equal distances requires the Harmonic Mean: $HM = \frac{2}{\frac{1}{40} + \frac{1}{60}} = 48\text{ km/h}$.
* **Correct Answer**: **B**  
* **Explanation**: Because the distance $d$ in both directions is equal, the truck spends more time traveling at the slower speed than the faster speed ($t_1 = d/40, t_2 = d/60$). Total speed = $\frac{2d}{t_1 + t_2} = \frac{2}{\frac{1}{40} + \frac{1}{60}} = \frac{240}{5} = 48\text{ km/h}$. Arithmetic mean ($50\text{ km/h}$) is incorrect. Option B is correct.

---

## Topic 02: Probability: Joint, Conditional and Marginal

### 1. What the Topic Is
Probability theory formalizes uncertainty across stochastic events within an overarching sample space $\Omega$:
- **Joint Probability $P(A \cap B)$**: The likelihood that event $A$ AND event $B$ happen simultaneously.
- **Marginal Probability $P(A)$**: The unconditional probability of event $A$ occurring regardless of outcome $B$, derived by summing (marginalizing) the joint probabilities across all possible states of $B$.
- **Conditional Probability $P(A \mid B)$**: The revised probability of event $A$ occurring given that condition $B$ has already occurred ($P(A \mid B) = \frac{P(A \cap B)}{P(B)}$).

### 2. Intuition
Consider a contingency table of 1,000 banking customers categorized by Credit Score (Good vs Bad) and Loan Default (Yes vs No).
- *Joint probability* points to a single cell (e.g. Bad Credit Score AND Defaults).
- *Marginal probability* looks at the row or column total written in the margin (total default rate, ignoring credit score).
- *Conditional probability* zooms into one specific column: "Given that someone has a Bad Credit Score, what fraction of that column defaults?" Conditioning collapses the denominator from the whole universe to just group $B$.

### 3. Mathematical Formula
- **Conditional Probability**: $P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$.
- **General Multiplication Rule**: $P(A \cap B) = P(A \mid B)P(B) = P(B \mid A)P(A)$.
- **Independence Test**: $A \perp B \iff P(A \cap B) = P(A)P(B) \iff P(A \mid B) = P(A)$.
- **Law of Total Probability**: $P(A) = \sum_{j=1}^k P(A \mid B_j) P(B_j)$.

---

### 4. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)

#### The "Teller" / Trigger Keywords
- *"Given that / knowing that event B has occurred"* $\to$ **Conditional Probability**.
- *"Overall probability across all groups / unconditioned probability"* $\to$ **Marginal Probability**.
- *"Both A and B occur simultaneously / intersection ($A \cap B$)"* $\to$ **Joint Probability**.
- *"Are events A and B statistically independent?"* $\to$ **Check if $P(A \cap B) = P(A)P(B)$**.
- *"Can two mutually exclusive events with positive probabilities be independent?"* $\to$ **Never ($P(A \cap B) = 0 \ne P(A)P(B)$)**.

#### Question Archetypes & Answering Strategies
1. **Contingency Table Conditioning Archetype**:
   - *Teller Clue*: Given $P(A)$, $P(B)$, and $P(A \cap B)$. Find $P(B \mid A)$ and check for independence.
   - *Answering Rule*:
     1. $P(B \mid A) = \frac{P(A \cap B)}{P(A)}$.
     2. Check independence: If $P(A \cap B) = P(A) \times P(B)$, independent. If not, dependent.
2. **Mutual Exclusivity vs Independence Trap Archetype**:
   - *Teller Clue*: Question asks whether disjoint (mutually exclusive) events can be statistically independent.
   - *Answering Rule*: Mutually exclusive means $P(A \cap B) = 0$. For independent events, $P(A \cap B) = P(A)P(B) > 0$. Therefore, non-trivial mutually exclusive events are **strictly dependent**.

---

### 5. Exam-Style MCQs Covering the Possibilities

#### MCQ 1: Conditional Calculation & Independence Check
**Question**: On an e-commerce platform, 30% of visitors browse electronics ($P(E) = 0.30$), 20% make a checkout purchase ($P(C) = 0.20$), and 12% both browse electronics and make a purchase ($P(E \cap C) = 0.12$). What is the probability that a visitor makes a purchase given that they browsed electronics, and are browsing and purchasing independent?
- A) $P(C \mid E) = 0.40$; Events are Independent
- B) $P(C \mid E) = 0.40$; Events are Dependent
- C) $P(C \mid E) = 0.60$; Events are Independent
- D) $P(C \mid E) = 0.25$; Events are Dependent

* **Spotting the Clue (The Teller)**: *"Probability of purchase GIVEN that they browsed electronics ... are they independent"*.
* **Answering Strategy**: $P(C \mid E) = \frac{P(E \cap C)}{P(E)} = \frac{0.12}{0.30} = 0.40$. Independence requires $P(E \cap C) = P(E)P(C) = 0.30 \times 0.20 = 0.06 \ne 0.12 \implies$ Dependent.
* **Correct Answer**: **B**  
* **Explanation**: $P(C \mid E) = 0.12 / 0.30 = 0.40$. Because $P(E) \cdot P(C) = 0.06 \ne 0.12$, browsing electronics doubles purchase likelihood from 20% to 40%, proving the events are statistically dependent. Option B is correct.

---

## Topic 03: Bayes' Theorem

### 1. What the Topic Is
Bayes' Theorem provides a mathematically rigorous mechanism for computing inverse conditional probabilities. It specifies how an initial degree of belief in a hypothesis $H$ (the **Prior Probability**, $P(H)$) should be updated in light of newly observed empirical evidence $E$ (producing the **Posterior Probability**, $P(H \mid E)$), by accounting for the diagnostic probability of the evidence given the hypothesis (the **Likelihood**, $P(E \mid H)$) normalized by the total probability of observing that evidence ($P(E)$).

### 2. Intuition
Suppose a rare disease affects 1 in 1,000 individuals ($0.1\%$). A laboratory diagnostic test boasts $99\%$ accuracy (both sensitivity and specificity). If you test positive, what is the probability that you actually have the disease?
Human intuition erroneously jumps to $99\%$. However, out of 100,000 citizens:
- 100 have the disease, of whom 99 test positive.
- 99,900 do not have the disease, but with $1\%$ false positives, 999 healthy people test positive.
Total positive tests = $99 + 999 = 1,098$. Your probability of actually having the disease given a positive test is $\frac{99}{1,098} \approx 9\%$. 
Bayes' Theorem systematically guards against the **Base Rate Fallacy** by anchoring diagnostic test results to the underlying prevalence of the phenomenon.

### 3. Mathematical Formula
$$P(H \mid E) = \frac{P(E \mid H) \cdot P(H)}{P(E)} = \frac{P(E \mid H) P(H)}{P(E \mid H)P(H) + P(E \mid \neg H)P(\neg H)}$$

---

### 4. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)

#### The "Teller" / Trigger Keywords
- *"False positive rate / test sensitivity / test specificity"*.
- *"Base rate fallacy / rare disease / rare fraud rate"*.
- *"What is the probability that an alert is genuinely positive given that the test fired?"*.
- *"Prior probability updated with new evidence $\to$ Posterior probability"*.

#### Question Archetypes & Answering Strategies
1. **Rare Disease / Fraud Posterior Calculation Archetype**:
   - *Teller Clue*: Given Prior $P(F) = 0.01$, Sensitivity $P(A \mid F) = 0.95$, False Alarm $P(A \mid \neg F) = 0.05$. Find $P(F \mid A)$.
   - *Answering Rule*: Set up Bayes equation:
     $$\text{Numerator} = P(A \mid F)P(F) = 0.95 \times 0.01 = 0.0095$$
     $$\text{Denominator} = 0.0095 + (0.05 \times 0.99) = 0.0095 + 0.0495 = 0.0590$$
     $$\text{Posterior} = \frac{0.0095}{0.0590} \approx 16.1\%$$

---

### 5. Exam-Style MCQs Covering the Possibilities

#### MCQ 1: Base Rate Fallacy in Fraud Detection
**Question**: A wire transfer fraud detection model has a prior probability $P(\text{Fraud}) = 0.01$. If a transfer is fraudulent, the system triggers an alert with sensitivity $P(\text{Alert} \mid \text{Fraud}) = 0.95$. If a transfer is legitimate, the false alarm rate is $P(\text{Alert} \mid \text{Legitimate}) = 0.05$. When an alert fires, what is the exact posterior probability that the transfer is genuinely fraudulent?
- A) $0.9500$
- B) $0.6575$
- C) $0.1610$
- D) $0.0100$

* **Spotting the Clue (The Teller)**: *"Prior $P(F)=0.01$, Sensitivity 0.95, False alarm 0.05 ... probability genuine given alert"*.
* **Answering Strategy**: Bayes' theorem calculation: $\frac{0.95 \times 0.01}{(0.95 \times 0.01) + (0.05 \times 0.99)} = \frac{0.0095}{0.0590} \approx 0.1610$.
* **Correct Answer**: **C**  
* **Explanation**: Because the base rate of fraud is very low (1%), the small 5% false alarm rate acting on the 99% legitimate transactions produces more total false alerts than true alerts. Thus, roughly 84% of alerts are false alarms, yielding a true posterior of $16.1\%$. Option C is correct.

---

## Topic 04: Probability Distributions: Normal, Binomial, Poisson

### 1. What the Topic Is
Probability distributions are mathematical functions that assign probabilities to all possible values taken by a random variable:
- **Binomial Distribution $\mathcal{B}(n, p)$**: Discrete count of successes $k$ in a fixed number $n$ of independent Bernoulli trials with constant probability $p$.
- **Poisson Distribution $\text{Pois}(\lambda)$**: Discrete count of events occurring within a specified continuous interval of time or space, where events occur independently at constant rate $\lambda$.
- **Normal (Gaussian) Distribution $\mathcal{N}(\mu, \sigma^2)$**: Continuous bell-shaped distribution governed by mean $\mu$ and standard deviation $\sigma$.

### 2. Intuition
- **Binomial**: Flipping a coin 20 times and counting heads. Number of trials $n$ is hard-capped (cannot get 21 heads out of 20 flips).
- **Poisson**: Counting cars arriving at a toll booth between 2 PM and 3 PM. No hard cap; arrivals happen continuously at rate $\lambda$.
- **Normal**: Heights of adults or factory part dimensions. The sum of many small independent random perturbations naturally forms a bell curve.

### 3. Mathematical Formula
- **Binomial PMF**: $P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad E[X] = np, \quad \text{Var}(X) = np(1-p)$.
- **Poisson PMF**: $P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad E[X] = \lambda, \quad \text{Var}(X) = \lambda$ (**Equidispersion**).
- **Normal PDF & Standardization**: $Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1)$. Empirical rule: $68\% \in \pm 1\sigma, 95.5\% \in \pm 2\sigma, 99.7\% \in \pm 3\sigma$.

---

### 4. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)

#### The "Teller" / Trigger Keywords
- *"Fixed $n$ independent trials with binary success/failure"* $\to$ **Binomial**.
- *"Count of events per hour / arrivals per day / constant rate $\lambda$"* $\to$ **Poisson**.
- *"Equidispersion property: mean equals variance ($E[X] = \text{Var}(X)$)"* $\to$ **Poisson**.
- *"68-95-99.7 empirical rule / bell curve / standardized Z-score"* $\to$ **Normal**.
- *"Limiting form of Binomial when $n \to \infty$ and $p \to 0$"* $\to$ **Poisson**.

#### Question Archetypes & Answering Strategies
1. **Poisson Interval Scaling Archetype**:
   - *Teller Clue*: Average arrival rate is 3 per hour. What is the probability of 2 arrivals in a 2-hour window?
   - *Answering Rule*: Scale $\lambda$ proportionally to time! $\lambda_{\text{new}} = 3 \times 2 = 6$. Then substitute $k = 2$ into $P(X=2) = \frac{6^2 e^{-6}}{2!} = 18e^{-6}$.
2. **Distribution Identification Archetype**:
   - *Teller Clue*: An empirical dataset has sample mean 4.0 and sample variance 4.1. Which distribution model should be selected?
   - *Answering Rule*: Mean $\approx$ Variance indicates the **Poisson distribution**.

---

### 5. Exam-Style MCQs Covering the Possibilities

#### MCQ 1: Poisson Rate Scaling Calculation
**Question**: An IT helpdesk receives server crash alerts at an average rate of $\lambda_0 = 3$ crashes per hour, following a Poisson process. What is the exact probability that during a randomly chosen 2-hour window, the helpdesk experiences exactly 2 server crashes?
- A) $\frac{3^2 e^{-3}}{2!} \approx 0.2240$
- B) $\frac{6^2 e^{-6}}{2!} = 18 e^{-6} \approx 0.0446$
- C) $\binom{6}{2} (0.5)^6 \approx 0.2344$
- D) $\frac{6^1 e^{-6}}{1!} \approx 0.0148$

* **Spotting the Clue (The Teller)**: *"Rate 3 crashes per hour ... over a 2-hour window ... exactly 2 crashes"*.
* **Answering Strategy**: Scale rate: $\lambda = 3 \times 2 = 6$. Apply Poisson PMF: $P(X = 2) = \frac{6^2 e^{-6}}{2!} = 18 e^{-6} \approx 0.0446$.
* **Correct Answer**: **B**  
* **Explanation**: For a 2-hour interval, the expected count is $\lambda = 3 \times 2 = 6$. Applying the Poisson formula for $k = 2$: $P(X = 2) = \frac{6^2 e^{-6}}{2!} = 18 e^{-6} \approx 0.0446$. Option B is correct.

---

## Topic 05: Central Limit Theorem

### 1. What the Topic Is
The Central Limit Theorem (CLT) establishes that given a population with arbitrary distribution shape (skewed, uniform, bimodal, or discrete), possessing a finite mean $\mu$ and finite non-zero variance $\sigma^2$, the sampling distribution of the standardized sample mean $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$ converges in distribution to a standard Normal distribution $\mathcal{N}(0, 1)$ as the sample size $n$ approaches infinity ($n \ge 30$).

### 2. Intuition
Rolling a single fair die produces a flat, uniform distribution (probabilities 1/6 for 1 to 6). 
However, roll 50 dice and take their average. Getting an average of 1.0 or 6.0 is virtually impossible; the vast majority of averages cluster tightly around 3.5. Repeating this thousands of times forms a smooth Gaussian bell curve. CLT guarantees that **averages behave normally even when individual data points do not**.

### 3. Mathematical Formula
$$\bar{X}_n \xrightarrow{d} \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right) \implies Z = \frac{\bar{X} - \mu}{\frac{\sigma}{\sqrt{n}}} \xrightarrow{d} \mathcal{N}(0, 1)$$
- **Standard Error of the Mean**: $SE(\bar{X}) = \frac{\sigma}{\sqrt{n}}$.
- **Square-Root Law**: To cut $SE$ in half, sample size $n$ must be quadrupled ($4\times$).

---

### 4. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)

#### The "Teller" / Trigger Keywords
- *"Population is non-normal / heavily skewed, but sample size $n \ge 30$"*.
- *"Sampling distribution of the sample mean $\bar{X}$"*.
- *"Standard error $SE = \sigma/\sqrt{n}$"*.
- *"Quadruple sample size to halve the margin of error"*.
- *"CLT breakdown: Cauchy distribution (infinite variance)"*.

#### Question Archetypes & Answering Strategies
1. **Sampling Distribution Shape Archetype**:
   - *Teller Clue*: "A population is severely right-skewed with mean 8 and std dev 3. A sample of $n = 36$ is drawn. What is the shape of the sampling distribution of the mean?"
   - *Answering Rule*: By CLT, since $n = 36 \ge 30$, the distribution of $\bar{X}$ is **approximately Normal** with $SE = 3/\sqrt{36} = 0.5$.
2. **Sample Size Scaling Archetype**:
   - *Teller Clue*: How much must sample size $n$ increase to reduce the standard error from 10 to 5?
   - *Answering Rule*: Halving $SE$ ($1/2$) requires increasing $n$ by $2^2 = 4\times$.

---

### 5. Exam-Style MCQs Covering the Possibilities

#### MCQ 1: Skewed Population with Large Sample Size
**Question**: An e-commerce warehouse processes order weights that are severely right-skewed with population mean $\mu = 8.0\text{ kg}$ and population standard deviation $\sigma = 3.0\text{ kg}$. A quality control inspector draws a random sample of $n = 36$ orders. What is the distribution and standard error of the sample mean $\bar{X}$, and what is $P(\bar{X} > 9.0\text{ kg})$?
- A) Severely right-skewed; $SE = 3.00\text{ kg}$; $P \approx 0.3707$
- B) Approximately Normal; $SE = 0.50\text{ kg}$; $P \approx 0.0228$
- C) Strictly Uniform; $SE = 0.50\text{ kg}$; $P \approx 0.0500$
- D) Approximately Normal; $SE = 0.083\text{ kg}$; $P \approx 0.0001$

* **Spotting the Clue (The Teller)**: *"Severely right-skewed ... sample $n = 36 \ge 30$ ... distribution of sample mean"*.
* **Answering Strategy**: CLT guarantees approximate normality for $n \ge 30$. $SE = \frac{3.0}{\sqrt{36}} = 0.5$. $Z = \frac{9.0 - 8.0}{0.5} = +2.0 \implies P(Z > 2) = 1 - 0.9772 = 0.0228$.
* **Correct Answer**: **B**  
* **Explanation**: By CLT, the sampling distribution of $\bar{X}$ is approximately Normal. $SE = 3.0 / 6 = 0.50\text{ kg}$. $Z = (9.0 - 8.0)/0.50 = 2.0$. $P(Z > 2.0) = 0.0228$. Option B is correct.

---

## Topic 06: Correlation and Covariance

### 1. What the Topic Is
Correlation and covariance quantify the direction, joint variation, and strength of association between two continuous random variables:
- **Covariance $\text{Cov}(X, Y)$**: Measures unstandardized joint variability about their means.
- **Correlation**: Dimensionless standardized covariance bounded strictly between $[-1.0, +1.0]$.

### 2. Intuition
Covariance tells you if two variables climb together (+) or move opposite (-), but its number depends on whether you measure in grams or kilograms. Correlation rescales this to a universal gauge between $-1$ and $+1$.

### 3. Mathematical Formula
- **Population Covariance**: $\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)] = E[XY] - E[X]E[Y]$.
- **Sample Covariance**: $s_{xy} = \frac{1}{n-1}\sum (x_i - \bar{x})(y_i - \bar{y})$.
- **Covariance Matrix $\boldsymbol{\Sigma}$**: Symmetric and positive semi-definite.

---

### 4. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)

#### The "Teller" / Trigger Keywords
- *"Unstandardized joint variability / units are product of X and Y units"*.
- *"Zero covariance does NOT imply statistical independence"*.
- *"Non-linear U-shaped relationship ($Y = X^2$ with symmetric $X$)"*.

#### Question Archetypes & Answering Strategies
1. **Zero Covariance vs Independence Archetype**:
   - *Teller Clue*: Let $X \sim \mathcal{N}(0, 1)$ and $Y = X^2$. What is $\text{Cov}(X, Y)$, and are they independent?
   - *Answering Rule*: $\text{Cov}(X, Y) = E[X^3] - E[X]E[Y] = 0 - 0 = 0$. But $Y$ is completely determined by $X$, so they are **strictly dependent**. Zero covariance only implies independence for bivariate normal distributions!

---

### 5. Exam-Style MCQs Covering the Possibilities

#### MCQ 1: Zero Covariance with Perfect Non-Linear Dependency
**Question**: Let $X$ be a standard normal random variable ($X \sim \mathcal{N}(0, 1)$), and let $Y = X^2$. What is the covariance $\text{Cov}(X, Y)$, and what does this imply about their statistical independence?
- A) $\text{Cov}(X, Y) = 1$; Variables are statistically independent
- B) $\text{Cov}(X, Y) = 0$; Variables are statistically independent
- C) $\text{Cov}(X, Y) = 0$; Variables are strictly dependent
- D) $\text{Cov}(X, Y) = 2$; Variables are strictly dependent

* **Spotting the Clue (The Teller)**: *"Standard normal $X$, $Y = X^2$ ... Covariance and independence"*.
* **Answering Strategy**: Covariance is 0 because $E[X^3] = 0$, but $Y = X^2$ is perfectly deterministic $\implies$ dependent.
* **Correct Answer**: **C**  
* **Explanation**: $E[X] = 0$, $E[XY] = E[X^3] = 0$ due to symmetry. Thus $\text{Cov}(X, Y) = 0$. However, $Y$ is completely determined by $X$, proving that zero covariance does not imply independence. Option C is correct.

---

## Topic 07: Pearson Correlation

### 1. What the Topic Is
The Pearson Product-Moment Correlation Coefficient ($r$ or $\rho$) measures the strength and direction of a strictly linear relationship between two continuous variables on a standardized scale from $-1.0$ to $+1.0$.

### 2. Intuition
Measuring how tightly data points hug a straight line. Pearson's $r$ is blind to curvature: a perfect parabola ($y = x^2$) can have $r \approx 0$ despite having a 100% deterministic relationship.

### 3. Mathematical Formula
$$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}} = \frac{s_{xy}}{s_x s_y}$$
- **$t$-Test Statistic for $H_0: \rho = 0$**: $t = \frac{r \sqrt{n-2}}{\sqrt{1 - r^2}} \sim t_{n-2}$.
- **Coefficient of Determination**: $r^2$ (% of variance explained).

---

### 4. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)

#### The "Teller" / Trigger Keywords
- *"Linear association between two continuous variables"*.
- *"Anscombe’s quartet / same $r = 0.816$ but different non-linear shapes"*.
- *"Testing significance of correlation $H_0: \rho = 0$"*.
- *"Invariance under positive linear transformations ($aX + b$)"*.

#### Question Archetypes & Answering Strategies
1. **Correlation $t$-Statistic Calculation Archetype**:
   - *Teller Clue*: Given $r = 0.60$ and $n = 18$. Calculate the $t$-statistic to test $H_0: \rho = 0$.
   - *Answering Rule*: $t = \frac{r \sqrt{n-2}}{\sqrt{1 - r^2}} = \frac{0.60 \sqrt{16}}{\sqrt{1 - 0.36}} = \frac{0.60 \times 4}{0.80} = \frac{2.40}{0.80} = 3.00$.

---

### 5. Exam-Style MCQs Covering the Possibilities

#### MCQ 1: Pearson Significance $t$-Test
**Question**: An analyst computes a sample correlation of $r = 0.60$ between advertising spend and sales across $n = 18$ retail stores. Testing the null hypothesis $H_0: \rho = 0$ against $H_1: \rho > 0$, what is the calculated $t$-statistic?
- A) $t = 2.00$
- B) $t = 3.00$
- C) $t = 3.46$
- D) $t = 1.34$

* **Spotting the Clue (The Teller)**: *"Sample correlation $r = 0.60$, $n = 18$, testing $H_0: \rho = 0$"*.
* **Answering Strategy**: $t = \frac{r\sqrt{n-2}}{\sqrt{1-r^2}} = \frac{0.60 \times 4}{0.80} = 3.00$.
* **Correct Answer**: **B**  
* **Explanation**: $df = 18 - 2 = 16 \implies \sqrt{16} = 4$. $\sqrt{1 - 0.60^2} = \sqrt{0.64} = 0.80$. $t = (0.60 \times 4)/0.80 = 2.40 / 0.80 = 3.00$. Option B is correct.

---

## Topic 08: Hypothesis Testing

### 1. What the Topic Is
Hypothesis testing (Null Hypothesis Significance Testing - NHST) determines whether sample evidence justifies rejecting a default baseline proposition (Null Hypothesis $H_0$) in favor of a research claim (Alternative Hypothesis $H_1$).

### 2. Intuition
Criminal trial: Defendant is presumed innocent ($H_0$) until proven guilty beyond a reasonable doubt ($\alpha$). 
- **Type I Error ($\alpha$)**: Convicting an innocent person (False Positive).
- **Type II Error ($\beta$)**: Letting a guilty criminal walk free (False Negative).
- **Statistical Power ($1 - \beta$)**: Probability of successfully convicting a truly guilty person.

### 3. Mathematical Formula
- $\alpha = P(\text{Reject } H_0 \mid H_0 \text{ is true})$.
- $\beta = P(\text{Fail to reject } H_0 \mid H_0 \text{ is false})$.
- $\text{Power} = 1 - \beta = P(\text{Reject } H_0 \mid H_1 \text{ is true})$.
- Decision: Reject $H_0$ if $p$-value $\le \alpha$.

---

### 4. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)

#### The "Teller" / Trigger Keywords
- *"Significance level $\alpha$ vs Type I error"*.
- *"Type II error $\beta$ vs Statistical Power ($1 - \beta$)"*.
- *"Decision rule: $p$-value $\le \alpha \implies$ Reject $H_0$"*.
- *"Factors that increase Statistical Power: larger sample size $n$, larger effect size $\delta$"*.

#### Question Archetypes & Answering Strategies
1. **Power & Error Calculation Archetype**:
   - *Teller Clue*: Given significance level $\alpha = 0.01$ and Statistical Power $= 0.85$. What is the probability of a Type II error?
   - *Answering Rule*: $\beta = 1 - \text{Power} = 1 - 0.85 = 0.15$ ($15\%$). Ignore $\alpha$!

---

### 5. Exam-Style MCQs Covering the Possibilities

#### MCQ 1: Calculating Type II Error from Power
**Question**: A clinical trial evaluates a new drug. The researchers set $\alpha = 0.01$ and calculate statistical power $1 - \beta = 0.85$. If the drug is actually effective ($H_0$ is false), what is the exact probability that the researchers commit a Type II error and fail to discover the drug's efficacy?
- A) $0.01$
- B) $0.15$
- C) $0.85$
- D) $0.99$

* **Spotting the Clue (The Teller)**: *"Statistical power $1 - \beta = 0.85$ ... probability of committing Type II error"*.
* **Answering Strategy**: Type II error rate is $\beta = 1 - \text{Power} = 1 - 0.85 = 0.15$.
* **Correct Answer**: **B**  
* **Explanation**: By definition, $\text{Power} = 1 - \beta$. Therefore, $\beta = 1 - 0.85 = 0.15$ (15%). Significance level $\alpha$ governs Type I error, not Type II error. Option B is correct.

---

## Topic 09: Z-Test and Chi-Square Test

### 1. What the Topic Is
- **$Z$-Test**: Parametric test for population means/proportions when population variance $\sigma^2$ is known or $n \ge 30$.
- **Chi-Square ($\chi^2$) Test**: Non-parametric test for categorical independence or goodness-of-fit based on contingency tables.

### 2. Intuition
- $Z$-test measures how many standard deviations a sample mean sits away from the null mean.
- Chi-Square measures how much observed counts ($O$) deviate from expected counts ($E$) under the null of independence.

### 3. Mathematical Formula
- **One-Sample $Z$-Test**: $Z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}$.
- **Chi-Square Statistic**: $\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$.
- **Contingency Expected Frequency**: $E_{ij} = \frac{\text{Row Total}_i \times \text{Col Total}_j}{N}$. Degrees of freedom: $df = (r-1)(c-1)$.

---

### 4. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)

#### The "Teller" / Trigger Keywords
- *"Population standard deviation $\sigma$ is KNOWN / $n \ge 30$"* $\to$ **$Z$-Test**. (If $\sigma$ unknown and $n < 30 \to$ **$t$-Test**).
- *"Contingency table $r \times c$ / test of independence between categorical attributes"* $\to$ **Chi-Square Test**.
- *"Expected cell count formula: $(\text{Row} \times \text{Col})/N$"*.
- *"Degrees of freedom: $(r-1)(c-1)$"*.

#### Question Archetypes & Answering Strategies
1. **Contingency Table Degrees of Freedom & Expected Frequency Archetype**:
   - *Teller Clue*: A $3 \times 2$ table has grand total $N = 600$, Row 1 total $= 200$, Col 1 total $= 300$. Find $df$ and $E_{11}$.
   - *Answering Rule*: $df = (3-1)(2-1) = 2 \times 1 = 2$. $E_{11} = \frac{200 \times 300}{600} = 100$.

---

### 5. Exam-Style MCQs Covering the Possibilities

#### MCQ 1: Degrees of Freedom & Expected Value
**Question**: An A/B test analyzes subscription preferences across 3 customer tiers (Bronze, Silver, Gold) and 2 layout designs (A and B) in a $3 \times 2$ contingency table with grand total $N = 600$. What are the degrees of freedom for the Chi-Square Test of Independence, and what is the expected frequency for Bronze customers who saw Layout A if there are 200 Bronze customers and 300 total users saw Layout A?
- A) $df = 6$; $E = 150$
- B) $df = 2$; $E = 100$
- C) $df = 5$; $E = 100$
- D) $df = 2$; $E = 50$

* **Spotting the Clue (The Teller)**: *"3 customer tiers, 2 layout designs, $N = 600$, 200 Bronze, 300 Layout A"*.
* **Answering Strategy**: $df = (3-1)(2-1) = 2$. $E = (200 \times 300)/600 = 100$.
* **Correct Answer**: **B**  
* **Explanation**: $df = (r-1)(c-1) = (3-1)(2-1) = 2$. Expected frequency $E_{11} = \frac{200 \times 300}{600} = 100$. Option B is correct.

---

## Topic 10: Regression Analysis

### 1. What the Topic Is
Regression analysis models the functional relationship between a continuous dependent variable $Y$ and one or more independent predictors $X_1, \dots, X_p$. Ordinary Least Squares (OLS) minimizes the sum of squared prediction residuals.

### 2. Intuition
Finding the optimal line through a scatter plot where the total squared vertical discrepancies between points and the line are minimized.

### 3. Mathematical Formula
- **Multiple Regression**: $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$.
- **Normal Equations**: $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$.
- **Adjusted $R^2$**: $R^2_{\text{adj}} = 1 - \left[ \frac{(1 - R^2)(n - 1)}{n - p - 1} \right]$.

---

### 4. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)

#### The "Teller" / Trigger Keywords
- *"Normal equations matrix form"*, *"Adjusted R-squared penalizing feature count"*.
- *"Gauss-Markov BLUE theorem (Linearity, Independence, Homoscedasticity, Normality)"*.
- *"Calculation of Adjusted $R^2$ from $R^2, n, p$"*.

#### Question Archetypes & Answering Strategies
1. **Adjusted $R^2$ Calculation Archetype**:
   - *Teller Clue*: Given $n = 31$, $p = 5$, $R^2 = 0.80$. Compute $R^2_{\text{adj}}$.
   - *Answering Rule*: $R^2_{\text{adj}} = 1 - \left[ \frac{(1 - 0.80)(31 - 1)}{31 - 5 - 1} \right] = 1 - \left[ \frac{0.20 \times 30}{25} \right] = 1 - 0.24 = 0.7600$.

---

### 5. Exam-Style MCQs Covering the Possibilities

#### MCQ 1: Adjusted $R^2$ Computation
**Question**: An analyst runs a multiple regression model with $n = 31$ observations and $p = 5$ independent predictors. The model yields $R^2 = 0.80$. What is the calculated Adjusted $R^2$ ($R^2_{\text{adj}}$)?
- A) $0.8000$
- B) $0.7600$
- C) $0.7440$
- D) $0.6500$

* **Spotting the Clue (The Teller)**: *"$n = 31$, $p = 5$, $R^2 = 0.80$ ... calculate Adjusted $R^2$"*.
* **Answering Strategy**: $R^2_{\text{adj}} = 1 - \frac{(1 - R^2)(n-1)}{n - p - 1} = 1 - \frac{0.20 \times 30}{25} = 0.76$.
* **Correct Answer**: **B**  
* **Explanation**: $R^2_{\text{adj}} = 1 - [(0.20 \times 30) / 25] = 1 - [6.0 / 25] = 1 - 0.24 = 0.7600$. Option B is correct.

---

## Topic 11: Predictive Modelling

### 1. What the Topic Is
Predictive modelling learns an approximating function $\hat{f}: \mathcal{X} \to \mathcal{Y}$ from historical training data to accurately forecast unseen future instances, governed by the Bias-Variance tradeoff.

### 2. Intuition
Striking the balance between underfitting (model too simple, high bias) and overfitting (model too complex, high variance).

### 3. Mathematical Formula
$$\text{Expected Test MSE} = \text{Bias}^2(\hat{f}) + \text{Var}(\hat{f}) + \sigma^2_\epsilon$$
Where $\sigma^2_\epsilon$ is irreducible random noise.

---

### 4. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)

#### The "Teller" / Trigger Keywords
- *"Bias-Variance tradeoff"*, *"Irreducible error $\sigma^2_\epsilon$"*.
- *"Increasing polynomial degree / model complexity: Bias drops, Variance increases, Test error forms U-shape"*.

#### Question Archetypes & Answering Strategies
1. **Complexity Effect Archetype**:
   - *Teller Clue*: "What happens to training error, test error, bias, and variance as model complexity increases?"
   - *Answering Rule*: Training error drops monotonically; Bias drops monotonically; Variance increases monotonically; Test error forms a **U-shaped curve**.

---

### 5. Exam-Style MCQs Covering the Possibilities

#### MCQ 1: Bias-Variance Dynamics
**Question**: An engineer transitions from a simple linear model to a 15th-degree polynomial model. What is the guaranteed mathematical effect on training error, model bias, and model variance?
- A) Training error increases; Bias increases; Variance decreases
- B) Training error decreases; Bias decreases; Variance increases; Test error forms a U-shape
- C) Both training error and test error decrease monotonically
- D) Training error decreases; Bias increases; Variance increases

* **Spotting the Clue (The Teller)**: *"Transitions from simple linear to 15th-degree polynomial ... effect on bias and variance"*.
* **Answering Strategy**: Higher complexity = lower bias, higher variance, U-shaped test error.
* **Correct Answer**: **B**  
* **Explanation**: Increasing model complexity fits training points tightly, decreasing training error and bias. However, sensitivity to noise increases (variance increases), causing out-of-sample test error to rise after an optimal point (U-shape). Option B is correct.

---

## Topic 12: Supervised Segmentation and Decision Trees

### 1. What the Topic Is
Supervised segmentation partitions data into homogeneous subsets with respect to a target variable using decision trees (ID3, C4.5, CART).

### 2. Intuition
Playing "20 questions" where each split maximizes subset purity.

### 3. Mathematical Formula
- **Shannon Entropy**: $H(S) = - \sum p_i \log_2 p_i$.
- **Gini Impurity**: $I_G = 1 - \sum p_i^2$.
- **Information Gain**: $IG(S, A) = H(S) - \sum \frac{|S_v|}{|S|} H(S_v)$.

---

### 4. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)

#### The "Teller" / Trigger Keywords
- *"Entropy"*, *"Information Gain"*, *"Gini Impurity"*, *"Pure split"*.
- *"Binary split with equal positive and negative instances ($p = 0.5$)"* $\to$ $H(S) = 1.0\text{ bit}$.
- *"Pure node ($p = 1.0$)"* $\to$ $H(S) = 0.0\text{ bits}$.

#### Question Archetypes & Answering Strategies
1. **Entropy & Gain Calculation Archetype**:
   - *Teller Clue*: Node has 10 positive, 10 negative ($H = 1.0$). Split creates two completely pure leaves. What is $IG$?
   - *Answering Rule*: Children entropy = 0. $IG = H(\text{parent}) - 0 = 1.0 - 0.0 = 1.0\text{ bit}$.

---

### 5. Exam-Style MCQs Covering the Possibilities

#### MCQ 1: Information Gain Calculation
**Question**: A parent node $S$ contains 10 positive instances and 10 negative instances ($p_+ = 0.5, p_- = 0.5$). An attribute $A$ splits $S$ into two subsets: Left child $S_L$ contains 10 positives and 0 negatives; Right child $S_R$ contains 0 positives and 10 negatives. What is the Information Gain $IG(S, A)$?
- A) $1.0\text{ bit}$
- B) $0.5\text{ bit}$
- C) $0.0\text{ bits}$
- D) $2.0\text{ bits}$

* **Spotting the Clue (The Teller)**: *"10 pos, 10 neg split into pure left (10 pos) and pure right (10 neg)"*.
* **Answering Strategy**: Parent entropy = 1.0 bit; Children entropy = 0.0 bits; $IG = 1.0 - 0 = 1.0$ bit.
* **Correct Answer**: **A**  
* **Explanation**: $H(S) = - (0.5 \log_2 0.5 + 0.5 \log_2 0.5) = 1.0\text{ bit}$. Both child nodes are completely pure, so their entropy is 0. $IG = 1.0 - 0 = 1.0\text{ bit}$. Option A is correct.

---

## Topic 13: Overfitting, Generalization and Cross-Validation

### 1. What the Topic Is
Overfitting occurs when a model memorizes sample-specific noise rather than generalizable distribution patterns. Cross-validation estimates out-of-sample generalization error without data leakage.

### 2. Intuition
Memorizing practice exam questions vs learning the underlying subject matter.

### 3. Mathematical Formula
- $CV_{(K)} = \frac{1}{K} \sum_{k=1}^K MSE_k$.
- Tree Cost-Complexity Pruning: $R_\alpha(T) = R(T) + \alpha |T|$.

---

### 4. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)

#### The "Teller" / Trigger Keywords
- *"Data leakage (standardizing before train/test split)"*.
- *"High training accuracy, terrible test accuracy (overfitting)"*.
- *"Cost-complexity pruning parameter $\alpha$"*.

#### Question Archetypes & Answering Strategies
1. **Data Leakage Preprocessing Archetype**:
   - *Teller Clue*: "An analyst performs feature scaling / imputation on the ENTIRE dataset before splitting into train and test sets."
   - *Answering Rule*: This is **Data Leakage**. Test set statistics leak into training, causing artificially optimistic validation metrics that collapse in production.

---

### 5. Exam-Style MCQs Covering the Possibilities

#### MCQ 1: Identifying Data Leakage
**Question**: An analyst performs feature scaling (standardization to zero mean and unit variance) on the entire dataset of 10,000 rows *before* splitting the data into a 70% training set and a 30% testing set. What methodological error has occurred, and what is its operational consequence?
- A) Horizon effect; Model underestimates training error
- B) Data leakage; Test performance will be artificially optimistic and fail to reflect true production generalization
- C) Underfitting; Model variance drops to zero
- D) Multicollinearity; Inversion of $(\mathbf{X}^T\mathbf{X})$ fails

* **Spotting the Clue (The Teller)**: *"Standardization on entire dataset BEFORE splitting into train and test sets"*.
* **Answering Strategy**: Preprocessing across the whole dataset before split = Data Leakage.
* **Correct Answer**: **B**  
* **Explanation**: Test set parameters (mean and standard deviation) leak into the training feature distributions. The test set is no longer an independent hold-out, producing over-optimistic evaluation metrics that degrade on unseen production data. Option B is correct.

---

## Topic 14: Evaluating Classifiers

### 1. What the Topic Is
Evaluating classifiers involves quantitative performance assessment frameworks (Confusion Matrix, Precision, Recall, Specificity, $F_1$, ROC-AUC) under symmetric and asymmetric misclassification costs.

### 2. Intuition
In cancer screening or fraud detection, a False Negative (missing cancer) is catastrophic, while a False Positive (a routine follow-up biopsy) is minor. Classifier evaluation prioritizes Precision and Recall over naive accuracy.

### 3. Mathematical Formula
$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

### 4. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)

#### The "Teller" / Trigger Keywords
- *"Confusion matrix counts: TP, FP, FN, TN"*.
- *"Precision vs Recall vs F1 calculation"*.
- *"Cost-sensitive evaluation / Expected Value Framework"*.

#### Question Archetypes & Answering Strategies
1. **Confusion Matrix Metric Calculation Archetype**:
   - *Teller Clue*: Given $TP = 80, FP = 20, FN = 120, TN = 780$. Compute Precision, Recall, and $F_1$.
   - *Answering Rule*:
     - $\text{Precision} = \frac{80}{80 + 20} = 0.80$ ($80\%$).
     - $\text{Recall} = \frac{80}{80 + 120} = 0.40$ ($40\%$).
     - $F_1 = \frac{2 \times 0.80 \times 0.40}{0.80 + 0.40} = \frac{0.64}{1.20} = 0.5333$.

---

### 5. Exam-Style MCQs Covering the Possibilities

#### MCQ 1: Metric Calculation from Confusion Matrix
**Question**: An anti-money laundering classification model produces the following confusion matrix counts on a test set: $TP = 80$, $FP = 20$, $FN = 120$, $TN = 780$. What are the Precision, Recall, and $F_1$-Score of this classifier?
- A) $\text{Precision} = 0.80$; $\text{Recall} = 0.40$; $F_1 = 0.5333$
- B) $\text{Precision} = 0.40$; $\text{Recall} = 0.80$; $F_1 = 0.5333$
- C) $\text{Precision} = 0.80$; $\text{Recall} = 0.80$; $F_1 = 0.8000$
- D) $\text{Precision} = 0.86$; $\text{Recall} = 0.40$; $F_1 = 0.6000$

* **Spotting the Clue (The Teller)**: *"$TP = 80, FP = 20, FN = 120, TN = 780$ ... calculate Precision, Recall, F1"*.
* **Answering Strategy**: Precision = 80/100 = 0.80. Recall = 80/200 = 0.40. $F_1 = 2(0.8 \times 0.4)/1.2 = 0.5333$.
* **Correct Answer**: **A**  
* **Explanation**: Precision = $80 / (80+20) = 0.80$. Recall = $80 / (80+120) = 0.40$. $F_1 = 2 \times \frac{0.80 \times 0.40}{0.80 + 0.40} = 0.5333$. Option A is correct.

---

## Topic 15: Sampling and Estimation

### 1. What the Topic Is
Sampling and estimation use statistics calculated from a representative sample to estimate unknown population parameters via Point Estimates and Confidence Intervals ($[\bar{x} \pm Z \cdot SE]$).

### 2. Intuition
Tasting a single spoonful of soup to judge the whole pot, provided the soup was thoroughly stirred (randomized).

### 3. Mathematical Formula
- **Confidence Interval for Mean**: $\bar{x} \pm Z_{\alpha/2} \left( \frac{\sigma}{\sqrt{n}} \right)$ or $\bar{x} \pm t_{\alpha/2, n-1} \left( \frac{s}{\sqrt{n}} \right)$.
- **Sample Size for Proportion (Conservative $p = 0.5$)**: $n = \frac{Z_{\alpha/2}^2 \cdot 0.25}{E^2}$.
- **Finite Population Correction (FPC)**: $\sqrt{\frac{N - n}{N - 1}}$ (used when $n/N > 0.05$).

---

### 4. Question Indicators & "Tellers" (How to Spot the Concept in Exam Questions)

#### The "Teller" / Trigger Keywords
- *"Margin of error $E$ / confidence level 95% ($Z = 1.96$)"*.
- *"Sample size calculation for proportion with no prior knowledge ($p = 0.5$)"*.
- *"Finite Population Correction factor"*.

#### Question Archetypes & Answering Strategies
1. **Conservative Sample Size Calculation Archetype**:
   - *Teller Clue*: Margin of error $E = \pm 0.04$, confidence $95\%$ ($Z = 1.96$), no prior knowledge of $p$. What minimum $n$ is required?
   - *Answering Rule*: Conservative $p = 0.50 \implies p(1-p) = 0.25$.
     $$n = \frac{(1.96)^2 \times 0.25}{(0.04)^2} = \frac{3.8416 \times 0.25}{0.0016} = \frac{0.9604}{0.0016} = 600.25 \implies \mathbf{601}$$

---

### 5. Exam-Style MCQs Covering the Possibilities

#### MCQ 1: Sample Size Determination for Polls
**Question**: A polling agency wishes to estimate a mayoral candidate's voting proportion to within an absolute margin of error $E = \pm 0.04$ ($\pm 4\%$) at a $95\%$ confidence level ($Z_{0.025} = 1.96$). Assuming no prior knowledge of the proportion (requiring the most conservative variance assumption $p = 0.50$), what minimum sample size $n$ must be surveyed?
- A) $n = 385$
- B) $n = 601$
- C) $n = 1,067$
- D) $n = 2,401$

* **Spotting the Clue (The Teller)**: *"Margin of error $\pm 0.04$, 95% confidence ($Z = 1.96$), conservative $p = 0.50$ ... minimum $n$"*.
* **Answering Strategy**: $n = \frac{Z^2 \cdot 0.25}{E^2} = \frac{3.8416 \times 0.25}{0.0016} = 600.25 \implies 601$.
* **Correct Answer**: **B**  
* **Explanation**: $n = \frac{(1.96)^2 \times 0.5 \times 0.5}{(0.04)^2} = \frac{0.9604}{0.0016} = 600.25$. Rounding up to the next integer ensures the error threshold is met: $n = 601$. Option B is correct.

---
