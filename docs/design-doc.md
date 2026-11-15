# Argus ML — Design Document

## 1. Problem Definition and BAF Dataset

Argus addresses the challenge of building an autonomous, end-to-end machine learning engineering workflow that transitions from raw tabular data to an audited, registered, and monitored production fraud detection service.

Standard fraud detection benchmarks (e.g., Credit Card Fraud Detection) often present simplified problems: they lack temporal progression, contain pre-transformed PCA components, and omit demographic features necessary for fairness analysis. Argus targets the **Bank Account Fraud (BAF)** dataset suite (introduced at NeurIPS 2022 by Feedzai Research: *"Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation"*).

### BAF Key Characteristics
- **Dataset Variants**: Six variants are supported: `Base` (baseline synthetic data), `Variant I` (strong fraud performance drop), `Variant II` (increased demographic bias), `Variant III` (temporal distribution drift), `Variant IV` (combined bias and performance degradation), and `Variant V` (severe concept drift).
- **Scale and Class Imbalance**: ~1,000,000 application rows per variant, exhibiting severe class imbalance with a positive minority fraud rate of ~1.1% (`fraud_bool == 1`).
- **Temporal Structure**: An explicit `month` column (values `0` to `7`) reflecting sequential time periods.
- **Fairness Benchmarking**: Explicit customer demographic age groups (`customer_age`) enabling parity evaluation between younger applicants (`< 50`) and older applicants (`>= 50`).

---

## 2. Data Validation and Preprocessing

Data validation is performed autonomously by `DataAgent` (`agents/data_agent.py`) utilizing `tools/data_tools.py`.

### Deterministic Sentinel Handling
Per the BAF published specification, missing numeric values are encoded using sentinel values (`-1`), **not** standard IEEE NaNs. Standard data science pipelines relying solely on `df.isnull()` fail to identify these unobserved values:
- `validate_dataset` inspects all numeric columns for `-1` sentinel encodings and reports exact column counts.
- `DataAgent` post-processes validation results via `_enforce_deterministic_issues`: if any column exhibits sentinel `-1` counts exceeding 10% of total rows, it appends a mandatory `sentinel_missing` alert.

### Deterministic Severity Classification
The `validate_dataset` tool evaluates data health deterministically:
- `critical`: Triggered if `month` values fall outside expected bounds `[0, 7]`, maximum column sentinel ratio exceeds 50%, or duplicate row ratio exceeds 10%.
- `warning`: Triggered if maximum sentinel ratio exceeds 10% or minority class ratio drops below 1.0%.
- `ok`: Passed all checks.

If severity is `critical`, `DataAgent` immediately returns status `failure`, halting execution before corrupted data propagates downstream.

---

## 3. Temporal Splitting and Leakage Prevention

### Fixed Temporal Boundaries
Financial fraud exhibits non-stationary behavior and temporal drift. Standard random k-fold or shuffle-based train/test splits introduce severe future-leakage, artificially inflating evaluation scores.

Argus enforces fixed temporal splitting constants in `tools/ml_tools.py`:
- `TRAIN_MONTHS = range(0, 6)`: Months `0, 1, 2, 3, 4, 5` (~800,000 rows).
- `EVAL_MONTHS = range(6, 8)`: Months `6, 7` (~200,000 rows).

### Multi-Layer Leakage Defense
The temporal split key (`month`) and target variable (`fraud_bool`) must never enter the model feature space:
1. **Tool-Level Defense (`create_features`)**: Pre-execution validation verifies that `fraud_bool` and `month` are absent from `feature_spec.features`. If present, it raises `ValueError("leakage_detected: ...")`.
2. **Agent-Level Defense (`FeatureAgent._enforce_month_absence`)**: A code-level post-processing check scans `output.artifact["manifest"]["feature_columns"]`. If `month` is detected, the agent overrides status to `failure`.
3. **Critic-Level Gate (`CriticAgent`)**: Deterministic rules `month_not_in_feature_columns` and `fraud_bool_not_in_feature_columns` audit the manifest. If detected, the Critic rejects the run and routes execution back to `feature_engineering`.

---

## 4. Feature Engineering

Executed by `FeatureAgent` (`agents/feature_agent.py`) using `create_features` in `tools/ml_tools.py`.

### Multicollinearity Reduction
During exploratory data analysis (`run_eda`), pairwise Pearson correlations across all numeric features are computed. Feature pairs with absolute correlation $|r| > 0.90$ are flagged in `high_correlation_pairs`.
- `FeatureAgent` extracts these pairs from `data_validation_artifact["eda"]` and passes them to `create_features` via `feature_spec.drop_correlated_pairs`.
- For each collinear pair, the second feature is dropped, preventing multicollinearity issues in linear and neural models while reducing feature dimensionality.
- All dropped features and corresponding reasons are permanently recorded in the feature manifest (`{feature_id}_manifest.json`).

### Preprocessing and Distribution Baselines
Missing values are imputed using column medians computed from training data. Concurrently, baseline feature distributions (quantiles, deciles, mean, std) and the baseline training fraud rate are persisted to `experiments/features/{feature_id}_distribution.json` to serve as the ground-truth reference for production drift detection.

---

## 5. Model Choices and Training Methodology

Managed by `MLAgent` (`agents/ml_agent.py`) using `tools/ml_tools.py`. Argus evaluates four model families representing diverse inductive biases:

1. **Logistic Regression (`logistic_regression`)**: Baseline linear model fitted with `StandardScaler` and `max_iter=1000`. Fast, interpretable reference.
2. **Random Forest (`random_forest`)**: Bagged ensemble of decision trees (`n_estimators=100`, `random_state=42`) robust to unscaled features and non-linearities.
3. **XGBoost (`xgboost`)**: Gradient boosted decision trees using `XGBClassifier`. Imbalance is addressed directly via `scale_pos_weight = (n_negatives / n_positives)`.
4. **PyTorch Tabular MLP (`pytorch_baseline`)**: Deep learning baseline (`PyTorchTabularMLP`) featuring linear layers, ReLU activations, dropout regularization, and `BCEWithLogitsLoss` parameterized with positive class weight scaling.

### Leakage-Free Hyperparameter Search (`search_hyperparams`)
Hyperparameter optimization for tree-based models (`xgboost`, `random_forest`) uses a holdout validation slice partitioned **strictly within the training months** (e.g., months 0–4 for fitting, month 5 for holdout evaluation). The evaluation set (months 6–7) is never touched during tuning. Once the optimal hyperparameter combination is selected based on holdout PR-AUC, the final winning model is trained on the full training set (months 0–5).

### Unified Bundle Storage
Models and fitted scalers are serialized together as a single atomic dictionary bundle (`bundle = {"model": model, "scaler": scaler, "metadata": metadata}`) into `experiments/models/{model_id}.joblib`. This guarantees that evaluation and inference stages transform features using the exact scaler fitted during training, preventing train-test contamination or missing transformer states.

---

## 6. Evaluation Metrics and Model Selection

In highly imbalanced fraud detection (~1.1% fraud rate), accuracy and ROC-AUC are misleading: a trivial classifier predicting all negatives achieves 98.9% accuracy, and ROC-AUC can remain artificially inflated due to the dominant true negative volume.

Argus enforces fraud-domain evaluation metrics in `evaluate_model`:
- **PR-AUC (Precision-Recall Area Under Curve)**: Primary optimization metric reflecting the tradeoff between precision and recall across decision thresholds.
- **Recall at Fixed 5% FPR (`recall_at_fixed_fpr`)**: Measures the proportion of fraudulent applications caught when false alarms are held to an operationally acceptable ceiling of 5% (FPR $\le 0.05$).
- **Grounded Selection Rationale**: `MLAgent` must select the winning candidate by citing verified numeric evaluation metrics in `selection_reason`. `CriticAgent` inspects this reason via regex and verifies that cited numbers match actual candidate evaluation outputs within $\pm 0.015$ tolerance.

---

## 7. Fairness Evaluation

Argus implements demographic fairness auditing based on the BAF benchmark definition (`tools/ml_tools.py::check_fairness`):
- **Protected Attribute**: `customer_age`.
- **Demographic Slices**: Younger applicants (`customer_age < 50`) vs. Older applicants (`customer_age >= 50`).
- **Metric**: **Predictive Equality Ratio**, evaluating parity of False Positive Rates (FPR) across groups:
  $$\text{Predictive Equality Ratio} = \frac{\min(\text{FPR}_{\text{younger}}, \text{FPR}_{\text{older}})}{\max(\text{FPR}_{\text{younger}}, \text{FPR}_{\text{older}})}$$
- **Edge Cases**:
  - If both group FPRs are 0.0: ratio is defined as `1.0` (zero disparate harm).
  - If one group FPR is 0.0 and the other non-zero: ratio is defined as `0.0` (maximal disparity).
- **Threshold**: The Critic requires $\text{Predictive Equality Ratio} \ge 0.70$. Models disproportionately rejecting non-fraudulent applicants in one age group fail audit and cannot be published.

---

## 8. Deterministic Critic Rules

The `CriticAgent` (`agents/critic_agent.py`) audits the pipeline using 15 deterministic checks across 4 categories:

| Category | Check Name | Deterministic Threshold / Rule | Failure Implication |
| :--- | :--- | :--- | :--- |
| **Data** | `data_severity_not_critical` | `severity != "critical"` | Corrupted dataset or malformed schema |
| **Data** | `sentinel_missing_threshold` | Sentinel $-1 \le 30.0\%$ for all columns | Unusable column with excessive missing data |
| **Data** | `no_extreme_target_correlation` | $\lvert\text{corr}(\text{feature}, \text{fraud\_bool})\rvert \le 0.95$ | Probable target leakage in feature |
| **Data** | `minority_class_ratio` | $0.001 \le \text{minority\_ratio} \le 0.5$ | Extreme class imbalance anomaly |
| **Modeling** | `temporal_split_integrity` | `split_method == "temporal"`, train months `0-5`, eval months `6-7` | Invalid split strategy / future contamination |
| **Modeling** | `month_not_in_feature_columns` | `'month'` absent from `feature_columns` | Split-key temporal leakage |
| **Modeling** | `fraud_bool_not_in_feature_columns`| `'fraud_bool'` absent from `feature_columns` | Target label leakage |
| **Modeling** | `candidates_considered_minimum` | $\ge 2$ candidate models evaluated | Insufficient model comparison |
| **Modeling** | `generalization_gap_threshold` | $\lvert\text{train\_pr\_auc} - \text{eval\_pr\_auc}\rvert \le 0.15$ | Severe model overfitting |
| **Modeling** | `eval_pr_auc_plausibility_upper`| $\text{eval\_pr\_auc} \le 0.999$ | Suspiciously perfect performance / leakage |
| **Modeling** | `eval_pr_auc_plausibility_lower`| $\text{eval\_pr\_auc} > 0.05$ | Degraded model performing below chance |
| **Experiment**| `selection_reason_minimum_length`| $\ge 20$ characters | Empty or vacuous selection narrative |
| **Experiment**| `selection_reason_metric_grounding`| Cited numbers in selection reason match eval metrics within $\pm 0.015$ | Unsubstantiated or hallucinated claim |
| **Fairness** | `predictive_equality_ratio_threshold`| $\text{Predictive Equality Ratio} \ge 0.70$ | Demographic age disparity violation |
| **Fairness** | `fairness_groups_non_empty` | Group sizes `younger > 0` and `older > 0` | Empty demographic slice in eval set |

---

## 9. Category-Aware Rejection Routing

When Critic checks fail, the Orchestrator avoids blind retries by executing `_determine_critic_retry_target`:
1. **Priority 1 (`category == "data"`)**: Routes to `data_validation` to reload or clean raw data.
2. **Priority 2 (`category == "modeling"` and check in `["month_not_in_feature_columns", "fraud_bool_not_in_feature_columns"]`)**: Routes to `feature_engineering` to regenerate features without leaked keys.
3. **Priority 3 (Default Fallback)**: Modeling performance gaps, metric grounding errors, or fairness failures route to `model_training` to train alternative architectures or adjust hyperparameters.

Retry budgets (`MAX_RETRIES_PER_STAGE = 3`) prevent infinite looping. Context is enriched with the specific `failed_checks` on each retry.

---

## 10. MLflow Experiment Tracking and Model Registration

Experiment tracking and model governance are managed via MLflow (`experiments/mlflow.db`):
- **Parameters Logged**: `model_id`, `model_type`, `feature_set_path`, `dataset_variant`, `split_method`, `train_months`, `eval_months`, `code_version` (git commit hash), and all model hyperparameters.
- **Metrics Logged**: `train_accuracy`, `train_pr_auc`, `eval_precision`, `eval_recall`, `eval_pr_auc`, `eval_recall_at_fixed_fpr`, and `eval_predictive_equality_ratio`.
- **Critic Verdict Tagging**: In `pipeline/orchestrator.py`, the active MLflow run is dynamically tagged with `critic_verdict: "approve"` or `"reject"`.
- **Gated Registration**: `MLOpsAgent` verifies Critic approval (`critic_verdict == "approve"`). Only approved models are registered to the MLflow Model Registry (`argus_fraud_detector`) and transitioned to stage `Staging`.

---

## 11. FastAPI Serving

Production serving is provided by `serving/api.py`:
- **Lifespan Startup**: Automatically retrieves the latest model version in stage `Staging` from the MLflow Model Registry, loads the unified bundle (`model` + fitted `scaler`), and extracts expected feature columns. If no model is registered, the application starts gracefully in an unloaded state.
- **Endpoints**:
  - `GET /health`: Returns service health status, whether a model is loaded, and active model version.
  - `GET /model-info`: Returns loaded model type, registration timestamp, and evaluation metrics (503 if unloaded).
  - `POST /predict`: Real-time inference. Enforces strict input validation: incoming feature keys must match expected columns exactly (returns 422 with missing and unexpected keys on mismatch). Applies fitted scaler, computes class probabilities, measures wall-clock inference latency, and logs audit events to `monitoring/prediction_log.jsonl`. Returns 503 if no model is loaded.

---

## 12. Production Monitoring and Drift Detection

The monitoring subsystem (`monitoring/monitor.py`) reads `prediction_log.jsonl`, training distributions, and evaluation slices to evaluate 5 production reliability signals:
1. **Latency Spike**: Flags if 95th percentile latency exceeds 500ms (`latency_p95 > 500.0ms`).
2. **Prediction Distribution Drift**: Flags if real-time predicted fraud rate deviates by more than $3\times$ from baseline training fraud rate ($< \text{base}/3$ or $> 3 \times \text{base}$).
3. **Input Feature Drift (PSI)**: Calculates Population Stability Index (PSI) per numeric feature against training baseline deciles. Features with $\text{PSI} > 0.20$ are flagged for significant distribution shift.
4. **Delayed-Label Performance Degradation**: Evaluates candidate models on delayed ground-truth batches (e.g. month 7 vs month 6). Flags if PR-AUC drops by $> 0.10$ from baseline.
5. **Inference Error Rate**: Flags if API inference errors exceed 1.0% of total requests.

---

## 13. Adversarial Evaluation Harness

Located in `eval/agent_eval/`, the adversarial evaluation suite stress-tests the multi-agent system across 10 failure-mode scenarios scored against dual rubrics (Rubric A: Model Performance; Rubric B: System Reliability):

1. **Scenario 1 (Missing dataset variant)**: Verifies clean `FileNotFoundError` propagation, `DataAgent` status `failure`, and Orchestrator termination in status `failed` without hallucinated artifacts.
2. **Scenario 2 (Corrupted/truncated CSV)**: Verifies rejection with explicit schema error.
3. **Scenario 3 (Extreme target leakage injected)**: Verifies Critic catches $|r| = 0.998$ correlation and rejects run.
4. **Scenario 4 (Malformed tool call arguments)**: Verifies tool argument errors are captured in `ToolCallLog` without unhandled crashes.
5. **Scenario 5 (Model training failure)**: Verifies unsupported model types and invalid hyperparams raise explicit typed errors.
6. **Scenario 6 (Critic retry-loop exhaustion)**: Verifies Orchestrator bounds execution and halts at `MAX_RETRIES_PER_STAGE = 3` without infinite looping.
7. **Scenario 7 (Fairness violation on skewed variant)**: Verifies demographic sanity checks on real skewed BAF variants.
8. **Scenario 8 (Prediction request with unavailable model)**: Verifies FastAPI serving returns graceful HTTP 503 rather than unhandled 500 errors.
9. **Scenario 9 (Drifted input at serving time)**: Verifies feature PSI signals flag shifted distributions ($\text{PSI} > 0.20$).
10. **Scenario 10 (Duplicate model registration)**: Verifies model registry handles duplicate versioning cleanly and idempotently.

---

## 14. Failure Handling and Known Limitations

1. **Stage Loop Timeout vs. Synchronous Execution**: The wall-clock timeout guard (default: 1800s) is evaluated at stage loop boundaries. It bounds total duration across retries and stage transitions, but does not asynchronously preempt an in-flight synchronous call (e.g., long-running model training) within an individual stage.
2. **Subprocess Sandbox Boundaries**: `tools/system_tools.py::run_python` enforces AST checks, timeout limits, and import blocking on network packages. It provides best-effort sandbox isolation suitable for automated ML scripting, but does not claim kernel-level container virtualization or defense against low-level C extension escapes (`ctypes`).
3. **MLflow Legacy Stage API**: Model registry transitions utilize MLflow's stage-based API (`Staging`), which is marked for future deprecation upstream in favor of model version aliases.

---

## 15. Future Improvements

- **Model Version Aliasing**: Migrate from legacy MLflow stages (`Staging`/`Production`) to modern MLflow model version tags and aliases (`@champion`, `@challenger`).
- **Dynamic Asynchronous Preemption**: Integrate asynchronous worker pools with thread/process cancellation to enable hard preemption of in-flight stage computations.
- **Automated Production Canary Promotion**: Implement canary serving and automated rollback based on live monitoring signals and delayed-label performance metrics.

