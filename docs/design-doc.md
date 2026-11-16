# Argus ML - Design Document

## 1. Problem Definition and BAF Dataset

Argus addresses the challenge of building an autonomous, end-to-end machine learning engineering workflow. It transitions from raw tabular data to an audited, registered, and monitored production fraud detection service.

Standard fraud detection benchmarks (e.g., Credit Card Fraud) often present simplified problems: they lack temporal progression, contain pre-transformed PCA components, and omit demographic features necessary for fairness analysis. Argus targets the **Bank Account Fraud (BAF)** dataset suite (introduced at NeurIPS 2022 by Feedzai Research). 

### BAF Key Characteristics
- **Dataset Variants**: Six variants are supported: `Base`, `Variant I` to `Variant V`.
- **Scale and Class Imbalance**: ~1,000,000 rows per variant. Positive minority fraud rate of ~1.1% (`fraud_bool == 1`).
- **Temporal Structure**: An explicit `month` column (values `0` to `7`).
- **Fairness Benchmarking**: Explicit customer demographic age groups (`customer_age`) enabling parity evaluation between younger applicants (`< 50`) and older applicants (`>= 50`).

## 2. Data Validation and Preprocessing

Data validation is performed autonomously by `DataAgent`. 

### Deterministic Sentinel Handling
Per the BAF published specification, missing numeric values are encoded using sentinel values (`-1`), not standard IEEE NaNs. `validate_dataset` inspects all numeric columns for `-1` sentinel encodings. If any column exhibits sentinel `-1` counts exceeding 10% of total rows, it appends a mandatory `sentinel_missing` alert.

### Deterministic Severity Classification
The `validate_dataset` tool evaluates data health:
- `critical`: Triggered if `month` values fall outside `[0, 7]`, sentinel ratio exceeds 50%, or duplicate rows exceed 10%.
- `warning`: Triggered if sentinel ratio exceeds 10% or minority class ratio drops below 1.0%.
- `ok`: Passed all checks.
If severity is `critical`, `DataAgent` immediately returns status `failure`.

## 3. Temporal Splitting and Leakage Prevention

Argus enforces fixed temporal splitting to prevent future-leakage:
- `TRAIN_MONTHS = range(0, 6)`
- `EVAL_MONTHS = range(6, 8)`

### Multi-Layer Leakage Defense
1. **Tool-Level**: `create_features` rejects `fraud_bool` and `month`.
2. **Agent-Level**: `FeatureAgent` scans the manifest and fails if `month` is present.
3. **Critic-Level**: Deterministic rules reject the run and route back to `feature_engineering`.

## 4. Feature Engineering

- **Multicollinearity Reduction**: Pairs with $|r| > 0.90$ are flagged. The second feature is dropped.
- **Preprocessing**: Missing values are imputed using training medians. Baseline distributions are persisted for production drift detection.

## 5. Model Choices and Training Methodology

Argus evaluates four model families:
1. **Logistic Regression**: Baseline linear model.
2. **Random Forest**: Bagged ensemble of decision trees.
3. **XGBoost**: Gradient boosted trees with `scale_pos_weight`.
4. **PyTorch Tabular MLP**: Deep learning baseline.

**Hyperparameter Search**: Runs on a holdout slice strictly within the training months. The eval set is never touched.

## 6. Evaluation Metrics

Argus enforces fraud-domain evaluation metrics:
- **PR-AUC**: Precision-Recall Area Under Curve.
- **Recall at Fixed 5% FPR**: Proportion of fraud caught when false alarms are capped at 5%.
- **Grounded Selection Rationale**: `MLAgent` must cite verified numeric metrics. `CriticAgent` verifies that cited numbers match actual outputs.

## 7. Fairness Evaluation

- **Metric**: Predictive Equality Ratio (parity of False Positive Rates between younger and older applicants).
- **Threshold**: Must be $\ge 0.70$.

## 8. Deterministic Critic Rules

The `CriticAgent` audits the pipeline using 15 deterministic checks across 4 categories:
1. **Data**: Severity, sentinel threshold, target correlation, minority ratio.
2. **Modeling**: Temporal split, no `month` in features, no `fraud_bool` in features, min candidates, generalization gap, PR-AUC bounds.
3. **Experiment**: Selection reason length, metric grounding.
4. **Fairness**: Predictive equality, non-empty groups.

## 9. Category-Aware Rejection Routing

When Critic checks fail, the Orchestrator executes `_determine_critic_retry_target`:
1. **Data failures** route to `data_validation`.
2. **Leakage failures** route to `feature_engineering`.
3. **Modeling/Fairness failures** route to `model_training`.

## 10. RAG (Retrieval-Augmented Generation)

To provide explainable audit trails, Argus implements RAG:
- **Corpus**: Rejected runs log their `failed_checks`. `tools/retrieval_tools.py` extracts these into a JSONL corpus.
- **Embedding**: Uses `sentence-transformers` to embed failure descriptions into a local NumPy cosine index.
- **Inference**: When `CriticAgent` rejects a model, it calls `retrieval_mcp` to fetch similar past failures, citing them in its rationale.

## 11. MLflow, FastAPI, and Streamlit

- **MLflow**: Tracks parameters, metrics, and `critic_verdict`. Only approved models are registered.
- **FastAPI Serving**: Loads the Staging model. Endpoints: `/health`, `/model-info`, `/predict`.
- **Streamlit UI**: Allows users to upload a BAF CSV, run the pipeline, view RAG citations, and download an HTML report.

## 12. Production Monitoring

The `monitoring/monitor.py` script tracks:
- **Latency**: p95 latency spikes.
- **Prediction Drift**: Real-time fraud rate deviating from training baseline.
- **Feature PSI**: Population Stability Index for input features.
- **Delayed-Label Degradation**: Drop in PR-AUC when ground truth arrives.

## 13. Adversarial Evaluation

A suite of 10 scenarios tests failure modes: missing data, corrupted CSV, target leakage, malformed tools, training failure, retry exhaustion, fairness violation, unavailable model, drifted input, and duplicate registration.

## 14. Known Limitations
1. Timeout does not preempt in-flight calls.
2. Sandbox is best-effort, not kernel-level.
3. MLflow stages are deprecated.
4. Retrieval index requires manual rebuild.

## 15. V2 Future Improvements
- **Generic Datasets**: Configurable target and time columns.
- **Conditional Critic**: Rules skip if columns are absent.
- **SHAP Integration**: For feature importance.