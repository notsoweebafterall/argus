# Argus ML

Argus is a multi-agent machine learning system. It takes a dataset, runs it through a full ML pipeline (validation, feature engineering, training, criticism, and deployment), and outputs a model. It is built for the Bank Account Fraud (BAF) dataset but is designed to be extended to other tabular datasets.

## What it does

Argus has five agents that work in sequence. A central orchestrator manages them.
1. Data Agent: Loads the dataset, checks for missing values, duplicates, and class imbalance.
2. Feature Agent: Splits the data by time (not randomly, to avoid leakage) and creates the feature set.
3. ML Agent: Trains models, evaluates them, and picks a winner based on metrics.
4. Critic Agent: Runs 15 deterministic rules to check the data, model, and fairness. If it finds a problem, it rejects the run and tells the orchestrator which stage needs fixing.
5. MLOps Agent: Registers the approved model in MLflow.

## Architecture

The system uses a Model Context Protocol (MCP) layer. Agents do not have direct access to the filesystem or database. They request tools through MCP servers.

- `data_mcp`: Tools for loading and validating data.
- `ml_mcp`: Tools for training and evaluating models.
- `system_mcp`: Tools for file access (locked down for security).
- `retrieval_mcp`: Tools for retrieving past failures (RAG).

The Critic Agent uses RAG. When it rejects a model, it searches a local index of past failures to see if a similar issue occurred before. This helps it explain why it rejected the model.

## Setup

1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt` (or `pip install -e ".[dev]"`).
3. Add the BAF dataset files to `data/` (Base.csv, Variant I.csv, etc.). You can get them from Kaggle.
4. Build the retrieval index (required for RAG): `python -m tools.retrieval_tools build-all`
5. Run the app: `streamlit run app.py`

## Project Structure

```text
Argus/
├── app.py                  # Streamlit user interface
├── app_handlers.py         # Demo logic for the UI
├── agents/                 # The five agents and base classes
├── pipeline/               # Orchestrator and task graph
├── mcp_servers/            # MCP tool servers (data, ml, system, retrieval)
├── tools/                  # Actual tool implementations
├── serving/                # FastAPI serving layer
├── monitoring/             # Drift and performance monitoring
├── eval/                   # Adversarial test scenarios
├── tests/                  # Unit and integration tests
├── docs/                   # Architecture and design notes
└── data/                   # Datasets and uploads (ignored by git)
```

## Limitations

-The pipeline currently only runs on the BAF dataset. It requires specific columns (fraud_bool and month). Uploading a generic CSV will preview the data, but the pipeline will not execute because the feature engineering and critic rules are hardcoded for BAF.
-The live LLM adapter (Groq) is slow and rate-limited on the free tier. The default mode uses a deterministic mock handler so the UI works reliably without API costs or timeouts.
-The retrieval index must be rebuilt after new runs (python -m tools.retrieval_tools build-all) if you want the RAG to include the latest failures. It does not update automatically.

## V2 Future Expansion

- The next version (V2) focuses on generalizing the system so it works with any tabular dataset.
- Generic CSV Ingestion: We will add a UI step to confirm the target column and the time column. The Data Agent will be updated to accept a file path and profile any dataset, instead of relying on hardcoded BAF variants.
- Configurable Feature Engineering: The hardcoded month and fraud_bool logic in create_features will be replaced with parameters. The temporal split will be optional if the dataset has no time column.
- Conditional Critic Rules: The 15 deterministic rules will be updated to check if the required columns exist. For example, the temporal split check will only run if the user confirmed a time column. This prevents non-BAF datasets from failing validation unfairly.
- Automated Project History: We will add a database layer to save projects and run history so users can reopen past experiments without relying on the MLflow UI.
- Advanced Explainability: We will integrate SHAP to provide local and global feature importance charts directly in the UI.
