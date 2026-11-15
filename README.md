# Argus — Autonomous ML Engineering Platform

Argus is a multi-agent system that takes a fraud detection project from raw data to a deployed, monitored model, without a human running each step by hand.

It works on the Bank Account Fraud (BAF) dataset, a large synthetic banking dataset with realistic problems baked in: class imbalance, time-based structure, and known fairness issues across age groups. Those problems give the system real things to catch, not just numbers to report.

## What it actually does

You point Argus at a dataset variant and it runs through five stages on its own:

```
Data Agent → Feature Agent → ML Agent → Critic → MLOps Agent
```

**Data Agent** loads the data and checks it. Missing values, weird encodings, duplicate rows, class balance, whether the time column makes sense.

**Feature Agent** builds the feature set. It splits the data by month instead of randomly, because this is time series data and a random split would leak future information into training. It also drops features that are too correlated with each other.

**ML Agent** trains several models (logistic regression, random forest, XGBoost, and a small neural net), tunes hyperparameters on the tree-based ones, picks a winner, and writes up why that model won using the actual numbers.

**Critic** is the part I'm most proud of. It's not another AI call making a judgment call. It's a set of hard-coded rules checking the data, the methodology, the results, and fairness (using the same age-based fairness metric the BAF dataset was built to test). If a model fails a check, the Critic sends it back to whichever stage can actually fix the problem, not just back to square one.

**MLOps Agent** registers the approved model in MLflow. From there a FastAPI service can serve predictions, and a monitoring module keeps an eye on things like input drift and performance drop-off after deployment.

## How it's built

Each agent only has access to the tools it needs for its job. There's an MCP layer underneath that exposes those tools, split across three servers, with the one that touches the filesystem locked down the tightest since it's the riskiest.

There's also a test suite that tries to break the whole thing on purpose. Missing files, bad inputs, a critic that keeps rejecting everything, prompt injection attempts. The point isn't that Argus never fails. It's that when it fails, it fails the right way and doesn't just crash or produce garbage silently.

## Stack

Python, PyTorch, scikit-learn, XGBoost, MLflow, FastAPI, Docker, SQLite, MCP, Antigravity.

## Project layout

```
argus-ml/
├── agents/        # the five agents and the shared runtime
├── tools/         # the actual tool implementations
├── mcp_servers/   # Data, ML, and System MCP servers
├── skills/        # small reusable manifests over MCP tools
├── pipeline/      # orchestrator, task graph, config
├── serving/       # the FastAPI app and Dockerfile
├── monitoring/    # production health checks
├── eval/          # the adversarial test suite
├── tests/         # regular unit and integration tests
└── docs/          # architecture notes and design writeup
```

## Running it

```
pytest tests/ -v
pytest eval/ -v
```

To see the model registry and past runs:
```
mlflow ui
```

To hit the serving API directly:
```
uvicorn serving.api:app
```
then open `/docs` for a live interface to try predictions.

## Why this is more than a fraud model

Training a fraud classifier isn't the hard part, plenty of tutorials do that. The interesting part is everything wrapped around it: agents that decide what to do next instead of following a fixed script, a critic that can actually reject bad work and explain exactly why with real numbers, and an orchestrator that routes problems back to whoever can fix them instead of just retrying blindly. That's the part worth talking about.
