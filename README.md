# Churn Prediction Agent

An agentic churn prediction system built with LangGraph, scikit-learn, and Gradio. Upload new customer data to predict churn, or ask about a specific customer from the historical dataset.

## How it works

- **Predict** — upload a CSV of customer transactions and get churn probabilities + expected lifetime for each customer
- **Lookup** — ask about a specific customer ID and get a summary of their purchase behaviour
  <img width="1913" height="953" alt="image" src="https://github.com/user-attachments/assets/dafa8e5c-8b00-4af8-b607-f5124410bdf0" />


The agent routes your query automatically using an LLM, runs the appropriate ML model (Random Forest + Weibull AFT survival model), then returns a plain-English explanation.

## Setup

**1. Clone the repo**
```bash
git clone <repo-url>
cd UCI
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

**3. Set your HuggingFace token**

Create a `.env` file in the project root:
```
HF_TOKEN=hf_your_token_here
```
Get your token at https://huggingface.co/settings/tokens (needs Inference API access).

**4. Train the models**
```bash
python -m models.train
```
This reads `dataset/online_retail_II.csv` and saves trained artifacts to `models/artifacts/`.

> Download the dataset from: https://archive.ics.uci.edu/dataset/502/online+retail+ii
> Place it at `dataset/online_retail_II.csv`

## Run locally

```bash
python app.py
```


## Project structure

```
UCI/
├── agent/
│   ├── graph.py       # LangGraph pipeline
│   ├── nodes.py       # router, predict, lookup, explain nodes
│   └── state.py       # AgentState TypedDict
├── models/
│   ├── train.py       # trains RF + Weibull AFT, saves artifacts
│   ├── predict.py     # loads artifacts, runs predictions
│   └── artifacts/     # saved model files (not in git)
├── dataset/           # place online_retail_II.csv here
├── app.py             # Gradio UI
├── requirements.txt
└── .env               # HF_TOKEN (not in git)
```

## Tech stack

- **ML:** scikit-learn (Random Forest), lifelines (Weibull AFT), pandas
- **Agent:** LangGraph, LangChain
- **LLM:** `meta-llama/Meta-Llama-3.1-8B-Instruct` via HuggingFace Inference API (Novita provider)
- **UI:** Gradio
  
