import os
import re
import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from agent.state import AgentState
from models.predict import predict_new_data, lookup_customer

load_dotenv()

_chain = None

def _get_chain():
    global _chain
    if _chain is None:
        endpoint = HuggingFaceEndpoint(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            huggingfacehub_api_token=os.getenv("HF_TOKEN"),
            provider="novita",
            max_new_tokens=256,
            temperature=0.4,
        )
        _chain = ChatHuggingFace(llm=endpoint)
    return _chain


def router_node(state: AgentState) -> AgentState:
    query = state["query"]
    messages = [
        ("system", "You are a router. Reply with only one word: 'predict' if the user wants churn prediction on new data, or 'lookup' if they are asking about a specific customer."),
        ("human", query)
    ]
    response = _get_chain().invoke(messages).content.strip().lower()
    route = "predict" if "predict" in response else "lookup"
    return {**state, "route": route}


def predict_node(state: AgentState) -> AgentState:
    csv_path = state.get("csv_path")
    if not csv_path:
        return {**state, "error": "No CSV file provided for prediction."}
    try:
        df = pd.read_csv(csv_path)
        result = predict_new_data(df)
        return {**state, "prediction_result": result.to_string(index=False), "error": None}
    except Exception as e:
        return {**state, "error": f"Prediction failed: {e}"}


def lookup_node(state: AgentState) -> AgentState:
    query = state["query"]
    id_match = re.search(r"\b(\d{4,6})\b", query)
    if not id_match:
        return {**state, "error": "Could not find a valid customer ID in your query."}
    customer_id = int(id_match.group(1))
    result = lookup_customer(customer_id)
    if isinstance(result, dict) and result.get("error"):
        return {**state, "error": result["error"]}
    import json
    return {**state, "customer_id": customer_id, "customer_info": json.dumps(result[:5], default=str), "error": None}


def explain_node(state: AgentState) -> AgentState:
    if state.get("error"):
        return {**state, "explanation": f"Error: {state['error']}"}

    llm = _get_chain()
    route = state.get("route")

    if route == "predict":
        messages = [
            ("system", "You are a customer analytics assistant. Summarise the churn predictions below concisely and highlight the top 3 highest-risk customers."),
            ("human", state.get("prediction_result", ""))
        ]
    else:
        messages = [
            ("system", "You are a customer analytics assistant. Given the customer transaction records below, summarise who this customer is and their purchasing behaviour in 3-4 sentences."),
            ("human", state.get("customer_info", ""))
        ]

    explanation = llm.invoke(messages).content.strip()
    return {**state, "explanation": explanation}
