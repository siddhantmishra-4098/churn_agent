from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import router_node, predict_node, lookup_node, explain_node


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router",  router_node)
    graph.add_node("predict", predict_node)
    graph.add_node("lookup",  lookup_node)
    graph.add_node("explain", explain_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "predict": "predict",
            "lookup":  "lookup",
        }
    )

    graph.add_edge("predict", "explain")
    graph.add_edge("lookup",  "explain")
    graph.add_edge("explain", END)

    return graph.compile()


if __name__ == "__main__":
    import argparse
    agent = build_graph()

    parser = argparse.ArgumentParser(description="Churn Prediction Agent")
    parser.add_argument("--query", required=True)
    parser.add_argument("--csv",   required=False, default=None)
    args = parser.parse_args()

    result = agent.invoke({
        "query":    args.query,
        "csv_path": args.csv,
    })

    print(result["explanation"])
