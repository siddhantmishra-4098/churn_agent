import gradio as gr
from agent.graph import build_graph

agent = build_graph()


def run_agent(query: str, csv_file):
    if not query.strip():
        return "Please enter a query."

    state = {
        "query":    query,
        "csv_path": csv_file.name if csv_file else None,
    }

    try:
        result = agent.invoke(state)
        return result.get("explanation", "No response generated.")
    except Exception as e:
        return f"Error: {e}"


with gr.Blocks(title="Churn Prediction Agent") as demo:
    gr.Markdown("## Churn Prediction Agent")
    gr.Markdown("Upload new customer data for churn predictions, or ask about a specific customer.")

    with gr.Row():
        query_box = gr.Textbox(
            label="Your Query",
            placeholder="e.g. 'Predict churn for this data' or 'Tell me about customer 12347'",
            lines=2
        )
        csv_upload = gr.File(label="Upload CSV (for predictions)", file_types=[".csv"])

    submit_btn = gr.Button("Run", variant="primary")
    output_box = gr.Textbox(label="Response", lines=10)

    submit_btn.click(fn=run_agent, inputs=[query_box, csv_upload], outputs=output_box)


if __name__ == "__main__":
    demo.launch(share=True)
