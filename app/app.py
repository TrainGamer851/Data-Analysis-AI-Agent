import gradio as gr
import pandas as pd
from pathlib import Path

# ----------------------------------
# Business dashboard theme
# ----------------------------------

business_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "Segoe UI", "Arial", "sans-serif"],
)

business_css = """
.gradio-container {
    max-width: 1260px !important;
    margin: 0 auto !important;
    padding: 22px 20px 40px !important;
}
#header {
    background: linear-gradient(115deg, #10243A 0%, #183B60 100%);
    border-radius: 18px;
    padding: 30px 34px;
    margin-bottom: 20px;
    box-shadow: 0 12px 28px rgba(15, 35, 55, 0.12);
}
#header h1 {
    color: #FFFFFF !important;
    font-size: 2rem !important;
    margin: 0 0 8px !important;
}
#header p {
    color: #D9E6F2 !important;
    margin: 0 !important;
    font-size: 1rem;
}
#workspace-heading {
    margin: 6px 0 12px;
}
#workspace-heading h2 {
    font-size: 1.18rem !important;
    font-weight: 650 !important;
}
#dataset-panel, #chat-panel {
    border: 1px solid #DCE4EC;
    border-radius: 16px;
    padding: 18px;
    background: var(--block-background-fill);
    box-shadow: 0 3px 14px rgba(15, 35, 55, 0.045);
}
#dataset-panel h3, #chat-panel h3 {
    font-size: 1.06rem !important;
}
#dataset-status {
    margin-top: 8px;
}
#chatbot {
    border: 1px solid #E0E7EF;
    border-radius: 12px;
}
#send-button {
    min-width: 92px;
}
#footer-note {
    margin-top: 14px;
    font-size: 0.83rem;
    color: var(--body-text-color-subdued);
}

/* Force both dashboard panels to the same height */
#dataset-panel,
#chat-panel {
    height: 820px !important;
    min-height: 820px !important;
    max-height: 820px !important;
    box-sizing: border-box !important;
    align-self: stretch !important;
}

/* Scroll dataset contents instead of expanding the panel */
#dataset-panel {
    overflow-y: auto !important;
}

/* Keep chat content inside its panel */
#chat-panel {
    overflow: hidden !important;
}

/* Limit preview table height */
#dataset-panel .dataframe {
    max-height: 220px !important;
    overflow: auto !important;
}
"""

# ----------------------------------
# Function to read CSV files
# ----------------------------------


def load_csv(file_path):
    if file_path is None:
        return (
            pd.DataFrame(),
            "Please upload a CSV file first.",
            "",
            "",
            pd.DataFrame(),
        )

    try:
        file_name = Path(file_path).name
        df = pd.read_csv(file_path)

        if df.empty:
            return (
                pd.DataFrame(),
                f"CSV file '{file_name}' is empty.",
                "",
                "",
                pd.DataFrame(),
            )

        total_rows = str(df.shape[0])
        total_columns = str(df.shape[1])

        return (
            df.head(100),
            "CSV file loaded successfully!",
            total_rows,
            total_columns,
            df,
        )

    except Exception as error:
        return (pd.DataFrame(), f"Error loading CSV: {error}", "", "", pd.DataFrame())


# ----------------------------------
# Function to Analyze Questions (Placeholder)
# ----------------------------------


def analyze_question(question, dataframe):
    if dataframe is None or dataframe.empty:
        return "Please upload a CSV file first."

    if not question or not question.strip():
        return "Please enter a question about your dataset."

    return (
        f"Question received: {question}\n\n" "AI analysis will be available in Stage 2."
    )


# ----------------------------------
# Chat interface callbacks
# ----------------------------------


def send_message(question, history, dataframe):
    """Append one user/assistant exchange to session chat history."""
    history = list(history or [])
    if not question or not question.strip():
        return history, ""

    question = question.strip()
    # UI-only integration point: replace analyze_question with the agent
    # runner once the backend team provides a stable function.
    answer = analyze_question(question, dataframe)
    history.extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": str(answer)},
        ]
    )
    return history, ""


def load_csv_and_reset_chat(file_path):
    """Load a dataset and start a fresh conversation for it."""
    preview, status, rows, columns, df = load_csv(file_path)
    return preview, status, rows, columns, df, []


# ----------------------------------
# Build the Gradio interface
# ----------------------------------

with gr.Blocks(title="Business Insights | Data Analysis") as demo:
    dataset_state = gr.State(value=None)

    with gr.Column(elem_id="header"):
        gr.Markdown(
            "# Business Insights Workspace\n"
            "Explore your business data, ask follow-up questions, "
            "and keep your analysis in one conversation."
        )

    with gr.Row(equal_height=True):
        with gr.Column(scale=4, min_width=300, elem_id="dataset-panel"):
            gr.Markdown("### Dataset Management")
            gr.Markdown(
                "Upload a CSV to review its structure and start an analysis session."
            )
            csv_file = gr.File(
                label="Business dataset (.csv)",
                file_types=[".csv"],
                type="filepath",
            )
            load_button = gr.Button("Load Dataset", variant="primary")
            status = gr.Textbox(
                label="Dataset status",
                interactive=False,
                elem_id="dataset-status",
            )
            with gr.Row():
                row_count = gr.Textbox(label="Total records", interactive=False)
                column_count = gr.Textbox(label="Data fields", interactive=False)
            with gr.Accordion("Data preview", open=False):
                dataset_table = gr.Dataframe(
                    label="First 100 records",
                    interactive=False,
                )

        with gr.Column(scale=7, min_width=420, elem_id="chat-panel"):
            gr.Markdown("### Business Analysis Assistant")
            gr.Markdown(
                "Ask about sales, profitability, customer trends, "
                "or operational performance."
            )
            chatbot = gr.Chatbot(
                label="Analysis conversation",
                height=420,
                value=[],
                elem_id="chatbot",
            )
            with gr.Row():
                question_input = gr.Textbox(
                    label="Ask a business question",
                    placeholder="e.g. Which category has the highest total profit?",
                    lines=2,
                    scale=8,
                )
                send_button = gr.Button(
                    "Send",
                    variant="primary",
                    scale=1,
                    elem_id="send-button",
                )
            clear_button = gr.Button(
                "Clear Conversation",
                variant="secondary",
            )

    gr.Markdown(
        "Analysis responses currently use the existing placeholder backend. "
        "The agent team can connect its runner to `analyze_question()` later.",
        elem_id="footer-note",
    )

    load_button.click(
        fn=load_csv_and_reset_chat,
        inputs=[csv_file],
        outputs=[
            dataset_table,
            status,
            row_count,
            column_count,
            dataset_state,
            chatbot,
        ],
    )
    send_button.click(
        fn=send_message,
        inputs=[question_input, chatbot, dataset_state],
        outputs=[chatbot, question_input],
    )
    question_input.submit(
        fn=send_message,
        inputs=[question_input, chatbot, dataset_state],
        outputs=[chatbot, question_input],
    )
    clear_button.click(fn=lambda: [], outputs=[chatbot])


if __name__ == "__main__":
    demo.launch(theme=business_theme, css=business_css)
