import gradio as gr
import pandas as pd
from pathlib import Path

# ----------------------------------
# Comic-inspired theme
# ----------------------------------

comic_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="yellow",
    font=["Comic Sans MS", "Trebuchet MS", "sans-serif"]
)

comic_css = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

#header {
    background-color: #FFE66D !important;
    border: 3px solid black;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    box-shadow: 5px 5px 0px black;
}

#header h1, #header p {
    color: black !important;
}

#instructions {
    background-color: #FFE66D !important;
    border: 3px solid black;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 5px 5px 0px black;
    color: black !important;
}

#instructions h1 {
    color: black !important;
    text-align: center;
}

#instructions p {
    color: black !important;
    text-align: left;
}

#upload-section {
    border: 2px solid black;
    border-radius: 15px;
    padding: 15px;
}
"""

# ----------------------------------
# Function to read CSV files
# ----------------------------------


from pathlib import Path

def load_csv(file_path):
    if file_path is None:
        return (
            pd.DataFrame(),
            "Please upload a CSV file first.",
            "",
            "",
            pd.DataFrame()
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
                pd.DataFrame()
            )

        total_rows = str(df.shape[0])
        total_columns = str(df.shape[1])

        return (
            df.head(100),
            "CSV file loaded successfully!",
            total_rows,
            total_columns,
            df
        )

    except Exception as error:
        return (
            pd.DataFrame(),
            f"Error loading CSV: {error}",
            "",
            ""
        )


# ----------------------------------
# Function to Analyze Questions (Placeholder)
# ----------------------------------


def analyze_question(question, df):
    if df is None or df.empty:
        return "Please upload a CSV file first.", pd.DataFrame()

    if not question or not question.strip():
        return "Please enter a question.", pd.DataFrame()

    question = question.lower().strip()

    if "total sales by category" in question:
        result = (
            df.groupby("Category", as_index=False)["Sales"]
            .sum()
        )
        return "Total sales grouped by category:", result

    elif "total sales" in question:
        total = df["Sales"].sum()
        return f"Total Sales: ${total:,.2f}", pd.DataFrame()

    elif "average profit" in question:
        average = df["Profit"].mean()
        return f"Average Profit: ${average:,.2f}", pd.DataFrame()

    elif "how many orders" in question:
        if "Order ID" in df.columns:
            count = df["Order ID"].nunique()
        else:
            count = len(df)

        return f"Number of unique orders: {count:,}", pd.DataFrame()

    else:
        return (
            "Question not supported yet. "
            "Try asking about total sales, average profit, "
            "number of orders, or total sales by category.",
            pd.DataFrame()
        )


# ----------------------------------
# Build the Gradio interface
# ----------------------------------

with gr.Blocks(
    title="Data Analysis Agent"
) as demo:
    dataset_state = gr.State(value=None)

    gr.Markdown(
        """
        <div id="header">
            <h1>Data Analysis Agent</h1>
            <p>Upload your CSV file and explore the data!</p>
        </div>
        """,
        elem_id="header"
    )

    gr.Markdown(
        """
        <div id="instructions">
            <h1>📄 Instructions</h1>
            <p>1. Upload a CSV file by dropping it into the upload area or clicking to select a file.</p>
            <p>2. Click "Load Dataset" to preview the data.</p>
            <p>3. Ask question(s) about your dataset in the input box under "Your Question".</p>
            <p>4. Click "Analyze Data" to get insights from your dataset and question.</p>
        </div>
        """,
        elem_id="instructions"
    )

    gr.Markdown("## 📁 Upload Your Dataset")

    with gr.Group(elem_id="upload-section"):

        csv_file = gr.File(
            label="Upload CSV File",
            file_types=[".csv"],
            type="filepath"
        )

        load_button = gr.Button(
            "🚀 Load Dataset",
            variant="primary"
        )

    status = gr.Textbox(
        label="Status",
        interactive=False
    )

    gr.Markdown("## 📈 Dataset Overview")

    with gr.Row():
        row_count = gr.Textbox(
            label="Total Rows",
            interactive=False
        )

        column_count = gr.Textbox(
            label="Total Columns",
            interactive=False
        )

    gr.Markdown("## 🔍 Dataset Preview")

    dataset_table = gr.Dataframe(
        label="CSV Data (First 100 Rows)",
        interactive=False
    )

    # Connect button to CSV loading function
    load_button.click(
        fn=load_csv,
        inputs=[csv_file],
        outputs=[
            dataset_table,
            status,
            row_count,
            column_count,
            dataset_state
        ]
    )

    gr.Markdown("## 💬 Ask Questions About Your Dataset")

    question_input = gr.Textbox(
        label="Your Question",
        placeholder="Example: What is the total sales?",
        lines=2
    )

    analyze_button = gr.Button(
        "⚡ Analyze Data",
        variant="primary"
    )

    analysis_output = gr.Textbox(
        label="Analysis Answer",
        interactive=False
    )

    analysis_table = gr.Dataframe(
        label="Analysis Results",
        interactive=False
    )

    analyze_button.click(
        fn=analyze_question,
        inputs=[question_input, dataset_state],
        outputs=[analysis_output, analysis_table]
    )

# ----------------------------------
# Start Gradio
# ----------------------------------

if __name__ == "__main__":
    demo.launch(
        theme=comic_theme,
        css=comic_css
    )
