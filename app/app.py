import gradio as gr
import pandas as pd
import threading
from agent.debug_logger import debug_log, DebugTimer

from pathlib import Path
from agent.agent_runner import run_analysis
from agent.analysis_controller import (
    AnalysisJob,
    run_controlled_analysis,
    shutdown_analysis_jobs,
)
from app.table_formatter import business_table_to_markdown
from app.chart_builder import build_chart
from agent.conversation_context import (
    build_conversation_context,
)

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
/* Chat conversation container */
#chatbot {
    border: 1px solid #E0E7EF;
    border-radius: 12px;
    overflow: hidden;
}
/* Animated assistant thinking indicator */
#chatbot .thinking-indicator {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 7px !important;
    min-width: 42px;
    min-height: 24px;
    padding: 4px 6px;
}

#chatbot .thinking-dot {
    display: inline-block !important;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: var(--body-text-color);
    opacity: 0.3;

    animation: thinking-pulse 1.2s infinite ease-in-out;
}

#chatbot .thinking-dot:nth-child(1) {
    animation-delay: 0s;
}

#chatbot .thinking-dot:nth-child(2) {
    animation-delay: 0.16s;
}

#chatbot .thinking-dot:nth-child(3) {
    animation-delay: 0.32s;
}

@keyframes thinking-pulse {
    0%, 60%, 100% {
        opacity: 0.3;
        transform: translateY(0);
    }

    30% {
        opacity: 1;
        transform: translateY(-4px);
    }
}

/* Accessibility: respect reduced-motion preferences */
@media (prefers-reduced-motion: reduce) {
    #chatbot .thinking-dot {
        animation: none;
        opacity: 0.8;
    }
}

/* External component labels */
#conversation-label,
#question-label {
    margin: 0 !important;
    padding: 0 !important;
}

#conversation-label p,
#question-label p {
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: var(--body-text-color) !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Question input and action button alignment */
#question-row {
    display: flex !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    gap: 12px !important;
}

/* Remove unnecessary textbox spacing */
#question-input {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* Fixed question entry height */
#question-input textarea {
    width: 100% !important;
    height: 60px !important;
    min-height: 60px !important;
    max-height: 60px !important;
    box-sizing: border-box !important;
    padding: 12px !important;
    margin: 0 !important;
    resize: none !important;
    border-radius: 8px !important;
    line-height: 20px !important;
}

/* Identical Send and Stop button dimensions */
#send-button,
#stop-button {
    flex: 0 0 100px !important;
    width: 100px !important;
    min-width: 100px !important;
    max-width: 100px !important;

    height: 60px !important;
    min-height: 60px !important;
    max-height: 60px !important;

    box-sizing: border-box !important;
    align-self: center !important;

    margin: 0 !important;
    padding: 0 12px !important;

    border-radius: 8px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
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

/* =========================================
   Business section labels
   ========================================= */

.business-blue-label,
#analysis-plot::before {
    display: inline-flex;
    align-items: center;
    width: fit-content;
    max-width: 100%;

    background: #2563eb;
    color: #ffffff !important;

    font-size: 14px;
    font-weight: 600;
    line-height: 20px;

    padding: 4px 8px;
    border-radius: 6px;

    position: static !important;
    box-sizing: border-box;
}

/* Conversation and question labels */

#conversation-label,
#question-label {
    position: relative !important;
    padding: 0 !important;
    margin: 0 0 6px 0 !important;
    min-height: 28px;
}

#conversation-label p,
#question-label p {
    margin: 0 !important;
    padding: 0 !important;
}

/* Visualization heading */

#analysis-plot::before {
    content: "Business Data Visualization";
    margin-bottom: 16px;
    flex-shrink: 0;
}

/* Keep heading and chart in separate rows */

#analysis-plot {
    flex-direction: column !important;
    align-items: stretch !important;
    padding: 12px !important;
    box-sizing: border-box;
}

#analysis-plot > * {
    min-width: 0;
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
            None,
        )

    try:
        timer = DebugTimer("DATASET")

        debug_log(
            "DATASET",
            "Reading uploaded CSV",
        )

        file_name = Path(file_path).name
        df = pd.read_csv(file_path)

        debug_log(
            "DATASET",
            f"Rows: {df.shape[0]:,} | "
            f"Columns: {df.shape[1]}",
        )

        debug_log(
            "DATASET",
            f"Available fields: {list(df.columns)}",
        )

        timer.finish("CSV read completed")

        if df.empty:
            return (
                pd.DataFrame(),
                f"CSV file '{file_name}' is empty.",
                "",
                "",
                None,
            )

        return (
            df.head(100),
            "CSV file loaded successfully!",
            str(df.shape[0]),
            str(df.shape[1]),
            str(file_path),
        )

    except Exception as error:
        return (
            pd.DataFrame(),
            f"Error loading CSV: {error}",
            "",
            "",
            None,
        )


# ----------------------------------
# Function to Analyze Questions (Placeholder)
# ----------------------------------


def analyze_question_with_chart(
    question,
    csv_path,
    job=None,
    context=None,
):
    if not csv_path:
        return "Please upload a CSV file first.", None

    if not question or not question.strip():
        return (
            "Please enter a question about your dataset.",
            None,
        )

    try:
        context = build_conversation_context(context)

        if job is None:
            response = run_analysis(
                csv_path,
                question,
                explain=True,
                context=context,
            )

        else:
            response = run_controlled_analysis(
                csv_path,
                question,
                job=job,
                explain=True,
                context=context,
            )

        if response.get("error"):
            if response["error"] == "Analysis stopped.":
                return "Analysis stopped.", None

            return (
                f"**Analysis Error:** {response['error']}",
                None,
            )

        answer = str(
            response.get("answer") or "No answer returned."
        )

        table = response.get("table")
        figure = None

        if table:
            presentation_timer = DebugTimer("PRESENTATION")

            debug_log(
                "PRESENTATION",
                "Preparing supporting data for frontend",
            )

            result_df = pd.DataFrame(table)

            result_df = result_df.drop(
                columns=["index", "level_0"],
                errors="ignore",
            )

            debug_log(
                "PRESENTATION",
                f"Supporting data rows: "
                f"{len(result_df):,} | "
                f"Columns: {len(result_df.columns)}",
            )

            debug_log(
                "PRESENTATION",
                f"Supporting data fields: "
                f"{list(result_df.columns)}",
            )

            try:
                formatted_table = (
                    business_table_to_markdown(result_df)
                )

                answer += "\n\n### Supporting Data\n\n"
                answer += formatted_table

                debug_log(
                    "PRESENTATION",
                    "Supporting Markdown table generated",
                )

            except Exception as error:
                debug_log(
                    "PRESENTATION ERROR",
                    f"Table formatting failed: "
                    f"{type(error).__name__}",
                )

                answer += (
                    "\n\nSupporting data could not "
                    "be formatted for display."
                )

            try:
                debug_log(
                    "CHART",
                    "Checking structured data for visualization",
                )

                figure = build_chart(result_df)

                debug_log(
                    "CHART",
                    "Chart generated successfully"
                    if figure is not None
                    else "No suitable chart for returned data",
                )

            except Exception as error:
                # A chart failure must not discard a valid answer.
                debug_log(
                    "CHART ERROR",
                    f"Chart generation failed: "
                    f"{type(error).__name__}",
                )

                figure = None

            presentation_timer.finish(
                "Result presentation completed"
            )

        return answer, figure

    except Exception:
        return (
            "An unexpected analysis error occurred. "
            "Please try again.",
            None,
        )


def analyze_question(
    question,
    csv_path,
    job=None,
    context=None,
):
    """Preserve the existing text-only analysis interface."""

    answer, _ = analyze_question_with_chart(
        question,
        csv_path,
        job=job,
        context=context,
    )

    return answer

# ----------------------------------
# Chat interface callbacks
# ----------------------------------


_active_jobs = {}
_jobs_lock = threading.Lock()

THINKING_INDICATOR = (
    '<span class="thinking-indicator" '
    'role="status" aria-label="Analyzing your dataset">'
    '<span class="thinking-dot"></span>'
    '<span class="thinking-dot"></span>'
    '<span class="thinking-dot"></span>'
    '</span>'
)

def send_message(question, history, csv_path):
    history = list(history or [])

    if not question or not question.strip():
        return history, ""

    question = question.strip()

    context = build_conversation_context(history)

    answer = analyze_question(
        question,
        csv_path,
        context=context,
    )

    history.extend([
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ])

    return history, ""

def send_controlled_message(
    question,
    history,
    csv_path,
    request: gr.Request,
):
    history = list(history or [])

    if not question or not question.strip():
        yield (
            history,
            "",
            *stop_processing(),
            gr.skip(),
        )
        return

    question = question.strip()
    session_id = request.session_hash
    
    job = AnalysisJob()

    timer = DebugTimer(
        "ANALYSIS",
        job_id=job.job_id,
    )

    debug_log(
        "FRONTEND",
        "New analysis requested",
        job_id=job.job_id,
    )

    debug_log(
        "QUESTION",
        question,
        job_id=job.job_id,
    )

    with _jobs_lock:
        if session_id in _active_jobs:
            return

        _active_jobs[session_id] = job

    debug_log(
        "FRONTEND",
        "Analysis job registered",
        job_id=job.job_id,
    )

    figure = None

    try:
        print(
            f"[FRONTEND DEBUG] History messages: {len(history)}",
            flush=True,
        )

        for message in history:
            if isinstance(message, dict):
                print(
                    f"[FRONTEND DEBUG] "
                    f"Role: {message.get('role')}, "
                    f"Content type: "
                    f"{type(message.get('content')).__name__}",
                    flush=True,
                )
            else:
                print(
                    f"[FRONTEND DEBUG] Message type: "
                    f"{type(message).__name__}",
                    flush=True,
                )

        context = build_conversation_context(history)

        debug_log(
            "CONTEXT",
            f"Extracted {len(context)} historical messages",
            job_id=job.job_id,
        )

        print(
            f"[FRONTEND DEBUG] Extracted context: "
            f"{len(context)} messages",
            flush=True,
        )

        history.append({
            "role": "user",
            "content": question,
        })

        # Display a temporary animated assistant message.
        # Do not add it to the actual conversation history.
        pending_history = list(history) + [
            {
                "role": "assistant",
                "content": THINKING_INDICATOR,
            }
        ]

        print(
            "[FRONTEND DEBUG] Thinking indicator displayed",
            flush=True,
        )

        yield (
            pending_history,
            "",
            *start_processing(),
            gr.update(value=None, visible=False),
        )

        print(
            "[FRONTEND DEBUG] Starting backend analysis",
            flush=True,
        )

        debug_log(
            "BACKEND",
            "Dispatching analysis request",
            job_id=job.job_id,
        )

        answer, figure = analyze_question_with_chart(
            question,
            csv_path,
            job=job,
            context=context,
        )

        debug_log(
            "BACKEND",
            "Analysis response received",
            job_id=job.job_id,
        )

        print(
            "[FRONTEND DEBUG] Backend analysis returned",
            flush=True,
        )

        if job.cancelled:
            print(
                "[FRONTEND DEBUG] Analysis cancelled by user",
                flush=True,
            )

            answer = "Analysis stopped."
            figure = None

        # Log the final answer.
        debug_log(
            "ANSWER",
            answer,
            job_id=job.job_id,
        )

        # Log chart generation status.
        debug_log(
            "CHART",
            "Chart generated" if figure is not None
            else "No chart returned",
            job_id=job.job_id,
        )

        # Replace the temporary indicator with the final answer.
        history.append({
            "role": "assistant",
            "content": answer,
        })

        print(
            "[FRONTEND DEBUG] Thinking indicator removed",
            flush=True,
        )

        print(
            "[FRONTEND DEBUG] Final answer ready",
            flush=True,
        )

        print(
            f"[FRONTEND DEBUG] Chart available: "
            f"{figure is not None}",
            flush=True,
        )

        yield (
            list(history),
            "",
            *start_processing(),
            gr.update(
                value=figure,
                visible=figure is not None,
            ),
        )

    finally:
        job.cancel()

        with _jobs_lock:
            if _active_jobs.get(session_id) is job:
                _active_jobs.pop(session_id, None)

            debug_log(
                "CLEANUP",
                "Frontend job released",
                job_id=job.job_id,
            )

            timer.finish("Analysis lifecycle finished")

    yield (
        list(history),
        "",
        *stop_processing(),
        gr.update(
            value=figure,
            visible=figure is not None,
        ),
    )

def cancel_active_analysis(request: gr.Request):
    session_id = request.session_hash

    with _jobs_lock:
        job = _active_jobs.get(session_id)

        if job is not None:
            debug_log(
                "CANCEL",
                "Stop button pressed",
                job_id=job.job_id,
            )

            job.cancel()

            debug_log(
                "CANCEL",
                "Cancellation signal sent to controller",
                job_id=job.job_id,
            )

def shutdown_application():
    """Cancel active analyses before exiting."""

    print("\n[Shutdown] Stopping active analyses...")

    with _jobs_lock:
        for job in _active_jobs.values():
            job.cancel()

    cleaned = shutdown_analysis_jobs(timeout=60)

    if cleaned:
        print("[Shutdown] Analysis cleanup completed.")
    else:
        print("[Shutdown] Some jobs did not finish cleanup.")

def load_csv_and_reset_chat(file_path):
    preview, status, rows, columns, csv_path = load_csv(file_path)
    return preview, status, rows, columns, csv_path, []


# ----------------------------------
# Build the Gradio interface
# ----------------------------------

def start_processing():
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )


def stop_processing():
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
    )

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
            # Conversation label outside the chatbot
            gr.Markdown(
                '<span class="business-blue-label">'
                'Analysis Conversation'
                '</span>',
                elem_id="conversation-label",
            )

            chatbot = gr.Chatbot(
                label="Analysis conversation",
                show_label=False,
                height=420,
                value=[],
                elem_id="chatbot",
            )

            # Question label outside the input row
            gr.Markdown(
                '<span class="business-blue-label">'
                'Ask a Business Question'
                '</span>',
                elem_id="question-label",
            )

            with gr.Row(
                equal_height=True,
                elem_id="question-row",
            ):
                question_input = gr.Textbox(
                    label="Ask a business question",
                    show_label=False,
                    placeholder=(
                        "e.g. Which category has the highest total profit?"
                    ),
                    lines=2,
                    max_lines=2,
                    scale=8,
                    min_width=0,
                    container=False,
                    elem_id="question-input",
                )

                send_button = gr.Button(
                    "Send",
                    variant="primary",
                    scale=0,
                    min_width=100,
                    elem_id="send-button",
                )

                stop_button = gr.Button(
                    "Stop",
                    variant="stop",
                    scale=0,
                    min_width=100,
                    visible=False,
                    elem_id="stop-button",
                )

            clear_button = gr.Button(
                "Clear Conversation",
                variant="secondary",
            )

    analysis_plot = gr.Plot(
        label="Business Data Visualization",
        show_label=False,
        visible=False,
        elem_id="analysis-plot",
    )

    gr.Markdown(
        "AI-powered analysis using Python, pandas, and mini-swe-agent. "
        "Review calculated results before making business decisions.",
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

    load_button.click(
        fn=lambda: gr.update(
            value=None,
            visible=False,
        ),
        inputs=None,
        outputs=[analysis_plot],
        queue=False,
    )

    processing_controls = [
        send_button,
        stop_button,
        question_input,
        load_button,
        clear_button,
    ]

    # Start a cancellable analysis using Send or Enter.
    analysis_event = gr.on(
        triggers=[
            send_button.click,
            question_input.submit,
        ],
        fn=send_controlled_message,
        inputs=[
            question_input,
            chatbot,
            dataset_state,
        ],
        outputs=[
            chatbot,
            question_input,
            *processing_controls,
            analysis_plot,
        ],
        show_progress="hidden",
        concurrency_limit=None,
        trigger_mode="once",
    )

    # Signal the controller to terminate the active analysis.
    stop_button.click(
        fn=cancel_active_analysis,
        inputs=None,
        outputs=None,
        queue=False,
    )

    clear_button.click(
        fn=lambda: (
            [],
            gr.update(value=None, visible=False),
        ),
        outputs=[
            chatbot,
            analysis_plot,
        ],
    )


if __name__ == "__main__":
    try:
        demo.queue().launch(
            theme=business_theme,
            css=business_css,
        )

    except KeyboardInterrupt:
        pass

    finally:
        shutdown_application()