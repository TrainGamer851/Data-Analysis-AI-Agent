# CS179M Data Analysis Agent

**Team:** Glen Bautista, Shafin Kazi, Ivan Garcia-Mora, Truong Le

## Project overview

We are building an AI agent that lets users upload a CSV file and ask questions about
its data in natural language. We plan to use Gradio, pandas, a lightweight LLM, and
mini-swe-agent. We will use Tableau Sample Superstore for initial development.

## How are we going to build this?

We will use four development stages:

### Stage 1 — Basic application (our current focus)

Create a Gradio interface, upload a CSV file, and display the dataset using pandas.

### Stage 2 — AI integration

Connect our lightweight LLM and mini-swe-agent so users can ask questions about the dataset.

### Stage 3 — Data analysis

Enable the agent to execute Python/pandas code and explain calculated results safely.

### Stage 4 — Testing and improvements

Use benchmark questions, evaluate accuracy, fix failures, and improve the application.

## Stage 1: Completed
Stage 1 is complete when this upload-to-table flow works. 

### Current files

```text
Data-Analysis-AI-Agent/
├── README.md      # Project overview and development stages
├── app.py         # Stage 1 application code, to be written by the team
└── .gitignore     # Keeps local environment files out of Git
```

## Stage 2:  AI Integration (In progress)
**Goal: ** A question and the dataset's columns go to the LLM, which writes pandas code to answer it.

### Done on main
- **`agent/llm_client.py`**: connects to the model through the OpenAI client library.
- **NRP** (default): hosted `qwen3`, needs `NRP_API_KEY` in `.env `.
- **Ollama**: local `qwen3:4b`, set `LLM_PROVIDER-ollama`.
- `generate_response(messages)` sends one request and returns the reply text.
- **`agent/prompt.py`**: the instructions given to the model.
- `SYSTEM_PROMPT`: write only pandas code, use only real columns, never intent or hardcode numbers, and store the answer in variable `result`.
- `build_analysis_prompt(question, df)`: adds the dataset's columns, data types, and the user's question.
- `build_recommendation_prompt(...)`: turns calculated results into a business explanation. Written, but not called anywhere yet.


### In Progress
- [] Add `mini-swe-agent` to `requirements.txt`. It is imported but not listed.
- [] Finish `run_analysis(csv_file, question)`: create and run the agent, then return `{answer, table, error}`
- [] Set a retry limit so the fix-and-retry loop can't run forever.
- [] Merge `runner` into `main`
- [] IN `app/app.py`, replace the placeholder in `analyze_question()` with a call to `run_analysis()`

## Stage 3: Data Analysis
**Goal:** Run the LLM's code safely on the user's data and explain the result.

### Done: code execution sandbox
`execute_code(code, df)` in **`agent/execution_env.py`** runs generated code in a seperate worker process (**`agent/sandbox_worker.py`**):
- Removes markdown code fences from the model's output
- Allows only these imports: `pandas`, `numpy`, `math`, `datatime`, `re`.
- Gives the code only a restricted set of Python built-ins. File and network access are blocked.
- Caps memory and stops code that runs too long (default timeout: 10 seconds).
- Returns `{success, result, result_type, stddout, error}`. Error messages are written so the LLM can use them to fix its code.

| Backend | HOw it runs | Use for | 
| `local` (default) | A separate restricted Python process | 
Development only. Not a full sandbox.

Docker Desktop must be running. 

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env # then add your NRP API key
```
`.env` settings: 
```text
LLM_Provider=nrp # or "ollama"
NRP_API_KEY=your-key-here
NRP_MODEL=qwen3
NRP_API_BASE=https://ellm.nrp-nautilus.io/v1
OLLAMA_MODEL=qwen3.db #only needed for Ollama
SANDBOX_BACKEND=docker  # or "local" for development
```

## Running
```bash
python app/app.py # start Gradio app
python -m pytest  # runs the tests
```
## Project structure

```text
Data-Analysis-AI-Agent/
|---app/app.py
|---agent/
|  |---llm_client.py              # NRP / Ollama connection
|  |---prompts.py                 # System, analysis and recommendation prompts
|  |---agent_runner.py            # mini-swe-agent, integration
|  |---execution_env.py           # execute_code(): sends code to the sandbox
|  |---sandbox_worker.py          # Runs code under restrictions
|---docker/Dockerfile             # Sandbox container image
|--- data/superstore_orders.csv   # Development dataset
|--- evaluation/                  # Benchmark questions and runner
|--- tests/
|--- ARCHITECTURE.md
|--- requirements.txt


**Current status:** Stage 2 in progress; agent runner is unmerged. Stage 3 is half done. The sandbox is built to run generated pandas code safely but is missing the agent. 
