
# Data Analysis AI Agent
## CS179M — Option A Architecture

## 1. Project Overview

Our goal is to develop an AI-powered data analysis agent
that allows users to upload CSV datasets and ask questions
about their data using natural language.

The agent will use an LLM to generate Python/pandas code,
execute the code in an isolated environment, and return
accurate calculations, explanations, and analytical insights.

The system is designed to support different CSV datasets
without hardcoding calculations for a single dataset.

Our initial demonstration and evaluation will use the
Sample Superstore dataset.

### Selected Approach: Option A

We are using LLM-generated Python code rather than
predefined calculation functions.

The LLM will:
- Understand the user's question.
- Examine the dataset structure and available columns.
- Generate Python/pandas code for the analysis.
- Receive execution results or errors.
- Correct its approach when necessary.
- Explain the final results to the user.

Mini-swe-agent will provide the core agent loop and
coordinate communication between the model and tools.

---

## 2. Project Directory Structure

    Data-Analysis-AI-Agent/
    |
    |-- app/
    |   |-- __init__.py
    |   |-- app.py
    |
    |-- agent/
    |   |-- __init__.py
    |   |-- llm_client.py
    |   |-- prompts.py
    |   |-- agent_runner.py
    |   |-- execution_env.py
    |
    |-- data/
    |   |-- superstore_orders.csv
    |
    |-- tests/
    |   |-- test_app.py
    |   |-- test_llm_client.py
    |   |-- test_agent_runner.py
    |   |-- test_execution_env.py
    |
    |-- evaluation/
    |   |-- benchmark_questions.json
    |   |-- run_benchmarks.py
    |
    |-- ARCHITECTURE.md
    |-- .env.example
    |-- .gitignore
    |-- requirements.txt
    |-- README.md

---

## 3. Component Responsibilities

### Component 1: Gradio Interface

Files:
- app/app.py
- tests/test_app.py

Responsibilities:
- Allow users to upload CSV datasets.
- Load and preview uploaded data.
- Accept natural-language questions.
- Send questions and dataset references to the agent.
- Display final answers, tables, and errors.
- Handle invalid uploads and user-facing errors.

The existing Gradio interface will be reused and
connected to the agent once integration is ready.

### Component 2: LLM Integration

Files:
- agent/llm_client.py
- agent/prompts.py
- tests/test_llm_client.py

Responsibilities:
- Configure the selected LLM through NRP.
- Manage API connection settings and authentication.
- Integrate the model with mini-swe-agent.
- Develop instructions for Python/pandas code generation.
- Provide relevant dataset information to the LLM.
- Guide the LLM to explain results clearly.
- Test model connectivity and basic responses.

Important:
Mini-swe-agent already supports model integration.
The implementation should reuse its existing model
interface wherever practical rather than creating
an unnecessary second agent loop.

### Component 3: Agent Integration

Files:
- agent/agent_runner.py
- tests/test_agent_runner.py

Responsibilities:
- Integrate mini-swe-agent into our application.
- Connect the LLM and execution environment.
- Receive the user's question and dataset reference.
- Configure and start the agent's analysis process.
- Use mini-swe-agent's existing agent loop.
- Handle final results and agent-level failures.
- Return a consistent response to Gradio.

Proposed public function:

    run_analysis(csv_path, question)

Proposed response format:

    {
        "answer": "...",
        "table": None,
        "error": None
    }

The exact interface and response format must be
agreed upon before integration.

The runner should coordinate existing components,
not recreate the entire mini-swe-agent framework.

### Component 4: Code Execution and Security

Files:
- agent/execution_env.py
- tests/test_execution_env.py

Responsibilities:
- Configure isolated execution of LLM-generated code.
- Provide controlled access to the uploaded dataset.
- Make Python/pandas available inside the environment.
- Capture execution results and error messages.
- Apply execution time and resource limits.
- Restrict filesystem and network access.
- Prevent generated code from accessing host secrets.
- Integrate execution with mini-swe-agent's tool system.

Security Warning:

Mini-swe-agent's default local command execution is
not a secure sandbox.

Untrusted LLM-generated Python code must not execute
unrestricted on the host computer.

A properly isolated execution environment must be
implemented and tested before enabling generated-code
execution for uploaded datasets.

---

## 4. Agent Workflow

Step 1: Dataset Upload

The user uploads a CSV through the Gradio interface.

Step 2: User Question

The user asks a natural-language question about
the dataset.

Example:

    What are the total sales by region?

Step 3: Agent Initialization

Gradio sends the question and dataset reference to
agent_runner.py.

The runner initializes mini-swe-agent using the
configured LLM and execution environment.

Step 4: Code Generation

The LLM receives the question and relevant dataset
information.

It generates Python/pandas code to perform the
requested analysis.

Step 5: Code Execution

The generated code is submitted to the isolated
execution environment.

The execution environment runs the code and captures
the results or errors.

Step 6: Agent Feedback

Execution results are returned to mini-swe-agent.

If execution fails, the LLM can examine the error
and attempt a correction.

Successful execution does not automatically mean
the calculation is correct.

Step 7: Final Response

The agent produces a final answer based on the
execution results.

The runner returns the response to Gradio.

Step 8: Results Display

Gradio displays the answer and any supported
tables or analytical results.

---

## 5. Testing Strategy

Each contributor is responsible for testing their
respective component.

### test_app.py

Test:
- CSV loading.
- Dataset preview.
- Missing or invalid files.
- Interface integration.

### test_llm_client.py

Test:
- Model configuration.
- API connection handling.
- Response handling.
- Missing or invalid credentials.

### test_agent_runner.py

Test:
- Request handling.
- Integration with model and execution components.
- Final response formatting.
- Error propagation.
- Agent behavior using mocked dependencies.

### test_execution_env.py

Test:
- Python execution in the isolated environment.
- Dataset access restrictions.
- Successful result capture.
- Syntax and runtime errors.
- Execution time limits.
- Filesystem and network restrictions.

Security tests should verify that restrictions
actually work, not simply assume they are enforced.

---

## 6. Agent Evaluation

Directory:
- evaluation/

Files:
- benchmark_questions.json
- run_benchmarks.py

Our proposal includes approximately 25 benchmark
questions for evaluating the completed agent.

### Evaluation Criteria

1. Numerical Accuracy
   - Does the agent calculate the correct answer?
   - Compare results against independently verified
     expected answers.

2. Code Execution Success
   - Does the generated Python code execute correctly?
   - Track successful and failed executions.

3. Error Recovery
   - Can the agent identify execution failures?
   - Can it correct unsuccessful attempts?

4. Explanation Quality
   - Does the agent explain its findings clearly?
   - Are explanations supported by the actual data?
   - Does it avoid unsupported conclusions?

### Important Distinction

Code that executes successfully can still produce
incorrect numerical results.

For example, an agent might calculate an average
when the user requested a sum.

Therefore, execution success and answer accuracy
must be measured separately.

---

## 7. Development Milestones

### Milestone 1: Basic End-to-End Integration

Goal:
Successfully answer one simple question using
LLM-generated pandas code.

Example:

    What are the total sales?

Expected process:
- Upload Superstore CSV.
- Send question to mini-swe-agent.
- Generate pandas code.
- Execute code in the isolated environment.
- Return the calculated result.
- Display the answer in Gradio.

### Milestone 2: Basic Analytical Questions

Support questions involving:
- Sums and averages.
- Counts.
- Grouping.
- Filtering.
- Sorting.
- Minimum and maximum values.

The LLM generates the necessary code dynamically.
These are not manually hardcoded calculation functions.

### Milestone 3: Intermediate Analysis

Support:
- Comparisons between groups.
- Trends over time.
- Date-based analysis.
- Calculated metrics.
- Multi-step analytical questions.

### Milestone 4: Reliability Improvements

Improve:
- Error handling.
- Code generation instructions.
- Execution reliability.
- Numerical correctness.
- Result explanations.

### Milestone 5: Evaluation and Final Demonstration

- Complete approximately 25 benchmark questions.
- Compare actual answers against expected results.
- Document failures and limitations.
- Demonstrate the final agent through Gradio.

---

## 8. Development Guidelines

- Preserve existing teammate implementations.
- Do not hardcode calculations to Sample Superstore.
- Reuse mini-swe-agent's existing agent loop.
- Never commit API keys or real .env files.
- Keep credentials in environment variables.
- Never run untrusted generated code unrestricted.
- Write tests for each component.
- Agree on component interfaces before integration.
- Keep development focused on the core agent first.

Optional features, including web research, advanced
visualizations, and domain-specific analysis, should
be considered only after the main agent works reliably.

---

## 9. Current Project Status

Completed:
- GitHub repository initialized.
- Conda development environment configured.
- Existing Gradio interface created.
- Sample Superstore CSV added.
- Initial Gradio CSV-loading tests created.
- Option A architecture selected.
- Project folders and placeholder files created.

Not Yet Implemented:
- NRP model integration.
- Agent prompts.
- Mini-swe-agent integration.
- Isolated execution environment.
- Gradio-to-agent connection.
- Benchmark evaluation system.

The new Python files are currently placeholders.
Their existence does not mean the components
have been implemented.

---

## 10. Next Steps

1. Review and approve this architecture as a team.
2. Confirm component ownership.
3. Agree on the interface between Gradio and the runner.
4. Configure LLM integration.
5. Set up and test isolated execution.
6. Integrate the components using mini-swe-agent.
7. Test a simple end-to-end analysis question.
8. Expand supported questions and run evaluations.
