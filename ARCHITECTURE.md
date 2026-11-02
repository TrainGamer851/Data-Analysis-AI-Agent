
# Business Data Analysis AI Agent
## CS179M — Option A Architecture

## 1. Project Overview

Our goal is to develop an AI-powered business data analysis
agent that allows users to upload CSV datasets and ask
business-related questions using natural language.

The agent will use a Large Language Model (LLM) to generate
Python/pandas code, execute that code in an isolated
environment, and return accurate calculations, explanations,
and actionable business insights.

### Project Focus

Our agent will specialize in business data analytics,
including:

- Sales and revenue analysis
- Profit and profitability analysis
- Product and category performance
- Regional performance comparisons
- Monthly, quarterly, and yearly trends
- Business performance metrics
- Identification of trends and underperforming areas
- Data-supported business recommendations

Our primary development and evaluation dataset will be
the Tableau Sample Superstore dataset.

However, the agent should not be hardcoded to work only
with Sample Superstore. It should be able to analyze
other business-related CSV datasets based on their
available columns and structure.

The agent should not assume that every dataset contains
sales, profit, dates, or other specific business fields.

### Selected Approach: Option A

We are using LLM-generated Python code rather than
predefined pandas calculation functions.

The LLM will:

- Understand the user's business question.
- Examine the dataset structure and available columns.
- Determine which calculations are necessary.
- Generate Python/pandas code to perform the analysis.
- Receive execution results or errors.
- Attempt corrections when necessary.
- Explain the findings in business-friendly language.

Mini-swe-agent will provide the core agent loop,
coordinating interactions between the LLM and tools.

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

- Allow users to upload business-related CSV datasets.
- Load and preview uploaded data.
- Accept natural-language business questions.
- Send user questions and dataset references to the agent.
- Display numerical results, tables, and explanations.
- Present business insights in an understandable format.
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
- Guide the LLM toward accurate business calculations.
- Encourage clear explanations of analytical findings.
- Prevent unsupported business conclusions.
- Test model connectivity and basic responses.

The prompts should encourage the agent to:

- Inspect available columns before performing calculations.
- Identify appropriate business metrics.
- Handle missing or invalid data appropriately.
- Interpret dates and numerical columns correctly.
- Distinguish total sales from profit and other metrics.
- Avoid assuming unavailable information.
- Explain findings using actual calculated results.
- Avoid presenting correlations as proven causes.

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

The runner should coordinate the existing components,
not recreate the entire mini-swe-agent framework.

### Component 4: Code Execution and Security

Files:
- agent/execution_env.py
- tests/test_execution_env.py

Responsibilities:

- Configure isolated execution of LLM-generated code.
- Provide controlled access to uploaded CSV datasets.
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

### Step 1: Dataset Upload

The user uploads a business CSV dataset through Gradio.

Example:
Sample Superstore sales data.

### Step 2: User Question

The user asks a natural-language business question.

Example:

    Which region generated the highest total profit?

### Step 3: Agent Initialization

Gradio sends the question and dataset reference to
agent_runner.py.

The runner initializes mini-swe-agent using the
configured LLM and execution environment.

### Step 4: Code Generation

The LLM receives the question and relevant dataset
information.

It determines what calculations are required and
generates Python/pandas code.

For example, the analysis might require grouping
data by region and calculating total profit.

### Step 5: Code Execution

The generated code is submitted to the isolated
execution environment.

The execution environment runs the code and captures
the results or errors.

### Step 6: Agent Feedback

Execution results return to mini-swe-agent.

If execution fails, the LLM can examine the error
and attempt a correction.

Successful execution does not automatically mean
the calculation is correct.

### Step 7: Business Interpretation

The agent interprets the calculated results.

For example, it may identify which region has
the highest total profit and explain the comparison.

The agent should base its explanations on actual data
rather than inventing reasons for business performance.

### Step 8: Final Response

The runner returns the final result to Gradio.

Gradio displays the answer, supporting calculations,
and any available tables or explanations.

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

The benchmark will focus on business data analytics,
primarily using Sample Superstore.

### Example Benchmark Categories

#### Basic Business Calculations

- What are the total sales?
- What is the total profit?
- What is the average sales value?
- How many unique orders are there?

#### Grouping and Comparisons

- Which region has the highest total sales?
- Which category generates the most profit?
- What are total sales by state?
- Which products are the least profitable?

#### Business Trends

- How did monthly sales change over time?
- Which quarter generated the highest sales?
- How did profits differ between quarters?
- Which months experienced declining sales?

#### Business Performance Metrics

- What is the overall profit margin?
- Which categories have the highest profit margins?
- How do sales and profits compare across regions?
- Which products have high sales but low profitability?

#### Business Insights

- Which areas of the business are underperforming?
- What patterns appear in sales and profit?
- What potential business actions are supported
  by the calculated results?

These are example categories and questions.
The final benchmark set will be finalized by the team.

Questions must use metrics and fields that are
actually available in the evaluation dataset.

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
   - Does it avoid unsupported business conclusions?

5. Business Relevance

   - Does the response address the user's question?
   - Are the selected metrics appropriate?
   - Are recommendations supported by the findings?

### Important Distinction

Code that executes successfully can still produce
incorrect numerical results.

For example, an agent might calculate average sales
when the user requested total revenue.

Therefore, execution success and answer accuracy
must be measured separately.

Business interpretations must also be evaluated
for accuracy and relevance.

---

## 7. Development Milestones

### Milestone 1: Basic End-to-End Integration

Goal:

Successfully answer one simple business question
using LLM-generated pandas code.

Example:

    What are the total sales?

Expected process:

- Upload Superstore CSV.
- Send question to mini-swe-agent.
- Generate pandas code.
- Execute code in the isolated environment.
- Return the calculated result.
- Display the answer in Gradio.

### Milestone 2: Basic Business Calculations

Support questions involving:

- Total sales and profit.
- Average sales values.
- Order counts.
- Grouping and filtering.
- Sorting business performance.
- Minimum and maximum values.

The LLM generates the necessary code dynamically.
These are not manually hardcoded calculation functions.

### Milestone 3: Business Performance Analysis

Support:

- Regional sales and profit comparisons.
- Product and category performance.
- Monthly and quarterly sales trends.
- Profit margin calculations.
- Identification of underperforming areas.
- Multi-step business analytical questions.

### Milestone 4: Business Insights and Reliability

Improve:

- Analytical accuracy.
- Error handling.
- Code generation instructions.
- Execution reliability.
- Business-oriented explanations.
- Data-supported recommendations.

The agent should not claim that a business event
was caused by a particular factor unless sufficient
supporting evidence is available.

### Milestone 5: Evaluation and Final Demonstration

- Complete approximately 25 benchmark questions.
- Compare agent answers against expected results.
- Document failures and limitations.
- Evaluate numerical accuracy and business relevance.
- Demonstrate the completed agent through Gradio.

---

## 8. Development Guidelines

- Focus on business data analytics.
- Preserve existing teammate implementations.
- Do not hardcode calculations to Sample Superstore.
- Support other business CSV datasets where possible.
- Inspect dataset columns before performing analysis.
- Do not assume every dataset has identical columns.
- Reuse mini-swe-agent's existing agent loop.
- Never commit API keys or real .env files.
- Keep credentials in environment variables.
- Never run untrusted generated code unrestricted.
- Write tests for each component.
- Agree on component interfaces before integration.
- Keep development focused on the core agent first.
- Prioritize numerical accuracy and reliable explanations.

Optional features, including web research, advanced
visualizations, and external business data integration,
should be considered only after the main agent works
reliably.

---

## 9. Current Project Status

Completed:

- GitHub repository initialized.
- Conda development environment configured.
- Existing Gradio interface created.
- Sample Superstore CSV added.
- Initial Gradio CSV-loading tests created.
- Option A architecture selected.
- Business analytics selected as the project focus.
- Project folders and placeholder files created.
- Architecture documentation prepared.

Not Yet Implemented:

- NRP model integration.
- Agent prompts.
- Mini-swe-agent integration.
- Isolated execution environment.
- Gradio-to-agent connection.
- Business analytics benchmark evaluation system.

The new Python files are currently placeholders.
Their existence does not mean the components
have been implemented.

---

## 10. Next Steps

1. Review the updated business analytics scope as a team.
2. Confirm component ownership.
3. Agree on the interface between Gradio and the runner.
4. Configure LLM integration.
5. Set up and test isolated execution.
6. Integrate the components using mini-swe-agent.
7. Test a simple Superstore business analysis question.
8. Expand to business comparisons and trend analysis.
9. Build and run the business analytics benchmarks.
10. Prepare the final demonstration and evaluation.
