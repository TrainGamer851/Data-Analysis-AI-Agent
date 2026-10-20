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

Enable the agent to safely execute Python/pandas code and explain calculated results.

### Stage 4 — Testing and improvements

Use benchmark questions, evaluate accuracy, fix failures, and improve the application.

## Stage 1: What we are working on now

**Goal:** A user can open the application, upload a CSV file, and see its data in a table.

Our tasks:

1. Create the basic Gradio interface.
2. Add a CSV upload component.
3. Read the uploaded CSV using pandas.
4. Display the dataset in the interface.

Stage 1 is complete when this upload-to-table flow works. AI questions and answers will
come in later stages.

### Current files

```text
Data-Analysis-AI-Agent/
├── README.md      # Project overview and development stages
├── app.py         # Stage 1 application code, to be written by the team
└── .gitignore     # Keeps local environment files out of Git
```

**app.py** is intentionally empty. For Stage 1, the team will write the Gradio interface,
CSV loading, and table display in this one file. We will add more files when later stages
need them.

**Current status:** Project files are set up. Stage 1 coding has not started.
