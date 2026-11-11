from logging import config

import pandas as pd
from agent.prompts import SYSTEM_PROMPT, build_analysis_prompt
from agent.execution_env import execute_code
from agent.llm_client import generate_response, get_model_config
from minisweagent.agents.default import DefaultAgent


# PART 1: ENVIRONMENT ADAPTER

class DataAnalysisEnvironment:
    def __init__(self, df):
        self.df = df

    def execute(self, action: dict, cwd: str = "") -> dict:
        result = execute_code(action["code"], self.df)
        
        return {
            "output": str(result["result"]) if result["success"] else "",
            "returncode": 0 if result["success"] else 1,
            "exception_info": result["error"] or ""
        }

    def get_template_vars(self, **kwargs) -> dict:
        return {}

    def serialize(self) -> dict:
        return {"type": "data_analysis"}



def run_analysis(csv_file: str, question: str):
    df = pd.read_csv(csv_file)

    prompt = build_analysis_prompt(question, df)


# PART 2: MODEL ADAPTER

class QwenModelAdapter:
    def query(self, messages, **kwargs):
        # Send messages to Qwen using our teammate's function.
        api_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
            if msg["role"] in ("system", "user", "assistant")
        ]

        code = generate_response(api_messages)

        if not code or not code.strip():
            raise ValueError("Qwen returned no Python code.")

        # Give mini-swe-agent the generated code as an action.
        return {
            "role": "assistant",
            "content": code,
            "extra": {
                "actions": [{"code": code}],
                "cost": 0.0
            }
        }

    def format_message(self, **kwargs):
        return kwargs

    def format_observation_messages(
        self, message, outputs, template_vars=None
    ):
        output = outputs[0]

        # If the calculation succeeded, finish the task.
        if output["returncode"] == 0:
            return [{
                "role": "exit",
                "content": output["output"],
                "extra": {
                    "exit_status": "Submitted",
                    "submission": output["output"]
                }
            }]

        # If the calculation failed, ask Qwen to correct its code.
        return [{
            "role": "user",
            "content": (
                "Your Python code failed:\n"
                f"{output['exception_info']}\n"
                "Generate corrected Python code."
            )
        }]

    def get_template_vars(self, **kwargs):
        return {}


    def serialize(self):
        config = get_model_config()

        return {
        "info": {
            "config": {
                "model_type": "QwenModelAdapter",
                "model": config["model"]
            }
        }
    }




