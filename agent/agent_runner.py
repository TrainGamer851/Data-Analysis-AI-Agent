import pandas as pd
from agent.prompts import SYSTEM_PROMPT, build_analysis_prompt
from agent.execution_env import execute_code
from agent.llm_client import generate_response
from minisweagent.agents.default import DefaultAgent



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