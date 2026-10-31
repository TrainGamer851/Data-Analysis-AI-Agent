import pandas as pd
import pytest

from agent.execution_env import execute_code

# Mock data
@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "Region": ["West", "East", "West", "South"],
            "Sales": [100.0, 200.0, 50.5, None],
            "Order Date": pd.to_datetime(["2024-01-01", "2024-02-01", None, "2024-03-01"]),
        }
    )

# functionality
def test_scalar_result(df):
    out = execute_code("result = df['Sales'].sum()", df)
    assert out["success"] and out["result"] == 350.5
    assert out["result_type"] == "scalar"

def test_fenced_code(df):
    out = execute_code("```python\nresult = len(df)\n```", df)
    assert out["success"] and out["result"] == 4

def test_groupby_keeps_index_as_column(df):
    out = execute_code("result = df.groupby('Region')['Sales'].sum()", df)
    assert out["success"] and out["result_type"] == "table"
    assert {"Region": "West", "Sales": 150.5} in out["result"]

def test_timestamps_and_nan_serialize(df):
    out = execute_code("result = df", df)
    assert out["success"]
    assert out["result"][0]["Order Date"].startswith("2024-01-01")
    assert out["result"][2]["Order Date"] is None
    assert out["result"][3]["Sales"] is None

def test_table_truncation(df):
    out = execute_code("result = pd.DataFrame({'x': range(1000)})", df, max_rows=50)
    assert out["success"] and len(out["result"]) == 50

def test_comprehension_sees_top_level_names(df):
    code = "cols = ['Sales']\nresult = [c for c in df.columns if c in cols]"
    assert execute_code(code, df)["result"] == ["Sales"]

def test_allowed_import_and_exception_classes(df):
    code = (
        "import pandas as pd\nimport math\n"
        "try:\n    df['nope']\nexcept KeyError:\n    result = math.floor(2.7)"
    )
    out = execute_code(code, df)
    assert out["success"] and out["result"] == 2

def test_unique_values_serialize(df):
    out = execute_code("result = df['Region'].unique()", df)
    assert out["success"] and out["result"] == ["West", "East", "South"]

def test_stdout_captured_and_capped(df):
    out = execute_code("print('hi')\nresult = 1", df)
    assert out["stdout"] == "hi\n"
    big = execute_code("print('x' * 1_000_000)\nresult = 1", df)
    assert big["success"] and len(big["stdout"]) < 20_000

# error feedback tests
def test_syntax_error(df):
    out = execute_code("result = (", df)
    assert not out["success"] and "SyntaxError" in out["error"]

def test_runtime_error_has_line_and_columns(df):
    out = execute_code("x = 1\nresult = df['Profit'].sum()", df)
    assert not out["success"]
    assert "KeyError" in out["error"] and "line 2" in out["error"]
    assert "Available columns" in out["error"] and "Sales" in out["error"]

def test_missing_result(df):
    out = execute_code("x = 5", df)
    assert not out["success"] and "result" in out["error"]

def test_empty_inputs(df):
    assert not execute_code("", df)["success"]
    assert not execute_code("result = 1", None)["success"]
    assert not execute_code("result = 1", pd.DataFrame())["success"]

def test_timeout(df):
    out = execute_code("while True:\n    pass", df, timeout=2)
    assert not out["success"] and "timed out" in out["error"]


# Security tests: if restrictions actually work
def test_disallowed_import(df):
    out = execute_code("import os\nresult = 1", df)
    assert not out["success"] and "not allowed" in out["error"]

def test_cannot_read_arbitrary_files(df):
    out = execute_code("result = pd.read_csv('/etc/passwd').shape", df)
    assert not out["success"] and "Blocked" in out["error"]

def test_cannot_write_files(df, tmp_path):
    target = tmp_path / "leak.csv"
    out = execute_code(f"df.to_csv(r'{target}')\nresult = 1", df)
    assert not out["success"] and not target.exists()

def test_cannot_delete_files(df, tmp_path):
    target = tmp_path / "keep.txt"
    target.write_text("x")
    out = execute_code(f"pd.io.common.os.remove(r'{target}')\nresult = 1", df)
    assert not out["success"] and target.exists()

def test_secrets_not_inherited(df, monkeypatch):
    monkeypatch.setenv("NRP_API_KEY", "super-secret")
    out = execute_code("result = pd.io.common.os.environ.get('NRP_API_KEY')", df)
    assert out["success"] and out["result"] is None

def test_cannot_read_proc_environ(df):
    out = execute_code("result = pd.read_csv('/proc/self/environ').shape", df)
    assert not out["success"] and "Blocked" in out["error"]

def test_network_blocked(df):
    out = execute_code("result = pd.read_csv('http://127.0.0.1:9/x.csv').shape", df)
    assert not out["success"] and "Blocked" in out["error"]

def test_subprocess_blocked(df):
    out = execute_code("result = pd.io.common.os.system('echo pwned')", df)
    assert not out["success"] and "Blocked" in out["error"]

def test_parent_df_not_mutated(df):
    before = df.copy()
    execute_code("df.drop(df.index, inplace=True)\nresult = 1", df)
    pd.testing.assert_frame_equal(df, before)