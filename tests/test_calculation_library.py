import pandas as pd
import pytest
from agent.analysis_planner import try_planned_analysis
from agent.calculation_library import ratio, CalculationError

@pytest.mark.parametrize('metric,columns,expected', [
    ('roas', {'Revenue':[100,200], 'Advertising_Spend':[20,30]}, 6),
    ('marketing roi', {'Revenue':[100,200], 'Advertising_Spend':[20,30]}, 500),
    ('conversion rate', {'Conversions':[2,8], 'Clicks':[10,90]}, 10),
    ('click through rate', {'Clicks':[10], 'Impressions':[100]}, 10),
    ('customer acquisition cost', {'Acquisition_Cost':[100], 'New_Customers':[4]}, 25),
    ('cost per conversion', {'Advertising_Spend':[100], 'Conversions':[4]}, 25),
    ('inventory turnover', {'COGS':[100], 'Average_Inventory_Value':[20]}, 5),
    ('employee attrition rate', {'Employee_Departures':[2], 'Average_Headcount':[20]}, 10),
    ('net profit margin', {'Net_Profit':[20], 'Revenue':[100]}, 20),
    ('gross profit margin', {'Gross_Profit':[40], 'Revenue':[100]}, 40),
    ('investment roi', {'Investment_Return':[150], 'Investment_Cost':[100]}, 50),
])
def test_ratios(metric, columns, expected):
    assert ratio(pd.DataFrame(columns), metric)[0] == pytest.approx(expected)

@pytest.mark.parametrize('columns', [
    {'Revenue':[100], 'Advertising_Spend':[0]},
    {'Revenue':[100], 'Advertising_Spend':[None]},
    {'Revenue_USD':[100], 'Advertising_Spend_EUR':[20]},
    {'Revenue':[100], 'Sales':[100], 'Advertising_Spend':[20]},
    {'Revenue':['100'], 'Advertising_Spend':[20]},
])
def test_invalid_ratio(columns):
    with pytest.raises(CalculationError):
        ratio(pd.DataFrame(columns), 'roas')

@pytest.mark.parametrize('question,columns,expected', [
    ('Calculate the total inventory value', {'Stock_On_Hand':[2,3], 'Unit_Cost_USD':[10,20]}, '$80.00'),
    ('Calculate employee headcount', {'Employee_ID':['a','a','b']}, '2'),
    ('Calculate repeat purchase rate', {'Customer_ID':['a','a','a','b'], 'Order_ID':['1','1','2','3']}, '50.00%'),
    ('Calculate mean of column X', {'X':[1,2,3]}, '2'),
    ('Calculate sample variance of column X', {'X':[1,2,3]}, '1'),
    ('Calculate 50 percentile of column X', {'X':[1,2,3]}, '2'),
    ('Calculate Kendall correlation of columns X and Y', {'X':[1,2,3], 'Y':[3,2,1]}, '-1.0000'),
    ('Calculate break even units', {'Fixed_Costs':[101], 'Unit_Price':[20], 'Unit_Variable_Cost':[10]}, '11'),
])
def test_planned(question, columns, expected):
    df = pd.DataFrame(columns)
    before = df.copy(deep=True)
    out = try_planned_analysis(df, question)
    assert out['error'] is None
    assert f'**{expected}**' in out['answer']
    pd.testing.assert_frame_equal(df, before)

def test_complex_falls_through():
    assert try_planned_analysis(pd.DataFrame({'X':[1]}), 'Calculate ROAS by region') is None

def test_clarification():
    out = try_planned_analysis(pd.DataFrame({'Revenue':[100]}), 'Calculate ROAS')
    assert 'unique column' in out['answer']

def test_runner_uses_planner_without_model(tmp_path, monkeypatch):
    import agent.agent_runner as runner
    path = tmp_path / 'marketing.csv'
    pd.DataFrame({'Revenue':[100,200], 'Advertising_Spend':[20,30]}).to_csv(path, index=False)
    def forbidden(*args, **kwargs):
        pytest.fail('Unexpected general AI route')
    monkeypatch.setattr(runner, 'run_agent', forbidden)
    out = runner.run_analysis(str(path), 'Calculate ROAS', explain=True)
    assert out['error'] is None
    assert '**6**' in out['answer']
