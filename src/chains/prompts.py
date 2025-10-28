from __future__ import annotations

from typing import List

try:
    from langchain.prompts import PromptTemplate
except Exception:  # minimal fallback type
    class PromptTemplate:  # type: ignore
        def __init__(self, input_variables: List[str], template: str):
            self.input_variables = input_variables
            self.template = template


# Few-shot exemplars for better code generation reliability
QUERY_FEWSHOTS = """
Examples:
Q: Which merchants had the highest total revenue last month?
A (pandas):
result = (df[df['timestamp'] >= df['timestamp'].max() - pd.to_timedelta(30, 'D')]
            .groupby('merchant')['amount'].sum()
            .sort_values(ascending=False)
            .reset_index().head(10))

Q: What is the average transaction amount by country?
A (pandas):
result = df.groupby('country')['amount'].mean().reset_index().sort_values('amount', ascending=False)
"""


sql_prompt_template = PromptTemplate(
    input_variables=["question"],
    template=(
        "Translate the user question into safe Pandas code operating on a DataFrame named df.\n"
        "Rules: Only use pandas/numpy, no imports, no file I/O, no network, no os/sys.\n"
        "Assign the final table to a variable named 'result'.\n\n"
        f"{QUERY_FEWSHOTS}\n\n"
        "Question: {question}\nCode:"
    ),
)


SUMMARY_FEWSHOTS = """
Examples:
Question: Which merchants had the highest total revenue last month?
Result (head):
merchant,amount\nAmazon,124000\nUber,96000
Summary: Amazon led last month with $124.0K in revenue, followed by Uber at $96.0K.
"""

summary_prompt_template = PromptTemplate(
    input_variables=["question", "result"],
    template=(
        "You are a data analyst. Summarize the following query result in plain English.\n"
        "Be concise and mention key numbers.\n\n"
        f"{SUMMARY_FEWSHOTS}\n\n"
        "Question: {question}\nResult: {result}\nSummary:"
    ),
)


ROUTER_FEWSHOTS = """
Examples:
Q: What are the refund rules under $50?\nA: policy
Q: List suspicious late-night transactions this week.\nA: fraud
Q: Which merchants grew fastest this quarter?\nA: data
"""

router_prompt_template = PromptTemplate(
    input_variables=["question"],
    template=(
        "Classify the question into one of: data, fraud, policy.\n"
        "If the question asks about policies, rules, or guidelines → policy.\n"
        "If it asks to detect anomalies or fraud → fraud. Otherwise → data.\n\n"
        f"{ROUTER_FEWSHOTS}\n\n"
        "Question: {question}\nClass:"
    ),
)


