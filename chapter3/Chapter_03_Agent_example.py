from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

def check_interest_rate(rate: float) -> bool:                       #A
    # BFSI guardrail: interest rate must not exceed 30%
    return (0 <= rate <= 0.30)

def propose_credit_line(user_income: float):
    # Simple BFSI logic
    return user_income * 4

tools = [
    Tool(
        name="rate_check",
        func=check_interest_rate,
        description="Validates interest rates for BFSI compliance"
    ),
    Tool(
        name="credit_line",
        func=propose_credit_line,
        description="Proposes a credit line based on user income"
    ),
]

llm = OpenAI(temperature=0.0)
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    max_iterations=3,                                              #B
    handle_parsing_errors=True
)

prompt_text = """
User income: 50000.
Propose a credit line and confirm the interest rate is BFSI-compliant.
"""

result = agent.run(prompt_text)                                     #C
print("Agent Output:\n", result)                                    #D
