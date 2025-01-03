from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools 
from phi.tools.duckduckgo import DuckDuckGo

# Web Search Agent
agent_search = Agent(
    name="web_search_agent",
    role="Search the web for the information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=["Always include Sources"],
    show_tools_calls=True,
    markdown=True,
)

# Financial Agent
agent_finance = Agent( 
    name="Finance AI agent",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions=["Use tables to display the Data"],
    show_tools_calls=True,
    markdown=True,
)

# Multi-Agent System
multi_ai_agent=Agent(
    team=[agent_search,agent_finance],
    model=Groq(id="llama-3.1-70b-versatile"),
    instructions=["Use the web search agent to find information about the company and the financial agent to find information about the stock.", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)
multi_ai_agent.print_response("Summarize analyst recommendations and share the latest news for NVDA", stream=True)