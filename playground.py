from phi.agent import Agent
import phi.api
import phi.api.playground
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.groq import Groq
from dotenv import load_dotenv

import os
import phi
from phi.playground import Playground, serve_playground_app

load_dotenv()

phi.api= os.getenv("PHI_API_KEY")

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

app=Playground(agents=[agent_finance, agent_search]).get_app()


if __name__ == "__main__":
    serve_playground_app("playground:app",reload=True)