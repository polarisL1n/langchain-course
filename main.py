import json
from pydantic import BaseModel, Field

from dotenv import load_dotenv

_ = load_dotenv()

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openrouter import ChatOpenRouter
from tavily import TavilyClient
from langchain_tavily import TavilySearch

tavily = TavilyClient()

class Source(BaseModel):
  """Schema for a source used by the agent"""

  url:str = Field(description="The url of the source")

class AgentResponse(BaseModel):
  """Schema for the agent response with answer and sources"""
  answer: str = Field(description="The agent's answer to the query")
  sources: list[Source] = Field(default_factory=list, description="The sources used to generate the answer")

@tool
def search(query: str) -> str:
    """
    Tool that searches over the internet
    Args:
        query: the search query
    Returns:
        the search results
    """
    print(f"searching for {query}")
    response = tavily.search(query=query)
    return json.dumps(response["results"], ensure_ascii=False)

llm = ChatOpenRouter(
  model="openai/gpt-5.4",
)
tools = [TavilySearch()]
agent = create_agent(
  model=llm,
  tools=tools,
  response_format=ToolStrategy(AgentResponse)
)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({
      "messages": [HumanMessage(content="Search 3 jobs about Agent development in Beijing or Shanghai")]
    })
    print(result)


if __name__ == "__main__":
    main()
