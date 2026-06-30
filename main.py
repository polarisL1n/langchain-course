import json

from dotenv import load_dotenv

_ = load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek
from tavily import TavilyClient
from langchain_tavily import TavilySearch

tavily = TavilyClient()

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

llm = ChatDeepSeek(
  model="deepseek-v4-pro",
)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({
      "messages": [HumanMessage(content="What is the weather in Tokyo")]
    })
    print(result)


if __name__ == "__main__":
    main()
