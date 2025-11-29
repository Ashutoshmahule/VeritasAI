import os
import pprint

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.prompts.chat import MessageLike
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from schemas import State

load_dotenv()


async def extraction_agent(state: State):
    client = MultiServerMCPClient(
        {
            "bright_data": {
                "url": f"https://mcp.brightdata.com/sse?token={os.getenv('SCRAPER_MCP')}&pro=1",
                "transport": "sse",
            }
        }
    )
    tools = await client.get_tools()
    print("Available tools:", [tool.name for tool in tools])
    llm = ChatOpenAI(model="gpt-5.1", temperature=0)
    system_prompt = """
    Extract comprehensive data from social media posts
    You MUST use the available tools to extract data from social media posts
    For other platforms or if structured data fails: Use scrape_as_markdown tool
    Extract: post text, media URLs, user info, engagement metrics, timestamps
    """

    agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

    print("Testing ReAct Agent with available tools...")
    print("=" * 50)

    result = await agent.ainvoke({"messages": [state["messages"][-1]]})

    print("\nAgent Run Messages:")
    for message in result["messages"]:
        pprint.pprint(message)
    return {
        "extracted_info": result["messages"][-1],
        "messages": result["messages"],
    }
