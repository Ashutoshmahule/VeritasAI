import os
import tempfile

from langchain.agents import create_agent
import requests
from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from schemas import State

load_dotenv()


@tool
def load_pdf(url: str) -> list[Document]:
    """Download a PDF from a URL, parse with PyPDFLoader"""

    resp = requests.get(url)
    resp.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(resp.content)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    marked_docs = []
    for i, doc in enumerate(docs, start=1):
        content = (
            f"\n----- PAGE {i} START -----\n"
            f"{doc.page_content}\n"
            f"----- PAGE {i} END -----\n"
        )
        marked_docs.append(
            Document(page_content=content, metadata={**doc.metadata, "page_number": i})
        )

    return marked_docs


async def deep_cross_reference(state: State):
    llm = ChatOpenAI(model="gpt-5.1", temperature=0)
    client = MultiServerMCPClient(
        {
            "bright_data": {
                "url": f"https://mcp.brightdata.com/sse?token={os.getenv('SCRAPER_MCP')}&pro=1",
                "transport": "sse",
            }
        }
    )
    tools = await client.get_tools()
    research_instructions = """You are an expert researcher. Your job is to conduct thorough research and then write a polished report.
    For each claim, search multiple authoritative sources automatically
    Check news sites, fact-checkers, government sources, academic sources
    For media content, perform reverse image/video searches
    Document all sources and their credibility levels


    then finally:
    Structure your response with these sections:
    ## Post Summary - Describe what the post contained
    Claims Identified - List each claim found",
    Verification Results - Detail findings for each claim with sources",
    Final Verdict - Clear verdict with confidence score (0-100%)",
    """

    agent = create_agent(
        tools=[load_pdf] + tools, system_prompt=research_instructions, model=llm
    )

    result = await agent.ainvoke({"messages": state["messages"][-1]})

    return {"messages": [result["messages"][-1]]}
