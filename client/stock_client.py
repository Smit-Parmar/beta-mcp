from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from typing import cast, Dict, Any

from dotenv import load_dotenv
load_dotenv()

import asyncio

async def main():
    connections = {
        "stock": {
            "command": "python",
            "args": ["server/stock_server.py"],
            "transport": "stdio",
        }
    }
    
    client = MultiServerMCPClient(cast(Dict[str, Any], connections))

    import os
    os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY") or ""

    tools=await client.get_tools()
    model=ChatGroq(model="qwen-qwq-32b")
    agent=create_react_agent(
        model,tools
    )

    stock_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is historical data of RELIANCE.NS in last 30 days? what was the highest price in the last 30 days?"}]}
    )
    print("Stock response:", stock_response['messages'][-1].content)

asyncio.run(main())