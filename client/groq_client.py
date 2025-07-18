from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from typing import cast, Dict, Any

from dotenv import load_dotenv
load_dotenv()

import asyncio

async def main():
    connections = {
        "math": {
            "command": "python",
            "args": ["server/math_server.py"],
            "transport": "stdio",
        },
        "weather": {
            "url": "http://localhost:8000/mcp",
            "transport": "streamable_http",
        },
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

    math_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
    )

    print("Math response:", math_response['messages'][-1].content)

    weather_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what is the weather in California?"}]}
    )
    print("Weather response:", weather_response['messages'][-1].content)

    stock_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what is the current stock price of AAPL?"}]}
    )
    print("Stock response:", stock_response['messages'][-1].content)

asyncio.run(main())