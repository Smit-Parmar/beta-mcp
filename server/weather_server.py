from fastmcp import FastMCP

mcp = FastMCP(name="weather-server")

@mcp.tool()
async def get_weather(city: str) -> str:
    """_summary_
    Get the weather of a city
    """
    return f"The weather of {city} is sunny"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
