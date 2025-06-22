from fastmcp import FastMCP
# Create a new MCP server
mcp = FastMCP(name="math-server")

# Define a function to add two numbers
@mcp.tool()
def add(a: int, b: int) -> int:
    """_summary_
    Add two numbers
    Args:
        a (int): number a
        b (int): number b

    Returns:
        int: sum of a and b
    """
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """_summary_
    Multiply two numbers
    Args:
        a (int): number a
        b (int): number b

    Returns:
        int: product of a and b
    """
    return a * b

if __name__ == "__main__":
    """
    transport="stdio" is a transport that allows the server to be run in the terminal and communicate with the client
    """
    mcp.run(transport="stdio")