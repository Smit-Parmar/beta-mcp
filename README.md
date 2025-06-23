MCP Server

#Generate readme.md for this project
# MCP Server Project

This project demonstrates the use of Model Context Protocol (MCP) servers with multiple transport types. It includes a math server using stdio transport and a weather server using HTTP transport, both integrated with a Groq-powered client.

## Project Structure

```
beta-mcp/
├── client/
│   └── groq_client.py          # Main client that connects to MCP servers
├── server/
│   ├── math_server.py          # Math operations server (stdio transport)
│   └── weather_server.py       # Weather information server (HTTP transport)
├── main.py                     # Entry point for the application
├── pyproject.toml              # Python project configuration
├── requirements.txt            # Python dependencies
├── uv.lock                     # UV lock file for dependency management
└── README.md                   # This file
```

### Key Components

- **`client/groq_client.py`**: Multi-server MCP client that connects to both math and weather servers using different transport protocols
- **`server/math_server.py`**: MCP server providing mathematical operations via stdio transport
- **`server/weather_server.py`**: MCP server providing weather information via HTTP transport
- **`main.py`**: Application entry point
- **`pyproject.toml`**: Project configuration and metadata
- **`requirements.txt`**: Python package dependencies

## Installation

Install dependencies using uv:
```bash
uv add -r requirements.txt
```

## Usage

### Running the MCP Client
```bash
python client/groq_client.py
```

### Running the Yahoo Finance Example
To see a comprehensive demonstration of Yahoo Finance API usage:
```bash
python examples/yahoo_finance_example.py
```

This example shows:
- Basic stock information retrieval
- Historical data analysis
- Financial statement access
- Dividend information
- Multi-stock comparison
- Options data (if available)
- Real-time price checking