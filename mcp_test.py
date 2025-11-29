# Construct server URL with authentication
from urllib.parse import urlencode

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

base_url = "https://server.smithery.ai/@luminati-io/brightdata-mcp/mcp"
params = {
    "api_key": "74ec252d-1440-484b-af6f-f832bce2783d",
    "profile": "unlikely-rat-r6Jka7",
}
url = f"{base_url}?{urlencode(params)}"


async def main():
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # List available tools
            tools_result = await session.list_tools()
            print(f"Available tools: {', '.join([t.name for t in tools_result.tools])}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
