# Source: https://logfire.pydantic.dev/docs/integrations/

import logfire

logfire.configure()
logfire.instrument_fastapi()
logfire.instrument_httpx()

# ... your application code here ...
