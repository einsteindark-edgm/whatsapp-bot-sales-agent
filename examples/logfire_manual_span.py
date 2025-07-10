# Source: https://logfire.pydantic.dev/docs/reference/api/logfire/

import logfire

logfire.configure()

with logfire.span("This is a span {a=}", a="data"):
    logfire.info("new log 1")
