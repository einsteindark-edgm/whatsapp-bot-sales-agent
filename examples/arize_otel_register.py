# Source: https://arize.com/docs/ax/integrations/opentelemetry/opentelemetry-arize-otel

from arize.otel import register

# Setup OTel via Arize convenience function
tracer_provider = register(
    # See details in docs…
)

# Instrument your application using OpenInference Auto-Instrumentors
from openinference.instrumentation.openai import OpenAIInstrumentor

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
