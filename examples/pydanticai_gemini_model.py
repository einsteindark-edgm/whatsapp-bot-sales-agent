# Source: https://raw.githubusercontent.com/pydantic/pydantic-ai/main/docs/models/gemini.md
### Configuration

# `GeminiModel` lets you use Google's Gemini models through their [Generative Language API](https://ai.google.dev/api/all-methods), `generativelanguage.googleapis.com`.

#`GeminiModelName` contains a list of available Gemini models that can be used through this interface.

#To use `GeminiModel`, go to [aistudio.google.com](https://aistudio.google.com/apikey) and select "Create API key".

### Environment variable

#Once you have the API key, you can set it as an environment variable:

# export GEMINI_API_KEY=your-api-key

# You can then use `GeminiModel` by name:


from pydantic_ai import Agent

agent = Agent('google-gla:gemini-2.0-flash')


#!!! note
#    The `google-gla` provider prefix represents the [Google **G**enerative **L**anguage **A**PI](https://ai.google.dev/api/all-methods) for `GeminiModel`s.
#    `google-vertex` is used with [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models).

#Or initialise the model directly with just the model name and provider:


from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel

model = GeminiModel('gemini-2.0-flash', provider='google-gla')
agent = Agent(model)
