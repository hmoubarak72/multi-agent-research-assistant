from dataclasses import dataclass
from typing import Type, TypeVar
from anthropic import AsyncAnthropic, APIConnectionError, APITimeoutError, RateLimitError
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

T = TypeVar("T", bound=BaseModel)


@dataclass
class StructuredLLMResponse:
    parsed: BaseModel
    input_tokens: int
    output_tokens: int


class LLMService:
    """
    Executes Claude API calls using forced tool-calling to guarantee
    Pydantic-validated structured outputs and exact token accounting.
    """

    def __init__(self, api_key: str, model: str):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model

    @retry(
        retry=retry_if_exception_type(
            (APITimeoutError, APIConnectionError, RateLimitError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1.5, max=10.0),
        reraise=True,
    )
    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: Type[T],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> StructuredLLMResponse:
        tool_name = f"emit_{output_schema.__name__.lower()}"
        tool_definition = {
            "name": tool_name,
            "description": f"Emit structured output matching {output_schema.__name__}.",
            "input_schema": output_schema.model_json_schema(),
        }

        message = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[tool_definition],
            tool_choice={"type": "tool", "name": tool_name},
        )

        tool_payload = None
        for block in message.content:
            if block.type == "tool_use" and block.name == tool_name:
                tool_payload = block.input
                break

        if tool_payload is None:
            raise ValueError(
                f"Claude did not invoke required structured tool '{tool_name}'."
            )

        validated_obj = output_schema.model_validate(tool_payload)
        return StructuredLLMResponse(
            parsed=validated_obj,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )
