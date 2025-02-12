import os
from openai import OpenAI, AsyncOpenAI


def get_content_between_a_b(start_tag, end_tag, text):
    extracted_text = ""
    start_index = text.find(start_tag)
    while start_index != -1:
        end_index = text.find(end_tag, start_index + len(start_tag))
        if end_index != -1:
            extracted_text += text[start_index + len(start_tag) : end_index] + " "
            start_index = text.find(start_tag, end_index + len(end_tag))
        else:
            break

    return extracted_text.strip()


def extract(text, type):
    if text:
        target_str = get_content_between_a_b(f"<{type}>", f"</{type}>", text)
        if target_str:
            return target_str
        else:
            return text
    else:
        return ""


class LLM_client:
    def __init__(self, model_name: str = None) -> None:
        self.model = model_name

        if self.model == "deepseek-chat":
            self.api_key = os.environ.get("DEEPSEEK_API_KEY", None)
            self.base_url = os.environ.get("DEEPSEEK_BASE_URL", None)
        else:
            self.api_key = os.environ.get("OPENAI_API_KEY", None)
            self.base_url = os.environ.get("OPENAI_BASE_URL", None)

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def response(self, messages, **kwargs):
        try:
            response = self.client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=messages,
                n=kwargs.get("n", 1),
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 4000),
                timeout=kwargs.get("timeout", 180),
            )
        except Exception as e:
            model = kwargs.get("model", self.model)
            print(f"get {model} response failed: {e}")
            return None

        return response.choices[0].message.content

    async def async_response(self, messages, **kwargs):
        try:
            response = await self.async_client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=messages,
                n=kwargs.get("n", 1),
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 4000),
                timeout=kwargs.get("timeout", 180),
            )
        except Exception as e:
            model = kwargs.get("model", self.model)
            print(f"get {model} async response failed: {e}")
            return None

        return response.choices[0].message.content
