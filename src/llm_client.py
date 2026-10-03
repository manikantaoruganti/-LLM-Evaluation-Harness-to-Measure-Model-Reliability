import os
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class LLMClient:
    """
    A client for interacting with the OpenAI LLM API.
    """
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.system_prompt = (
            "You are answering benchmark questions. Answer the question accurately and directly. "
            "Do not invent information. Give the final answer clearly."
        )

    def generate_response(self, prompt: str) -> str:
        """
        Generates a text response from the LLM for a given prompt.

        Args:
            prompt (str): The input prompt for the LLM.

        Returns:
            str: The LLM's text response.

        Raises:
            OpenAIError: If there's an issue with the OpenAI API call.
            Exception: For other unexpected errors.
        """
        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,  # Ensure deterministic responses for benchmarking
                max_tokens=150 # Limit response length to keep it concise
            )
            return chat_completion.choices[0].message.content.strip()
        except OpenAIError as e:
            print(f"OpenAI API Error: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during LLM response generation: {e}")
            raise

if __name__ == "__main__":
    # Example usage (requires OPENAI_API_KEY in .env)
    try:
        llm_client = LLMClient()
        test_prompt = "What is the capital of France?"
        print(f"Sending prompt: '{test_prompt}'")
        response = llm_client.generate_response(test_prompt)
        print(f"LLM Response: '{response}'")

        test_prompt_2 = "What is 2 + 2?"
        print(f"Sending prompt: '{test_prompt_2}'")
        response_2 = llm_client.generate_response(test_prompt_2)
        print(f"LLM Response: '{response_2}'")

    except ValueError as e:
        print(f"Configuration Error: {e}")
    except OpenAIError as e:
        print(f"API Error during example run: {e}")
    except Exception as e:
        print(f"General Error during example run: {e}")
