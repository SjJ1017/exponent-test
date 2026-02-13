import os
import hashlib
import random
from typing import Literal

def llm(
    prompt: str, 
    error: Literal["raise", "random", "fixed"] = "raise"
) -> str:
    """
    Call OpenAI API to generate response
    
    Args:
        prompt: llm input
        error: 
            - "raise": raise error
            - "random": random response according to the prompt
            - "fixed": fixed response
    
    return:
        str: LLM response
    """

    def _get_deterministic_random_response(prompt: str) -> str:
        response_pool = [
            f"This is a fallback response {i}." for i in range(50)
        ]

        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        seed = int(prompt_hash, 16) % (2**32)
        
        rng = random.Random(seed)
        selected_response = rng.choice(response_pool)
        # Ensure the prompt has influence on the response (for testing retrieved documents already in prompt)
        prompt_prefix = prompt[:30] + "..." if len(prompt) > 30 else prompt
        return f"[Mock for '{prompt_prefix}']: {selected_response}"

    try:
        # Try calling OpenAI API
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not found")
        
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=1000
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("API returned None content")

        return content
    
    except Exception as e:
        if error == "raise":
            return f"Error: {type(e).__name__}: {str(e)}"
        else:
            print("Fails to use openai api, responsing as pre-defined...")
            if error == "random":
                return _get_deterministic_random_response(prompt)
            elif error == "fixed":
                return "API call failed. Using fallback response."
            else:
                return f"Error: Invalid error handling mode '{error}'"
        
if __name__ == '__main__':
    print(llm('hi', error='random'))