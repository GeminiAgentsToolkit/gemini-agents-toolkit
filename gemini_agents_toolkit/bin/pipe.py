"""Configure the client for requesting Gemini models from the toolkit"""

import sys
import argparse
import google.generativeai as genai
from google.generativeai import (GenerativeModel)
from config import (API_KEY, DEFAULT_MODEL)


def generate_client():
    """Setting up the client"""
    system_instruction = [
        """User will provide instruction and context where this instruction will be applied.
        Output will be send to a bash pipe,
        so your output should NOT have any explanations or anything, just required operations.""",
	    """if you asked to update something, most likely you need to show same data structure,
	     but updated, in the same format.
	     Do not show HOW to update data structure, update it and output the result.""",
	    "Here is how you might be used by user: cat file | you -p \"prompt\" >> result"
    ]
    genai.configure(api_key=API_KEY)

    return GenerativeModel(model_name=DEFAULT_MODEL, system_instruction=system_instruction)

gemini_model = generate_client()


def main():
    """Initiating Gemini client"""
    parser = argparse.ArgumentParser(description='Toolbox for using Gemini Agents SDK.')
    parser.add_argument('-p', '--prompt', type=str, required=True,
                        help='The prompt for the Gemini model.')
    args = parser.parse_args()

    lines = "".join(list(sys.stdin))

    msg = f"""User provided prompt: {args.prompt}
---
User provided content:
{lines}
    """
    print(gemini_model.generate_content(msg).text)
