import sys
import os

import google.generativeai as genai

from google.generativeai import (
    GenerativeModel,
)

import argparse


def generate_client():

    system_instruction = [
        "User will provide instruction and context where this instruction will be applied. Output will be send to a bash pipe so your output should NOT have any explanations or anything, jsut required operations.",
	    "if you asked to update something most likely you need to show same data stracture but updated, in the same format. Do not show HOW to update data structure, update it and output the result.",
	    "Here is how you might be used by user: cat file | you -p \"prompt\" >> result"
    ]
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    return GenerativeModel(model_name="gemini-1.5-pro", system_instruction=system_instruction)

gemini_model = generate_client()


def main():
    parser = argparse.ArgumentParser(description='Toolbox for using Gemini Agents SDK.')
    parser.add_argument('-p', '--prompt', type=str, required=True, help='The prompt for the Gemini model.')
    args = parser.parse_args()

    lines = "".join([line for line in sys.stdin])

    msg = f"""User provided prompt: {args.prompt}
---
User provided content:
{lines}
    """
    print(gemini_model.generate_content(msg).text)
