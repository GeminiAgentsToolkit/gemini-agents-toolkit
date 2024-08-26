import google.generativeai as genai
from gemini_toolbox import client
from dotenv import load_dotenv

import os

def get_current_time():
    """returns current time"""
    return "6pm PST"


def say_to_duck(say: str):
    """say something to a duck"""
    return f"duck answer is: duck duck {say} duck duck duck"

load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

all_functions = [get_current_time, say_to_duck]
clt = client.generate_chat_client_from_functions_list(all_functions)

print(clt.send_message("say to the duck message: I am hungry"))

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input == "exit":
            break
        response = clt.send_message(user_input)
        print("Jessica:", response)
