from gemini_agents_toolkit import agent

import vertexai


vertexai.init(project="gemini-trading-backend", location="us-west1")


def generate_duck_comms_agent():
    def say_to_duck(say: str):
        """say something to a duck"""
        return f"duck answer is: duck duck {say} duck duck duck"
    return agent.create_agent_from_functions_list(
        functions=[say_to_duck], 
        delegation_function_prompt="Agent can communicat to ducks and can say something to them. And provides the answer from the duck.", 
        model_name="gemini-1.5-flash")


def generate_time_checker_agent():
    def get_local_time():
        """get the current local time"""
        import datetime
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return agent.create_agent_from_functions_list(
        functions=[get_local_time], 
        delegation_function_prompt="Agent can provide the current local time.", 
        model_name="gemini-1.5-flash")


duck_comms_agent = generate_duck_comms_agent()
time_checker_agent = generate_time_checker_agent()

main_agent = agent.create_agent_from_functions_list(delegates=[time_checker_agent, duck_comms_agent], model_name="gemini-1.5-flash")

print(main_agent.send_message("say to the duck message: I am hungry"))
print(main_agent.send_message("can you tell me what time it is?"))
