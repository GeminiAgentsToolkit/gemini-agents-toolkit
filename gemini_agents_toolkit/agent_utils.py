def start_debug_chat(*, agent, history):
    """Start a debug chat from the history"""
    if history:
        agent.set_history(history)
        full_history = history
    else:
        full_history = []
    while True:
        user_input = input("You: ")
        if user_input == "exit":
            break
        response, h = agent.send_message(user_input, history=full_history)
        full_history += h
        print("Agent: " + response)