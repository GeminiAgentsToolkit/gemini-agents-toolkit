def summarize(*, agent, history):
    """Summarize the pipeline"""
    prompt = f"""Now if the final step of the pipeline/dialog, provide summary of main things that were done and why.
        do not omit any steps, and only print key details. This dialog was a pipline so do not assume user knows about
        any messages even if they were comming from the user before, now is the time to build proper summary for a user."""
    return agent.send_message(prompt, history=history)


def trim_history(*, history, max_length):
    """Trim history to only include the last specfified number of user messages"""
    if len(history) <= max_length:
        return history
    
    trimmed_history = []
    user_messages_count = 0

    # Iterate through the history in reverse
    for h in reversed(history):
        trimmed_history.append(h)
        if h.role == "user":
            user_messages_count += 1
            if user_messages_count == max_length:
                break

    return list(reversed(trimmed_history))