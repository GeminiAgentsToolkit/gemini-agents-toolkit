def summarize(*, agent, history):
    """Summarize the pipeline"""
    prompt = f"""Now if the final step of the pipeline/dialog, provide summary of main things that were done and why.
        do not omit any steps, and only print key details. This dialog was a pipline so do not assume user knows about
        any messages even if they were comming from the user before, now is the time to build proper summary for a user."""
    return agent.send_message(prompt, history=history)