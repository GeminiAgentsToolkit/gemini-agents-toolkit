from vertexai.generative_models import (
    Content,
    Part
)


def summarize(*, agent, history):
    """Summarize the pipeline"""
    prompt = """Now if the final step of the pipeline/dialog, provide summary of main things that were done and why.
        do not omit any steps, and only print key details. This dialog was a pipline so do not assume user knows about
        any messages even if they were coming from the user before, now is the time to build proper summary for a user."""
    return agent.send_message(prompt, history=history)


def calculate_total_tokens_used_per_model(*, history):
    """Calculate the total number of tokens used in the history"""
    tokens_per_model = {}
    for h in history:
        tokens_per_model.update(h.get("tokens_used", {}))
    return tokens_per_model


def trim_history(*, history, max_length):
    """Trim history to only include the last specified number of user messages"""
    if len(history) <= max_length:
        return history
    
    trimmed_history = []
    user_messages_count = 0

    # Iterate through the history in reverse
    for h in reversed(history):
        trimmed_history.append(h)
        if h["raw"].role == "user":
            user_messages_count += 1
            if user_messages_count == max_length:
                break

    return list(reversed(trimmed_history))


def print_history(history):
    """Print the history"""
    if not history:
        return
    
    for h in history:
        raw_h = h["raw"]
        if hasattr(raw_h.parts[0], "text"):
            print(f"{raw_h.role}: {raw_h.parts[0].text}")
        if hasattr(h, "function_call"):
            print(f"Function called: {h.function_call.name}")


def to_serializable_list(history):
    result_history = []
    for h in history:
        raw_h = h["raw"]
        if hasattr(raw_h.parts[0], "text"):
            result_history.append(
                {
                    "role": raw_h.role,
                    "text": raw_h.parts[0].text
                }
            )
    return result_history


def from_serializable_list(list_with_history):
    history = []
    for h in list_with_history:
        history.append(
            {
                'raw': Content(
                    role=h["role"],
                    parts=[Part.from_text(h["text"])]
                )
            }
        )
    return history
