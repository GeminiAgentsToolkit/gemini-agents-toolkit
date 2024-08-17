class LLMTask(object):

    def __init__(self, prompt, *, precondition_prompt, negative_prompt) -> None:
        self.prompt = prompt
        self.precondition_prompt = precondition_prompt
        self.negative_prompt = negative_prompt