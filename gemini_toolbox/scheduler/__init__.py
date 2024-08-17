from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from gemini_toolbox.scheduler.task import LLMTask


class ScheduledTaskExecutor:
    
    def __init__(self, debug=False):
        """
        Initializes the ScheduledTaskExecutor with the given GeminiChatClient instance.
        """
        self.scheduler = BackgroundScheduler()
        self.debug = debug
        self.gemini_chat_client = None

    def set_gemini_client(self, gemini_chat_client):
        self.gemini_chat_client = gemini_chat_client

    def start_scheduler(self):
        if self.gemini_chat_client is None:
            raise ValueError("GeminiChatClient instance is required to start")
        self.scheduler.start()

    def get_all_jobs(self):
        """
        Returns all the jobs currently scheduled in the scheduler.
        """
        return self.scheduler.get_jobs()

    def add_daily_task(self, prompt, *, precondition_prompt=None, negative_prompt=None):
        """
        Adds a new daily task to the scheduler. It will be executed once per day.
        All input prompts should be done in a natural language it will be sent to GPT/Gemini like LLM to be processed as is.
        In the prompts do not explicitly use funcitons name, assume that prompt is the message that use might have typed and sent to do the action, and user does not know which funcitons exist.

        Args:
            prompt: The prompt to be executed as part of the task. If precondition exists it will be executed after the precondition is met.
            if not precondition exists, the prompt will be executed directly.
            precondition_prompt: The prompt to verify the precondition before executing the task.
            negative_prompt: The prompt to be executed if the precondition is not met.
        """
        task = LLMTask(prompt, precondition_prompt=precondition_prompt, negative_prompt=negative_prompt)
        def task_function():
            return ScheduledTaskExecutor.execute_task(self.gemini_chat_client, task)
        return "job id is: " + self.scheduler.add_job(task_function, CronTrigger(hour='1'))

    def add_minute_task(self, prompt, *, precondition_prompt=None, negative_prompt=None):
        """
        Adds a new minute task to the scheduler. It will be executed once per minute.
        All input prompts should be done in a natural language it will be sent to GPT/Gemini like LLM to be processed as is.
        In the prompts do not explicitly use funcitons name, assume that prompt is the message that use might have typed and sent to do the action, and user does not know which funcitons exist.
        
        Args:
            prompt: The prompt to be executed as part of the task. If precondition exists it will be executed after the precondition is met.
            if not precondition exists, the prompt will be executed directly.
            precondition_prompt: The prompt to verify the precondition before executing the task.
            negative_prompt: The prompt to be executed if the precondition is not met.
        """
        task = LLMTask(prompt, precondition_prompt=precondition_prompt, negative_prompt=negative_prompt)
        def task_function():
            if self.debug:
                print("Executing task")
            return ScheduledTaskExecutor.execute_task(self.gemini_chat_client, task, self.debug)
        return "job id is: " + self.scheduler.add_job(task_function, CronTrigger(hour='*', minute='*'))

    @staticmethod
    def execute_task(gemini_chat_client, task, debug):
        """
        Evaluates the precondition (if any) and executes the rule if the conditions are met.

        Args:
            task: An instance of ScheduledTask representing the task to be executed.
        """
        if debug:
            print("Executing task (inside)")

        if task.precondition_prompt:
            # Construct the prompt with the additional instruction
            verification_prompt = (
                f"{task.precondition_prompt} Please execute all required tools "
                "to verify if this precondition is still met or not and return True/False."
            )

            # Send the verification prompt to the GeminiChatClient
            response = gemini_chat_client.send_message(verification_prompt)

            # Parse the response to extract the boolean value
            is_precondition_met = ScheduledTaskExecutor._parse_boolean_response(response)

            if is_precondition_met:
                gemini_chat_client.send_message(task.prompt)
            else:
                gemini_chat_client.send_message(task.negative_prompt)
        else:
            gemini_chat_client.send_message(task.prompt)

    @staticmethod
    def _parse_boolean_response(response):
        """
        Parses the response from the GeminiChatClient to extract a boolean value.

        Args:
            response: The text response from the GeminiChatClient.

        Returns:
            bool: True if the response indicates the precondition is met, False otherwise.
        """

        # Implement your parsing logic here based on how your LLM responds
        # For example, if the LLM returns "True" or "False" directly:
        return "true" in response.lower() 
