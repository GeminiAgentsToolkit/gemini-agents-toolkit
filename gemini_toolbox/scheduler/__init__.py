from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from google.cloud import storage
from gemini_toolbox.scheduler.task import LLMTask
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import json


class ScheduledTaskExecutor:
    
    def __init__(self, *, debug=False, gcs_bucket=None, gcs_blob=None):
        """
        Initializes the ScheduledTaskExecutor with the given GeminiChatClient instance.
        """
        self.scheduler = BackgroundScheduler({
            'apscheduler.executors.default': {
                'class': 'apscheduler.executors.pool:ThreadPoolExecutor',
                'max_workers': '1'
            },
            'apscheduler.executors.processpool': {
                'type': 'processpool',
                'max_workers': '1'
            },
        })
        self.debug = debug
        self.gemini_chat_client = None
        self.tasks = []
        self.gcs_bucket = gcs_bucket
        self.gcs_blob = gcs_blob

    def set_gemini_client(self, gemini_chat_client):
        self.gemini_chat_client = gemini_chat_client

    def start_scheduler(self):
        if self.gemini_chat_client is None:
            raise ValueError("GeminiChatClient instance is required to start")
        self.scheduler.start()
        if self.gcs_blob:
            self._download_json_from_gcs()

    def delete_job(self, job_id: str):
        """
        Deletes a job from the scheduler based on the job_id.
        """
        for task in self.tasks:
            if task.id == job_id:
                self.scheduler.remove_job(job_id)
                self.tasks.remove(task)
                if self.gcs_blob:
                    self._upload_json_to_gcs()
                return "job deleted"
        return "job not found"

    def get_all_jobs(self):
        """
        Returns all the jobs currently scheduled in the scheduler. If user asks about periodic tasks/jobs/etc this function should be used.
        """
        return [ task.__dict__ for task in self.tasks ]

    def add_daily_task(self, prompt: str, *, precondition_prompt: str = None, negative_prompt: str = None):
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
        task = LLMTask(prompt, precondition_prompt=precondition_prompt, negative_prompt=negative_prompt, frequency='daily')
        return self._add_task(task, CronTrigger(hour='1'))

    def add_minute_task(self, prompt: str, *, precondition_prompt: str = None, negative_prompt: str = None):
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
        task = LLMTask(prompt, precondition_prompt=precondition_prompt, negative_prompt=negative_prompt, frequency='minute')
        return self._add_task(task, CronTrigger(hour='*', minute='*'))
    
    def _add_task(self, task, cron_trigger, upload_to_gcs=True):
        gemini_chat_client = self.gemini_chat_client
        debug = self.debug
        task.id = self.scheduler.add_job(execute_task, cron_trigger, args=[gemini_chat_client, task, debug]).id
        self.tasks.append(task)
        if self.gcs_blob and upload_to_gcs:
            self._upload_json_to_gcs()
        return "job id is: " + task.id

    def _tasks_to_json(self):
        """Converts a list of LLMTask objects to a JSON string."""
        # Convert each task to a dictionary, excluding the 'id' attribute
        task_dicts = [{k: v for k, v in task.__dict__.items() if k != 'id'} for task in self.tasks]
        return json.dumps(task_dicts, indent=4)  # Formatted JSON for readability

    def _upload_json_to_gcs(self):
        """Uploads a JSON string to GCS."""

        storage_client = storage.Client()
        bucket = storage_client.bucket(self.gcs_bucket)
        blob = bucket.blob(self.gcs_blob)

        blob.upload_from_string(self._tasks_to_json(), content_type='application/json')
        print(f"Uploaded JSON to GCS: gs://{self.gcs_bucket}/{self.gcs_blob}")

    def _download_json_from_gcs(self):
        """Downloads a JSON string from GCS."""

        storage_client = storage.Client()
        bucket = storage_client.bucket(self.gcs_bucket)
        blob = bucket.blob(self.gcs_blob)
        if not blob.exists():
            return

        json_string = blob.download_as_string().decode('utf-8')
        task_dicts = json.loads(json_string)
        tasks = [LLMTask(**task_dict) for task_dict in task_dicts]
        for task in tasks:
            if task.frequency == 'daily':
                self._add_task(task, CronTrigger(hour='1'), upload_to_gcs=False)
            elif task.frequency == 'minute':
                self._add_task(task, CronTrigger(hour='*', minute='*'), upload_to_gcs=False)

    def save_jobs_to_gcs(self):
        """
        Saves all the jobs currently scheduled in the scheduler to GCS.
        """
        # convert all tasks to json and put to GCS
        
    
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