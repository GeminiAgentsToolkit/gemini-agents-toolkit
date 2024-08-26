from google.generativeai import (
    GenerativeModel,
)
import inspect
from gemini_toolbox import scheduler
import logging


class GeminiChatClient(object):

    def __init__(self, functions, model, *, debug=False, recreate_client_each_time=False, history_depth=-1, do_not_die=False):
        self.functions = {func.__name__: func for func in functions}
        self.chat = model.start_chat(enable_automatic_function_calling=True)
        self._model = model
        self.debug = debug
        self.recreate_client_each_time = recreate_client_each_time
        self.do_not_die = do_not_die
        if history_depth > 0 and recreate_client_each_time:
            raise ValueError("history_depth can only be used with recreate_client_each_time=False")
        
    def _maybe_trim_history(self):
        if self.history_depth > 0:
            trimmed_history = []
            user_messages_count = 0
            
            # Iterate through the history in reverse
            for h in reversed(self.chat._history):
                if h.role == "user":
                    if user_messages_count < self.history_depth:
                        trimmed_history.append(h)
                        user_messages_count += 1
                else:
                    trimmed_history.append(h)
                    
            # Reverse the trimmed history to restore the original order
            self.chat._history = list(reversed(trimmed_history))
        
    def _maybe_recreate_client(self):
        if self.recreate_client_each_time:
            self.chat = self._model.start_chat(enable_automatic_function_calling=True)

    def send_message(self, msg):
        if self.debug:
            print(f"about to send msg: {msg}")
        try:
            response = self.chat.send_message(msg)
            if self.debug:
                print(f"msg: '{msg}' is out")

            self._maybe_recreate_client()
        except Exception as e:
            if self.do_not_die:
                # log error to stadnart pytohng error logger
                logging.error(e, exc_info=True, stack_info=True)
                return {"error": str(e)}
            else:
                raise e
        # Extract the text from the final model response
        return response.text
    

def generate_chat_client_from_functions_package(package, model_name="gemini-1.5-pro", *, system_instruction="", debug=False, recreate_client_each_time=False, history_depth=-1, do_not_die=False, add_scheduling_functions=False, gcs_bucket=None, gcs_blob=None):
    all_functions = [
        func
        for _, func in inspect.getmembers(package, inspect.isfunction)
    ]
    return generate_chat_client_from_functions_list(all_functions, model_name=model_name, system_instruction=system_instruction, debug=debug, recreate_client_each_time=recreate_client_each_time, history_depth=history_depth, do_not_die=do_not_die, add_scheduling_functions=add_scheduling_functions, gcs_bucket=gcs_bucket, gcs_blob=gcs_blob)


def generate_chat_client_from_functions_list(all_functions, model_name="gemini-1.5-pro", *, system_instruction=None, debug=False, recreate_client_each_time=False, history_depth=-1, do_not_die=False, add_scheduling_functions=False, gcs_bucket=None, gcs_blob=None):
    if (not gcs_bucket and gcs_blob) or (gcs_bucket and not gcs_blob):
        raise ValueError("Both gcs_bucket and gcs_blob must be provided")
    if add_scheduling_functions:
        scheduler_instance = scheduler.ScheduledTaskExecutor(debug=debug, gcs_bucket=gcs_bucket, gcs_blob=gcs_blob)
        all_functions.extend([
            scheduler_instance.add_minute_task,
            scheduler_instance.add_daily_task,
            scheduler_instance.get_all_jobs,
            scheduler_instance.delete_job,
        ])
    if debug:
        print(f"all_functions: {all_functions}")
    model = GenerativeModel(model_name=model_name, tools=all_functions, system_instruction=system_instruction)
    clnt = GeminiChatClient(all_functions, model, debug=debug, recreate_client_each_time=recreate_client_each_time, history_depth=history_depth, do_not_die=do_not_die)
    if add_scheduling_functions:
        scheduler_instance.set_gemini_client(clnt)
        scheduler_instance.start_scheduler()
    return clnt
