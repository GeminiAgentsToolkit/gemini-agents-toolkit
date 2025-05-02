"""An agent for executing user's instructions"""

import logging
import uuid 

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type

from google.adk.sessions import InMemorySessionService
from google.adk.sessions.base_session_service import GetSessionConfig
from google.adk.runners import Runner

from google.genai import types as genai_types

import threading


class TooManyFunctionCallsException(Exception):
    def __init__(self,message):
        super().__init__(message)


def log_retry_error(retry_state):
    """Logs the retry attempt details including the error message."""
    print(f"Retrying {retry_state.fn.__name__}... Attempt #{retry_state.attempt_number}, "
        f"Last error: {retry_state.outcome.exception()}")


# pylint: disable-next=too-many-instance-attributes
class ADKAgenService:
    """An agent to request LLM for executing users instructions with tools (custom functions) provided by user"""

    # pylint: disable-next=too-many-instance-attributes, too-many-arguments, too-many-locals
    def __init__(
            self,
            agent,
            *,
            function_call_limit_per_chat=None,
            debug=False,
            on_message=None,
            session_service=None,
            app_name="adk_service",
            events_per_session=-1
    ):
        self.agent = agent
        self.function_call_limit_per_chat = function_call_limit_per_chat
        self.debug = debug
        self.on_message = on_message
        self.session_service = session_service if session_service else InMemorySessionService()
        self.runners = {}
        self.app_name = app_name
        # Lock for thread-safe access to runners dictionary during creation
        self.runner_lock = threading.Lock()
        self.events_per_session = events_per_session

    def _maybe_create_chat_session(self, *, user_id, session_id, num_recent_events, events=[]):
        get_config = None
        if num_recent_events > 0:
            get_config = GetSessionConfig(num_recent_events=num_recent_events)
        chat_session = self.session_service.get_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id,
            config=get_config)
        if not chat_session:
            chat_session = self.session_service.create_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id)
        if events:
            for event in events:
                self.session_service.append_event(session=chat_session, event=event)
        return chat_session

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=16), after=log_retry_error, retry=retry_if_not_exception_type(TooManyFunctionCallsException))
    def send_message(self, msg: str, *, user_id="default_user", session_id=None, events=[]) -> tuple[str, list]:
        """Initiate communication with LLM to execute user's instructions"""
        if events and session_id:
            raise ValueError()
        if not session_id:
            session_id = str(uuid.uuid4())
        session = self._maybe_create_chat_session(user_id=user_id, session_id=session_id, num_recent_events=self.events_per_session, events=events)
        
        if self.debug:
            print(f"about to send msg: {msg}")
        
        runner_id = user_id + session_id
        runner_instance = self.runners.get(runner_id)
        if not runner_instance:
            with self.runner_lock:
                runner_instance = self.runners.get(runner_id) # Double-check
                if not runner_instance:
                    try:
                        runner_instance = Runner(
                            agent=self.agent,
                            app_name=self.app_name,
                            session_service=self.session_service
                        )
                        self.runners[runner_id] = runner_instance
                    except Exception as e:
                        logging.exception(f"Fatal Error creating agent/runner for space {session_id}: {e}") # Use exception for traceback

        user_content = genai_types.Content(role='user', parts=[genai_types.Part(text=msg)])
        final_response_text = ""
        function_call_counter = 0

        try:
            for event in runner_instance.run(
                user_id=user_id,
                new_message=user_content,
                session_id=session_id):
                function_call_counter = function_call_counter + 1
                if self.function_call_limit_per_chat is not None and function_call_counter >= self.function_call_limit_per_chat:
                    raise TooManyFunctionCallsException(
                        f"Exceed allowed number of function calls: {self.function_call_limit_per_chat}"
                    )
                

                logging.debug(f"ADK Event ({session_id}): Author={event.author}, Content={event.content}")

                if event.error_message:
                    logging.error(f"ADK Runner Error ({session_id}): {event.error_message}")
                    break
                if self.debug:
                    print(str(event))

                if event.is_final_response() and event.content and event.content.parts:
                    text_part = next((part.text for part in event.content.parts if part.text), None)
                    if text_part:
                        final_response_text = text_part
                        logging.debug(f"ADK signaled final response with text ({session_id}): {final_response_text}")
                    break
        except Exception as e:
                logging.exception(f"Error during runner.run for space {session_id}: {e}")

        logging.info(f"Runner loop finished for space {session_id}.")
        if final_response_text:
                logging.debug(f"Note: ADK provided final text for space {session_id}, "
                    "but relying on agent to use send_chat_message_tool for output.")
        
        if self.on_message:
            self.on_message(final_response_text)
        
        return final_response_text, self._maybe_create_chat_session(
            session_id=session_id, user_id=user_id, num_recent_events=self.events_per_session).events
