"""An agent for executing user's instructions"""

import logging
import uuid 

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type

from google.adk.sessions import InMemorySessionService
from google.adk.sessions.base_session_service import GetSessionConfig
from google.adk.runners import Runner

from google.genai import types as genai_types
from google.api_core import exceptions as google_exceptions

import threading


class TooManyFunctionCallsException(Exception):
    def __init__(self,message):
        super().__init__(message)


def log_retry_error(retry_state):
    """Logs the retry attempt details including the error message."""
    logging.error(f"Retrying {retry_state.fn.__name__}... Attempt #{retry_state.attempt_number}, "
                  f"Last error: {retry_state.outcome.exception()}")


# pylint: disable-next=too-many-instance-attributes
class ADKAgentService:
    """An agent to request LLM for executing users instructions with tools (custom functions) provided by user"""

    # pylint: disable-next=too-many-instance-attributes, too-many-arguments, too-many-locals
    def __init__(
            self,
            agent,
            *,
            function_call_limit_per_chat=None,
            on_message=None,
            session_service=None,
            app_name="adk_service",
            events_per_session=-1
    ):
        logging.info("ADKAgentService initializing...")
        self.agent = agent
        self.function_call_limit_per_chat = function_call_limit_per_chat
        self.on_message = on_message
        self.session_service = session_service if session_service else InMemorySessionService()
        self.runners = {}
        self.app_name = app_name
        # Lock for thread-safe access to runners dictionary during creation
        self.runner_lock = threading.Lock()
        self.events_per_session = events_per_session
        logging.info(f"ADKAgentService initialized with: app_name='{self.app_name}', "
                     f"function_call_limit_per_chat={self.function_call_limit_per_chat}, "
                     f"events_per_session={self.events_per_session}")

    def _maybe_create_chat_session(self, *, user_id, session_id, num_recent_events, events=[]):
        logging.debug(f"Attempting to get/create chat session for user_id='{user_id}', session_id='{session_id}', app_name='{self.app_name}'")
        get_config = None
        if num_recent_events > 0:
            get_config = GetSessionConfig(num_recent_events=num_recent_events)
            logging.debug(f"GetSessionConfig created with num_recent_events={num_recent_events} for session_id='{session_id}'")

        chat_session = self.session_service.get_session(
            app_name=self.app_name,
            user_id=user_id,
            session_id=session_id,
            config=get_config)

        if not chat_session:
            logging.info(f"No existing session found for session_id='{session_id}'. Creating new session for user_id='{user_id}', app_name='{self.app_name}'.")
            chat_session = self.session_service.create_session(
                app_name=self.app_name,
                user_id=user_id,
                session_id=session_id)
            logging.info(f"New session created for session_id='{session_id}', user_id='{user_id}', app_name='{self.app_name}'.")
        else:
            logging.info(f"Fetched existing session for session_id='{session_id}', user_id='{user_id}', app_name='{self.app_name}'.")

        if events:
            logging.debug(f"Appending {len(events)} events to session_id='{session_id}'.")
            for event_count, event in enumerate(events, 1):
                self.session_service.append_event(session=chat_session, event=event)
            logging.info(f"Successfully appended {len(events)} events to session_id='{session_id}'.")
        return chat_session

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=16), after=log_retry_error, retry=retry_if_not_exception_type(TooManyFunctionCallsException))
    def send_message(self, msg: str, *, user_id="default_user", session_id=None, events=[]) -> tuple[str, list]:
        """Initiate communication with LLM to execute user's instructions"""
        # Log at the very beginning of the method
        effective_session_id = session_id if session_id else "new_session"
        logging.info(f"send_message called for user_id='{user_id}', session_id='{effective_session_id}'. Message: '{msg[:100]}{'...' if len(msg) > 100 else ''}'")

        if not session_id:
            session_id = str(uuid.uuid4())
            logging.info(f"No session_id provided. Generated new session_id='{session_id}' for user_id='{user_id}'.")
        
        session = self._maybe_create_chat_session(user_id=user_id, session_id=session_id, num_recent_events=self.events_per_session, events=events)
        
        logging.debug(f"about to send msg: {msg}")
        
        runner_id = user_id + session_id
        logging.debug(f"Runner ID for session_id='{session_id}' is '{runner_id}'.")
        runner_instance = self.runners.get(runner_id)
        if not runner_instance:
            with self.runner_lock:
                runner_instance = self.runners.get(runner_id) # Double-check
                if not runner_instance:
                    try:
                        logging.info(f"Creating new Runner instance for runner_id='{runner_id}', session_id='{session_id}'.")
                        runner_instance = Runner(
                            agent=self.agent,
                            app_name=self.app_name,
                            session_service=self.session_service
                        )
                        self.runners[runner_id] = runner_instance
                        logging.info(f"Runner instance created and cached for runner_id='{runner_id}', session_id='{session_id}'.")
                    except Exception as e:
                        # Original logging.exception is good here as it includes traceback
                        logging.exception(f"Fatal Error creating agent/runner for session_id='{session_id}', runner_id='{runner_id}': {e}")
                        # It might be appropriate to re-raise or return an error response here
                        # For now, we'll let it proceed, which likely results in an error later if runner_instance is None
                        # However, the original code also didn't explicitly handle this case beyond logging.
        
        if not runner_instance:
            logging.critical(f"Runner instance is None for session_id='{session_id}', runner_id='{runner_id}'. Cannot proceed.")
            # Option 1: Raise an exception
            # raise RuntimeError("Failed to initialize agent runner.")
            # Option 2: Return an error tuple (as per subtask preference)
            return "Failed to initialize agent runner.", []

        user_content = genai_types.Content(role='user', parts=[genai_types.Part(text=msg)])
        final_response_text = ""
        function_call_counter = 0
        
        logging.info(f"Preparing to call runner_instance.run() for session_id='{session_id}'.")
        try:
            for event in runner_instance.run( # This assumes runner_instance was successfully created.
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
                logging.debug(str(event))

                if event.is_final_response() and event.content and event.content.parts:
                    text_part = next((part.text for part in event.content.parts if part.text), None)
                    if text_part:
                        final_response_text = text_part
                        logging.debug(f"ADK signaled final response with text ({session_id}): {final_response_text}")
                    break
        except genai_types.BlockedPromptException as e:
                logging.warning(f"Prompt was blocked for session {session_id}: {e}")
                # Potentially set a specific error message for the user
                final_response_text = "Your prompt was blocked. Please modify your prompt and try again."
        except genai_types.StopCandidateException as e:
                logging.warning(f"Content generation stopped for session {session_id}: {e}")
                # Potentially set a specific error message for the user
                final_response_text = "The response could not be completed. Please try again."
        except google_exceptions.DeadlineExceeded as e:
                logging.error(f"API request timed out during runner.run for session_id='{session_id}': {e}")
                final_response_text = "The request timed out. Please try again later."
        except google_exceptions.GoogleAPIError as e:
                logging.error(f"A Google API error occurred during runner.run for session_id='{session_id}': {e}")
                final_response_text = "An API error occurred. Please try again later."
        except Exception as e: # Keep a general exception handler as a fallback
                logging.exception(f"An unexpected error occurred during runner.run for session_id='{session_id}': {e}")
                # It might be good to set a generic error message for final_response_text here too
                final_response_text = "An unexpected error occurred. Please try again."
        
        logging.info(f"runner_instance.run() completed or errored for session_id='{session_id}'.")

        # The ADK may provide a final text response here. However, the agent's design might rely on
        # a tool (e.g., send_chat_message_tool) to deliver the ultimate response to the user.
        # This logged text is the direct final output from the ADK runner.
        if final_response_text:
                logging.debug(f"Final response text for session_id='{session_id}' (prior to on_message): '{final_response_text[:100]}{'...' if len(final_response_text) > 100 else ''}'")
        else:
                logging.debug(f"No final response text was set from runner.run() for session_id='{session_id}'.")
        
        if self.on_message:
            logging.info(f"Calling on_message callback for session_id='{session_id}'.")
            self.on_message(final_response_text)
            logging.info(f"on_message callback completed for session_id='{session_id}'.")
        
        logging.info(f"Returning final response for session_id='{session_id}'. Response: '{final_response_text[:100]}{'...' if len(final_response_text) > 100 else ''}'")
        # The events returned here are from a fresh call to _maybe_create_chat_session, which fetches/creates and appends history.
        # This is the original behavior.
        returned_events = self._maybe_create_chat_session(
            session_id=session_id, user_id=user_id, num_recent_events=self.events_per_session).events
        logging.debug(f"Returning {len(returned_events)} events for session_id='{session_id}'.")
        return final_response_text, returned_events
