import unittest
from unittest.mock import Mock, patch, MagicMock

from gemini_agents_toolkit.agent import ADKAgentService, TooManyFunctionCallsException, log_retry_error
from google.genai import types as genai_types
from google.api_core import exceptions as google_exceptions
import logging
import uuid

# Basic configuration for logging to see output during test development if necessary
# logging.basicConfig(level=logging.DEBUG)

class TestADKAgentService(unittest.TestCase):

    def setUp(self):
        """Set up common test resources."""
        self.mock_agent = Mock()
        self.mock_session_service = Mock()
        
        # This patch is active for all methods in this class.
        # self.MockRunnerClass is the mock for the Runner class itself.
        # self.mock_runner_instance is the mock for an instance of Runner.
        patcher_runner = patch('gemini_agents_toolkit.agent.Runner')
        self.MockRunnerClass = patcher_runner.start()
        self.addCleanup(patcher_runner.stop)
        self.mock_runner_instance = self.MockRunnerClass.return_value # Default instance returned by Runner()
        
        self.mock_chat_session = Mock()
        self.mock_chat_session.events = [] 
        self.mock_session_service.get_session.return_value = self.mock_chat_session
        self.mock_session_service.create_session.return_value = self.mock_chat_session
        
        self.adk_service = ADKAgentService(
            agent=self.mock_agent,
            session_service=self.mock_session_service
        )
        
        # Default behavior for Runner instantiation and run method
        self.MockRunnerClass.return_value = self.mock_runner_instance
        self.mock_runner_instance.run.return_value = iter([])


    @patch('gemini_agents_toolkit.agent.ADKAgentService._maybe_create_chat_session')
    def test_send_message_with_events_and_session_id_no_value_error(self, mock_maybe_create_session):
        mock_session_with_events = Mock()
        mock_session_with_events.events = [] 
        mock_maybe_create_session.return_value = mock_session_with_events
        
        self.mock_runner_instance.run.return_value = iter([]) # Ensure run is iterable

        test_session_id = "test_session_123"
        test_events = [genai_types.Content(role="user", parts=[genai_types.Part(text="hello")])]
        
        try:
            self.adk_service.send_message(
                msg="Test message",
                user_id="test_user",
                session_id=test_session_id,
                events=test_events
            )
        except ValueError:
            self.fail("send_message raised ValueError unexpectedly")

        mock_maybe_create_session.assert_any_call(
            user_id="test_user",
            session_id=test_session_id,
            num_recent_events=self.adk_service.events_per_session,
            events=test_events
        )
        mock_maybe_create_session.assert_called_with( # Last call
            session_id=test_session_id,
            user_id="test_user",
            num_recent_events=self.adk_service.events_per_session
        )
        self.assertEqual(mock_maybe_create_session.call_count, 2)

    @patch('uuid.uuid4')
    @patch('logging.warning')
    @patch('gemini_agents_toolkit.agent.ADKAgentService._maybe_create_chat_session')
    def test_send_message_handles_BlockedPromptException(self, mock_maybe_create_session, mock_log_warning, mock_uuid):
        mock_uuid.return_value = "fixed_uuid_for_blocked_prompt_test"
        mock_session = Mock()
        mock_session.events = []
        mock_maybe_create_session.return_value = mock_session
        self.mock_runner_instance.run.side_effect = genai_types.BlockedPromptException("Test BlockedPromptException")

        final_response, _ = self.adk_service.send_message(msg="Test message", user_id="test_user")
        
        self.assertEqual(final_response, "Your prompt was blocked. Please modify your prompt and try again.")
        mock_log_warning.assert_called_once()
        logged_message = mock_log_warning.call_args[0][0]
        self.assertIn("Prompt was blocked for session fixed_uuid_for_blocked_prompt_test", logged_message)
        self.assertIn("Test BlockedPromptException", str(mock_log_warning.call_args[0][1]))


    @patch('uuid.uuid4')
    @patch('logging.warning')
    @patch('gemini_agents_toolkit.agent.ADKAgentService._maybe_create_chat_session')
    def test_send_message_handles_StopCandidateException(self, mock_maybe_create_session, mock_log_warning, mock_uuid):
        mock_uuid.return_value = "fixed_uuid_for_stop_candidate_test"
        mock_session = Mock()
        mock_session.events = []
        mock_maybe_create_session.return_value = mock_session
        self.mock_runner_instance.run.side_effect = genai_types.StopCandidateException("Test StopCandidateException")

        final_response, _ = self.adk_service.send_message(msg="Test message", user_id="test_user")

        self.assertEqual(final_response, "The response could not be completed. Please try again.")
        mock_log_warning.assert_called_once()
        logged_message = mock_log_warning.call_args[0][0]
        self.assertIn("Content generation stopped for session fixed_uuid_for_stop_candidate_test", logged_message)
        self.assertIn("Test StopCandidateException", str(mock_log_warning.call_args[0][1]))

    @patch('uuid.uuid4')
    @patch('logging.error')
    @patch('gemini_agents_toolkit.agent.ADKAgentService._maybe_create_chat_session')
    def test_send_message_handles_DeadlineExceeded(self, mock_maybe_create_session, mock_log_error, mock_uuid):
        mock_uuid.return_value = "fixed_uuid_for_deadline_exceeded_test"
        mock_session = Mock()
        mock_session.events = []
        mock_maybe_create_session.return_value = mock_session
        self.mock_runner_instance.run.side_effect = google_exceptions.DeadlineExceeded("Test DeadlineExceeded")

        final_response, _ = self.adk_service.send_message(msg="Test message", user_id="test_user")

        self.assertEqual(final_response, "The request timed out. Please try again later.")
        mock_log_error.assert_called_once()
        logged_message = mock_log_error.call_args[0][0]
        self.assertIn("API request timed out during runner.run for session_id='fixed_uuid_for_deadline_exceeded_test'", logged_message)
        self.assertIn("Test DeadlineExceeded", str(mock_log_error.call_args[0][1]))

    @patch('uuid.uuid4')
    @patch('logging.error')
    @patch('gemini_agents_toolkit.agent.ADKAgentService._maybe_create_chat_session')
    def test_send_message_handles_GoogleAPIError(self, mock_maybe_create_session, mock_log_error, mock_uuid):
        mock_uuid.return_value = "fixed_uuid_for_google_api_error_test"
        mock_session = Mock()
        mock_session.events = []
        mock_maybe_create_session.return_value = mock_session
        self.mock_runner_instance.run.side_effect = google_exceptions.GoogleAPIError("Test GoogleAPIError")

        final_response, _ = self.adk_service.send_message(msg="Test message", user_id="test_user")

        self.assertEqual(final_response, "An API error occurred. Please try again later.")
        mock_log_error.assert_called_once()
        logged_message = mock_log_error.call_args[0][0]
        self.assertIn("A Google API error occurred during runner.run for session_id='fixed_uuid_for_google_api_error_test'", logged_message)
        self.assertIn("Test GoogleAPIError", str(mock_log_error.call_args[0][1]))

    @patch('uuid.uuid4')
    @patch('logging.exception')
    @patch('gemini_agents_toolkit.agent.ADKAgentService._maybe_create_chat_session')
    def test_send_message_handles_generic_Exception_in_run(self, mock_maybe_create_session, mock_log_exception, mock_uuid):
        mock_uuid.return_value = "fixed_uuid_for_generic_exception_test"
        mock_session = Mock()
        mock_session.events = []
        mock_maybe_create_session.return_value = mock_session
        self.mock_runner_instance.run.side_effect = Exception("Test Generic Exception in run")

        final_response, _ = self.adk_service.send_message(msg="Test message", user_id="test_user")

        self.assertEqual(final_response, "An unexpected error occurred. Please try again.")
        mock_log_exception.assert_called_once()
        logged_message = mock_log_exception.call_args[0][0]
        self.assertIn("An unexpected error occurred during runner.run for session_id='fixed_uuid_for_generic_exception_test'", logged_message)
        self.assertIn("Test Generic Exception in run", str(mock_log_exception.call_args[0][1]))

    @patch('uuid.uuid4')
    @patch('logging.critical')
    @patch('gemini_agents_toolkit.agent.ADKAgentService._maybe_create_chat_session')
    def test_send_message_runner_creation_failure(self, mock_maybe_create_session, mock_log_critical, mock_uuid):
        # Ensure _maybe_create_chat_session still returns a valid session object
        mock_session = Mock()
        mock_session.events = [] # It should have an events attribute
        mock_maybe_create_session.return_value = mock_session
        
        # Make the Runner class constructor raise an exception
        runner_init_fail_exception = Exception("Runner init failed")
        self.MockRunnerClass.side_effect = runner_init_fail_exception
        
        mock_uuid.return_value = "fixed_uuid_for_runner_fail_test"
        
        # Call send_message - this should trigger the runner creation attempt
        response_text, response_events = self.adk_service.send_message(msg="Test message", user_id="test_user_runner_fail")

        # Assert the expected error response
        self.assertEqual(response_text, "Failed to initialize agent runner.")
        self.assertEqual(response_events, [])

        # Assert that logging.critical was called
        mock_log_critical.assert_called_once()
        logged_message = mock_log_critical.call_args[0][0]
        self.assertIn("Runner instance is None for session_id='fixed_uuid_for_runner_fail_test'", logged_message)
        self.assertIn("runner_id='test_user_runner_failfixed_uuid_for_runner_fail_test'", logged_message)
        
        # Assert that the original exception from Runner creation was logged by the general exception handler in __init__
        # This requires inspecting the logging.exception call within the Runner creation block.
        # For this, we need to patch logging.exception specifically for that block or check its general calls.
        # The current test setup doesn't distinguish logging.exception in runner creation vs. runner.run.
        # However, the critical log specific to runner_instance being None is the primary check here.


    @patch('logging.error')
    def test_log_retry_error_uses_logging(self, mock_log_error):
        mock_retry_state = Mock()
        mock_retry_state.fn.__name__ = "test_function"
        mock_retry_state.attempt_number = 3
        mock_exception = Exception("Test retry error")
        mock_retry_state.outcome.exception.return_value = mock_exception

        log_retry_error(mock_retry_state)

        mock_log_error.assert_called_once()
        logged_message = mock_log_error.call_args[0][0]
        expected_message_part1 = "Retrying test_function... Attempt #3"
        expected_message_part2 = f"Last error: {mock_exception}" # Note: str(mock_exception) might be more accurate if exception has custom __str__
        
        self.assertIn(expected_message_part1, logged_message)
        self.assertIn(expected_message_part2, logged_message)


if __name__ == '__main__':
    unittest.main()
