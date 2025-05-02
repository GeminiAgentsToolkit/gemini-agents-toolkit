"""The module to provide configuration values."""
# simple model is used for demo examples as a cost optimization
SIMPLE_MODEL = 'gemini-2.5-flash-preview-04-17'
PRO_MODEL = 'gemini-2.5-pro-preview-03-25'

# We highly recommend using the PRO model for production use cases
# There is a huge bug related to function calling that is much more
# prominent if flash model is used.
DEFAULT_MODEL = PRO_MODEL
