"""The module to provide configuration values."""

import os

# ***** Google Cloud project settings *****
PROJECT_ID =  os.getenv('GOOGLE_PROJECT')
API_KEY =  os.getenv('GOOGLE_API_KEY')
REGION = os.environ.get('GOOGLE_REGION', 'us-west1')

# simple model is used for demo examples as a cost optimization
SIMPLE_MODEL = 'gemini-1.5-flash-002'
PRO_MODEL = 'gemini-1.5-pro-002'

# We highly recommend using the PRO model for production use cases
# There is a huge bug related to funciton calling that is much more
# prominent if falsh model is used.
DEFAULT_MODEL = PRO_MODEL
