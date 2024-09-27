"""The module to provide configuration values."""

import os

# ***** Google Cloud project settings *****
PROJECT_ID =  os.getenv('GOOGLE_PROJECT')
API_KEY =  os.getenv('GOOGLE_API_KEY')
REGION = os.environ.get('GOOGLE_REGION', 'us-west1')

DEFAULT_MODEL = 'gemini-1.5-pro-002'
# simple model is used for demo examples as a cost optimization
SIMPLE_MODEL = 'gemini-1.5-flash-002'
