import os

# ***** Google Cloud project settings *****
project_id =  os.getenv('GOOGLE_PROJECT')
api_key =  os.getenv('GOOGLE_API_KEY')
region = os.environ.get('GOOGLE_REGION', 'us-west1')

default_model = 'gemini-1.5-pro'
# simple model is used for demo examples as a cost optimization
simple_model = 'gemini-1.5-flash'