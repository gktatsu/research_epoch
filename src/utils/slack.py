import requests
import json

import os
from dotenv import load_dotenv

load_dotenv()

def send_msg(text=""):
    slack_url = os.environ['SLACK_WEBHOOK_URL']
    content = {'content-type': 'application/json', 'text': text}
    requests.post(slack_url, data=json.dumps(content))

# send_msg("Hello World")