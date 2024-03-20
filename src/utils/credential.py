import os
from dotenv import load_dotenv

load_dotenv()

NEPTUNE_PROJECT_NAME = os.getenv('NEPTUNE_PROJECT_NAME')
NEPTUNE_PROJECT_CRED = os.getenv('NEPTUNE_PROJECT_CRED')
