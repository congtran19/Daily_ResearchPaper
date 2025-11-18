from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
SOURCE_NAME = os.getenv("SOURCE_NAME", "arxiv")
DIGEST_OUTPUT_PATH = os.getenv("DIGEST_OUTPUT_PATH", "./daily_digests/")
