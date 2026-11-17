"""API keys for the organiser, the looking conditions and the report extraction: read from a file given on the command line, or from
the environment (OPENAI_API_KEY, ANTHROPIC_API_KEY). Returns None when neither is set, so conditions without an API run without one."""
import os


def api_key(key_file=None, env="OPENAI_API_KEY"):
    if key_file:
        return open(os.path.expanduser(key_file)).read().strip().splitlines()[0].strip()
    return os.environ.get(env)
