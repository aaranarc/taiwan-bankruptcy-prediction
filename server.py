"""ASGI entrypoint for Vercel's Python `web` service. The UI itself stays ordinary Streamlit code.

Locally you still run `streamlit run streamlit_app.py`; this file is only needed where Streamlit runs
inside an ASGI host (Vercel). Same pattern as aaranarc/earnings-call-analyst.
"""
import os
from pathlib import Path

# Serverless bundles may omit hidden files such as .streamlit/config.toml; these mirror it.
for key, value in {
    'STREAMLIT_THEME_BASE': 'dark',
    'STREAMLIT_THEME_PRIMARY_COLOR': '#5B8DEF',
    'STREAMLIT_THEME_BACKGROUND_COLOR': '#0B1220',
    'STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR': '#131C2E',
    'STREAMLIT_THEME_TEXT_COLOR': '#E5EAF3',
    'STREAMLIT_SERVER_HEADLESS': 'true',
    'STREAMLIT_CLIENT_TOOLBAR_MODE': 'minimal',
    'STREAMLIT_SERVER_MAX_UPLOAD_SIZE': '4',  # MB; Vercel functions reject request bodies over 4.5 MB
}.items():
    os.environ.setdefault(key, value)

from starlette.responses import JSONResponse  # noqa: E402  (env must be set before Streamlit loads config)
from starlette.routing import Route  # noqa: E402
from streamlit import App, config  # noqa: E402

# An ASGI host can import Streamlit before this module; refresh its cached settings.
config.get_config_options(force_reparse=True)


async def health(request):
    # Booleans only: never echo the binding URL or any credential.
    return JSONResponse({'status': 'ok', 'interface': 'streamlit',
                         'api_binding_configured': bool(os.getenv('API_INTERNAL_URL')),
                         'commit': os.getenv('VERCEL_GIT_COMMIT_SHA', 'local')})


app = App(Path(__file__).with_name('streamlit_app.py'), routes=[Route('/health', health)])
