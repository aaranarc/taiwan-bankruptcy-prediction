import os

import pandas as pd
import pytest

# Set before any app import: no real Supabase, no Redis (caching off), test-only JWT secret.
# load_dotenv() never overrides variables that are already set, so a local .env is ignored here.
os.environ["JWT_SECRET_KEY"] = "test-only-secret-at-least-32-bytes-long"
os.environ["SUPABASE_URL"] = ""
os.environ["SUPABASE_KEY"] = ""
os.environ["REDIS_URL"] = ""
os.environ["KV_URL"] = ""

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "data.csv")


@pytest.fixture(scope="session")
def dataset():
    df = pd.read_csv(DATA)
    df.columns = df.columns.str.strip()
    return df


@pytest.fixture
def raw_features(dataset):
    """First firm in the dataset as a feature dict."""
    return {k: float(v) for k, v in dataset.drop(columns=["Bankrupt?"]).iloc[0].items()}


@pytest.fixture
def risky_features(dataset):
    """A real bankrupt firm the model scores as high risk (highest PD among the first bankrupt firms)."""
    from app.services.model_service import model_service
    bankrupt = dataset[dataset["Bankrupt?"] == 1].drop(columns=["Bankrupt?"]).head(40)
    pds = model_service.score(bankrupt)
    return {k: float(v) for k, v in bankrupt.iloc[int(pds.argmax())].items()}
