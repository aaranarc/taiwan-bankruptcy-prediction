"""LOCAL DEMO ONLY: the real API with an in-memory stand-in for Supabase, for screenshots and offline demos.

Signup/login/JWT/scoring/history run through the real routes; only the database is replaced, so nothing is
persisted. Never deployed: the Docker image copies only app/ and config/, and vercel.json excludes scripts/.

Usage: python -m scripts.demo_api   (then: BACKEND_URL=http://127.0.0.1:8000 streamlit run streamlit_app.py)
"""
import os
import uuid
from datetime import datetime, timezone

os.environ.setdefault("JWT_SECRET_KEY", "local-demo-only-not-a-secret")

import uvicorn  # noqa: E402

from app.core.dependencies import get_supabase  # noqa: E402
from app.main import app  # noqa: E402


class _Result:
    def __init__(self, data):
        self.data = data


class _Query:
    """Just enough of the supabase-py query builder for this app: select/eq/order/limit/insert/execute."""

    def __init__(self, rows: list):
        self.rows, self.filters, self.sort, self.new = rows, [], None, None

    def select(self, *args, **kwargs):
        return self

    def eq(self, column, value):
        self.filters.append((column, value))
        return self

    def order(self, column, desc=False):
        self.sort = (column, desc)
        return self

    def limit(self, n):
        return self

    def insert(self, row):
        self.new = {"id": str(uuid.uuid4()), "created_at": datetime.now(timezone.utc).isoformat(), **row}
        return self

    def execute(self):
        if self.new is not None:
            self.rows.append(self.new)
            return _Result([self.new])
        out = [r for r in self.rows if all(r.get(c) == v for c, v in self.filters)]
        if self.sort:
            out.sort(key=lambda r: r.get(self.sort[0]), reverse=self.sort[1])
        return _Result(out)


class MemoryDB:
    def __init__(self):
        self.tables = {"users": [], "predictions": []}

    def table(self, name):
        return _Query(self.tables.setdefault(name, []))


db = MemoryDB()
app.dependency_overrides[get_supabase] = lambda: db

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8000")))
