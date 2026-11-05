import json
from typing import Any, Optional

import redis

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger("redis_cache")

class RedisCache:
    def __init__(self):
        self.client: Optional[redis.Redis] = None
        self.connect()

    def connect(self):
        if not settings.REDIS_URL:
            logger.info("REDIS_URL not set. Caching is disabled.")
            return
        # Never log the URL itself: managed Redis URLs embed the password.
        logger.info("Connecting to Redis...")
        try:
            self.client = redis.Redis.from_url(
                settings.REDIS_URL, decode_responses=True, socket_timeout=2.0, socket_connect_timeout=2.0
            )
            self.client.ping()
            logger.info("Successfully connected to Redis.")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}. Caching will be disabled.")
            self.client = None

    def get(self, key: str) -> Optional[Any]:
        if not self.client:
            return None
        try:
            val = self.client.get(key)
            if val:
                return json.loads(val)
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
        return None

    def set(self, key: str, value: Any, expire_seconds: int = 86400) -> bool:
        if not self.client:
            return False
        try:
            self.client.setex(key, expire_seconds, json.dumps(value))
            return True
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
            return False

    def is_connected(self) -> bool:
        if not self.client:
            return False
        try:
            return bool(self.client.ping())
        except Exception:
            return False

redis_cache = RedisCache()
