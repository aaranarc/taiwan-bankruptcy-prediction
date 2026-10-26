from datetime import timedelta

from app.core.security import create_access_token, decode_access_token, hash_password, verify_password


def test_password_round_trip():
    hashed = hash_password("s3cret-pass")
    assert hashed != "s3cret-pass"
    assert verify_password("s3cret-pass", hashed)


def test_wrong_password_fails():
    assert not verify_password("wrong", hash_password("s3cret-pass"))


def test_jwt_round_trip():
    assert decode_access_token(create_access_token("user@example.com")) == "user@example.com"


def test_tampered_jwt_returns_none():
    token = create_access_token("user@example.com")
    tampered = token[:-2] + ("aa" if token[-2:] != "aa" else "bb")
    assert decode_access_token(tampered) is None


def test_expired_jwt_returns_none():
    assert decode_access_token(create_access_token("user@example.com", expires_delta=timedelta(seconds=-1))) is None
