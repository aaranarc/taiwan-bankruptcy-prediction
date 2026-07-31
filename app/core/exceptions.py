from fastapi import HTTPException, status

class CredentialsException(HTTPException):
    def __init__(self, detail: str = "Could not validate credentials"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )

class DatabaseException(HTTPException):
    def __init__(self, detail: str = "Database operation failed"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
        )

class EntityAlreadyExistsException(HTTPException):
    def __init__(self, detail: str = "Resource already exists"):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        )
