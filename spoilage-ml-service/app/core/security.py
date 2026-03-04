from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
from .config import settings

bearer = HTTPBearer(auto_error=False)

def require_user(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> dict:
    # ✅ Auth OFF mode (demo mode)
    if not getattr(settings, "AUTH_ENABLED", True):
        return {
            "sub": "demo_user",
            "role": "demo",
            "service": "spoilage-ml-service",
        }

    # ✅ Auth ON mode (normal)
    if creds is None or not creds.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing token"
        )

    try:
        payload = jwt.decode(
            creds.credentials,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM]
        )
        return payload

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
