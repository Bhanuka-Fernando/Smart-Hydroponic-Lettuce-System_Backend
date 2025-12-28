from datetime import datetime, timedelta, timezone
from secrets import token_urlsafe

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError
from sqlmodel import Session, select

from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests

from app.core.config import settings
from app.core.db import get_session
from app.core.security import (
    create_access_token,
    decode_access_token,
    get_password_hash,
    verify_password,
)
from app.models.token import RefreshToken
from app.models.user import User
from app.schemas.user_schema import (
    UserCreate,
    UserRead,
    LoginRequest,
    Token,
    RefreshRequest,
    GoogleAuthRequest,
)

router = APIRouter(prefix="/auth", tags=["auth"])

# IMPORTANT:
# Swagger "Authorize" uses OAuth2 password flow -> it will call tokenUrl and send form-data:
# username=<email>, password=<password>
oauth2_schema = OAuth2PasswordBearer(tokenUrl="/auth/token")


@router.post("/register", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def register_user(user_in: UserCreate, session: Session = Depends(get_session)):
    existing = session.exec(select(User).where(User.email == user_in.email)).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email is already registered.",
        )

    db_user = User(
        email=user_in.email,
        full_name=user_in.full_name,
        hashed_password=get_password_hash(user_in.password),
    )

    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user


# ✅ Keep this JSON login for Expo (email/password in JSON)
@router.post("/login", response_model=Token)
def login(login_in: LoginRequest, session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.email == login_in.email)).first()

    if not user or not verify_password(login_in.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)}, expires_delta=access_token_expires
    )

    refresh_value = token_urlsafe(32)
    refresh_expires = datetime.now(timezone.utc) + timedelta(
        days=settings.REFRESH_TOKEN_EXPIRE_DAYS
    )

    db_refresh = RefreshToken(
        user_id=user.id,
        token=refresh_value,
        expires_at=refresh_expires,
    )
    session.add(db_refresh)
    session.commit()

    return Token(access_token=access_token, refresh_token=refresh_value)


# ✅ NEW: OAuth2 form login for Swagger Authorize
@router.post("/token", response_model=Token)
def token_login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: Session = Depends(get_session),
):
    # Swagger sends email inside "username"
    email = form_data.username
    password = form_data.password

    user = session.exec(select(User).where(User.email == email)).first()

    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)}, expires_delta=access_token_expires
    )

    refresh_value = token_urlsafe(32)
    refresh_expires = datetime.now(timezone.utc) + timedelta(
        days=settings.REFRESH_TOKEN_EXPIRE_DAYS
    )

    db_refresh = RefreshToken(
        user_id=user.id,
        token=refresh_value,
        expires_at=refresh_expires,
    )
    session.add(db_refresh)
    session.commit()

    return Token(access_token=access_token, refresh_token=refresh_value)


def get_current_user(
    token: str = Depends(oauth2_schema),
    session: Session = Depends(get_session),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_access_token(token)
        user_id: str | None = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = session.get(User, int(user_id))
    if user is None or not user.is_active:
        raise credentials_exception

    return user


@router.get("/me", response_model=UserRead)
def read_me(current_user: User = Depends(get_current_user)):
    return current_user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required.",
        )
    return current_user


@router.get("/admin-only")
def admin_stuff(current_user: User = Depends(require_admin)):
    return {"message": "Only admins can see this"}


@router.post("/refresh", response_model=Token)
def refresh_token(req: RefreshRequest, session: Session = Depends(get_session)):
    db_token = session.exec(
        select(RefreshToken).where(RefreshToken.token == req.refresh_token)
    ).first()

    now = datetime.now(timezone.utc)

    if not db_token or db_token.revoked or db_token.expires_at < now:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token.",
        )

    user = session.get(User, db_token.user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive.",
        )

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)}, expires_delta=access_token_expires
    )

    return Token(access_token=access_token, refresh_token=req.refresh_token)


@router.post("/google", response_model=Token)
def google_login(req: GoogleAuthRequest, session: Session = Depends(get_session)):
    if not settings.GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Google auth not configured.",
        )

    try:
        idinfo = google_id_token.verify_oauth2_token(
            req.id_token,
            google_requests.Request(),
            settings.GOOGLE_CLIENT_ID,
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google ID token.",
        )

    email = idinfo.get("email")
    full_name = idinfo.get("name") or (email.split("@")[0] if email else "")

    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google token did not contain an email.",
        )

    user = session.exec(select(User).where(User.email == email)).first()
    if not user:
        user = User(
            email=email,
            full_name=full_name,
            hashed_password=get_password_hash("google-login"),
            is_active=True,
        )
        session.add(user)
        session.commit()
        session.refresh(user)

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)}, expires_delta=access_token_expires
    )

    refresh_value = token_urlsafe(32)
    refresh_expires = datetime.now(timezone.utc) + timedelta(
        days=settings.REFRESH_TOKEN_EXPIRE_DAYS
    )

    db_refresh = RefreshToken(
        user_id=user.id,
        token=refresh_value,
        expires_at=refresh_expires,
    )
    session.add(db_refresh)
    session.commit()

    return Token(access_token=access_token, refresh_token=refresh_value)
