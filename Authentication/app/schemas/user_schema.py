from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class UserCreate(BaseModel):
    email: EmailStr
    full_name: str
    password: str

class UserRead(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    is_active: bool 
    is_admin: bool
    phone: Optional[str] = None
    location: Optional[str] = None
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class UserStats(BaseModel):
    plants_monitored: int
    forecasts_made: int
    weight_scans: int
    disease_checks: int = 0


class UserProfile(BaseModel):
    user_id: str
    name: str
    email: str
    phone: Optional[str] = None
    location: Optional[str] = None
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    stats: UserStats


class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    bio: Optional[str] = None
    avatar_url: Optional[str] = None


class PreferencesResponse(BaseModel):
    user_id: str
    push_notifications: bool
    email_notifications: bool
    auto_sync: bool
    dark_mode: bool
    language: str
    updated_at: str


class PreferencesUpdate(BaseModel):
    push_notifications: Optional[bool] = None
    email_notifications: Optional[bool] = None
    auto_sync: Optional[bool] = None
    dark_mode: Optional[bool] = None
    language: Optional[str] = None


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class DeleteAccountRequest(BaseModel):
    password: str
    confirmation: str  # Must be "DELETE"

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    refresh_token: str | None = None
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str

class GoogleAuthRequest(BaseModel):
    id_token: str

