from typing import Optional
from datetime import datetime, timezone
from sqlmodel import SQLModel, Field

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    full_name: str
    hashed_password: str
    is_active: bool = Field(default=True)
    is_admin: bool = Field(default=False)
    
    # Extended profile fields
    phone: Optional[str] = Field(default=None, max_length=20)
    location: Optional[str] = Field(default=None, max_length=100)
    bio: Optional[str] = Field(default=None)
    avatar_url: Optional[str] = Field(default=None, max_length=500)
    
    # Timestamps
    created_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))
    deleted_at: Optional[datetime] = Field(default=None)


class Preference(SQLModel, table=True):
    __tablename__ = "preferences"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", unique=True, index=True)
    
    push_notifications: bool = Field(default=True)
    email_notifications: bool = Field(default=False)
    auto_sync: bool = Field(default=True)
    dark_mode: bool = Field(default=False)
    language: str = Field(default="English", max_length=50)
    
    created_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))