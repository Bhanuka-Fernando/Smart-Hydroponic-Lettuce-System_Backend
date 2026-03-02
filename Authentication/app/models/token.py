from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field

class RefreshToken(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    token: str = Field(index=True, unique=True)
    expires_at: datetime
    revoked: bool = Field(default=False)