from datetime import datetime, timezone
import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlmodel import Session, select

from app.core.db import get_session
from app.core.security import get_password_hash, verify_password
from app.models.user import User, Preference
from app.schemas.user_schema import (
    UserProfile, ProfileUpdate, UserStats, PreferencesResponse, PreferencesUpdate,
    ChangePasswordRequest, DeleteAccountRequest, UserRead
)
from app.routers.auth import get_current_user

router = APIRouter(prefix="/api/users", tags=["users"])

# Static directory for avatars
AVATAR_DIR = Path("static/avatars")
AVATAR_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


@router.get("/profile", response_model=UserProfile)
def get_user_profile(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Get current user's profile with statistics"""
    # TODO: Calculate real stats from ML service
    # For now, return mock stats
    stats = UserStats(
        plants_monitored=0,
        forecasts_made=0,
        weight_scans=0,
        disease_checks=0,
    )
    
    return UserProfile(
        user_id=str(current_user.id),
        name=current_user.full_name,
        email=current_user.email,
        phone=current_user.phone,
        location=current_user.location,
        bio=current_user.bio,
        avatar_url=current_user.avatar_url,
        created_at=current_user.created_at.isoformat() if current_user.created_at else None,
        updated_at=current_user.updated_at.isoformat() if current_user.updated_at else None,
        stats=stats,
    )


@router.put("/profile", response_model=UserRead)
def update_user_profile(
    profile_update: ProfileUpdate,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Update current user's profile"""
    if profile_update.full_name is not None:
        current_user.full_name = profile_update.full_name
    
    if profile_update.phone is not None:
        current_user.phone = profile_update.phone
    
    if profile_update.location is not None:
        current_user.location = profile_update.location
    
    if profile_update.bio is not None:
        current_user.bio = profile_update.bio
    
    if profile_update.avatar_url is not None:
        current_user.avatar_url = profile_update.avatar_url
    
    # Update timestamp
    current_user.updated_at = datetime.now(timezone.utc)
    
    session.add(current_user)
    session.commit()
    session.refresh(current_user)
    
    return current_user


@router.post("/avatar")
async def upload_avatar(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Upload user avatar image"""
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    
    # Read file and check size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024}MB",
        )
    
    # Generate unique filename
    unique_filename = f"{current_user.id}_{uuid.uuid4().hex}{file_ext}"
    file_path = AVATAR_DIR / unique_filename
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(contents)
    
    # Update user avatar URL
    # Note: Adjust the base URL according to your deployment
    avatar_url = f"/static/avatars/{unique_filename}"
    current_user.avatar_url = avatar_url
    current_user.updated_at = datetime.now(timezone.utc)
    
    session.add(current_user)
    session.commit()
    
    return {
        "avatar_url": avatar_url,
        "message": "Avatar uploaded successfully"
    }


@router.get("/preferences", response_model=PreferencesResponse)
def get_user_preferences(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Get user's app preferences"""
    preference = session.exec(
        select(Preference).where(Preference.user_id == current_user.id)
    ).first()
    
    # Create default preferences if not exist
    if not preference:
        preference = Preference(user_id=current_user.id)
        session.add(preference)
        session.commit()
        session.refresh(preference)
    
    return PreferencesResponse(
        user_id=str(current_user.id),
        push_notifications=preference.push_notifications,
        email_notifications=preference.email_notifications,
        auto_sync=preference.auto_sync,
        dark_mode=preference.dark_mode,
        language=preference.language,
        updated_at=preference.updated_at.isoformat() if preference.updated_at else datetime.now(timezone.utc).isoformat(),
    )


@router.put("/preferences", response_model=PreferencesResponse)
def update_user_preferences(
    preferences_update: PreferencesUpdate,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Update user's app preferences"""
    preference = session.exec(
        select(Preference).where(Preference.user_id == current_user.id)
    ).first()
    
    # Create if not exist
    if not preference:
        preference = Preference(user_id=current_user.id)
        session.add(preference)
    
    # Update fields
    if preferences_update.push_notifications is not None:
        preference.push_notifications = preferences_update.push_notifications
    
    if preferences_update.email_notifications is not None:
        preference.email_notifications = preferences_update.email_notifications
    
    if preferences_update.auto_sync is not None:
        preference.auto_sync = preferences_update.auto_sync
    
    if preferences_update.dark_mode is not None:
        preference.dark_mode = preferences_update.dark_mode
    
    if preferences_update.language is not None:
        preference.language = preferences_update.language
    
    # Update timestamp
    preference.updated_at = datetime.now(timezone.utc)
    
    session.add(preference)
    session.commit()
    session.refresh(preference)
    
    return PreferencesResponse(
        user_id=str(current_user.id),
        push_notifications=preference.push_notifications,
        email_notifications=preference.email_notifications,
        auto_sync=preference.auto_sync,
        dark_mode=preference.dark_mode,
        language=preference.language,
        updated_at=preference.updated_at.isoformat(),
    )


@router.post("/change-password")
def change_password(
    password_request: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Change user password with current password verification"""
    # Verify current password
    if not verify_password(password_request.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )
    
    # Validate new password (basic validation)
    if len(password_request.new_password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be at least 6 characters long",
        )
    
    # Update password
    current_user.hashed_password = get_password_hash(password_request.new_password)
    current_user.updated_at = datetime.now(timezone.utc)
    
    session.add(current_user)
    session.commit()
    
    return {
        "message": "Password changed successfully"
    }


@router.delete("/account")
def delete_account(
    delete_request: DeleteAccountRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Soft delete user account (requires password and 'DELETE' confirmation)"""
    # Verify password
    if not verify_password(delete_request.password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect password",
        )
    
    # Verify confirmation string
    if delete_request.confirmation != "DELETE":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Confirmation must be "DELETE"',
        )
    
    # Soft delete: set is_active to False and set deleted_at
    current_user.is_active = False
    current_user.deleted_at = datetime.now(timezone.utc)
    current_user.updated_at = datetime.now(timezone.utc)
    
    session.add(current_user)
    session.commit()
    
    return {
        "message": "Account deleted successfully"
    }
