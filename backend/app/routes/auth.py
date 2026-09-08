import hashlib
import hmac
import os
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session
from sqlalchemy import and_
import jwt

from ..config import settings
from ..deps import (
    build_refresh_cookie_settings,
    create_access_token,
    decode_refresh_token,
    generate_refresh_session,
    get_current_user,
    get_db,
    hash_refresh_token,
    rotate_refresh_session,
)
from ..models import RefreshSession, User
from ..schemas import LoginRequest, RegisterRequest, TokenResponse, UserResponse

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.get("/me", response_model=UserResponse)
def me(username: str = Depends(get_current_user), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


def _hash_password(password: str) -> str:
    salt = os.urandom(16)
    iterations = 100_000
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        _, iterations_raw, salt_hex, digest_hex = password_hash.split("$")
        iterations = int(iterations_raw)
        salt = bytes.fromhex(salt_hex)
    except (ValueError, TypeError):
        return False

    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(digest.hex(), digest_hex)


def _set_refresh_cookie(response: Response, refresh_token: str, expires_at: datetime) -> None:
    cookie_settings = build_refresh_cookie_settings()
    response.set_cookie(
        key=settings.refresh_cookie_name,
        value=refresh_token,
        expires=expires_at,
        **cookie_settings,
    )


def _clear_refresh_cookie(response: Response) -> None:
    cookie_settings = build_refresh_cookie_settings()
    response.delete_cookie(
        key=settings.refresh_cookie_name,
        path=cookie_settings["path"],
        domain=cookie_settings["domain"],
    )


def _issue_tokens(
    response: Response,
    db: Session,
    username: str,
    user_id: int | None,
    request: Request,
) -> TokenResponse:
    access_token = create_access_token(subject=username)
    refresh_token, family_id, token_jti, expire_at = generate_refresh_session(subject=username)
    refresh_session = RefreshSession(
        user_id=user_id,
        username=username,
        family_id=family_id,
        token_jti=token_jti,
        token_hash=hash_refresh_token(refresh_token),
        expires_at=expire_at,
        user_agent=(request.headers.get("user-agent") or "")[:255],
        ip_address=(request.client.host if request.client else None),
    )
    db.add(refresh_session)
    db.commit()
    _set_refresh_cookie(response, refresh_token, expire_at)
    return TokenResponse(access_token=access_token)


def _revoke_family(db: Session, family_id: str, now: datetime) -> None:
    db.query(RefreshSession).filter(
        and_(
            RefreshSession.family_id == family_id,
            RefreshSession.revoked_at.is_(None),
        )
    ).update({RefreshSession.revoked_at: now}, synchronize_session=False)
    db.commit()


def _normalize_dt(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.username == request.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists",
        )

    user = User(
        username=request.username,
        password_hash=_hash_password(request.password),
        name=request.name or request.username,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, response: Response, request: Request, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if user:
        if not _verify_password(payload.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
            )
        return _issue_tokens(response, db, user.username, user.id, request)

    # Backward-compatibility for single-admin settings-based auth.
    if payload.username != settings.auth_username or payload.password != settings.auth_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    return _issue_tokens(response, db, payload.username, None, request)


@router.post("/refresh", response_model=TokenResponse)
def refresh_access_token(response: Response, request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get(settings.refresh_cookie_name)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token is missing",
        )

    try:
        payload = decode_refresh_token(token)
    except jwt.InvalidTokenError:
        _clear_refresh_cookie(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    token_hash = hash_refresh_token(token)
    session = db.query(RefreshSession).filter(RefreshSession.token_hash == token_hash).first()
    if not session:
        _clear_refresh_cookie(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh session not found",
        )

    now = datetime.now(timezone.utc)
    if session.family_id != payload.get("fid") or session.token_jti != payload.get("jti"):
        _revoke_family(db, session.family_id, now)
        _clear_refresh_cookie(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token mismatch",
        )

    if session.revoked_at is not None or session.rotated_at is not None:
        session.reuse_detected_at = now
        db.add(session)
        _revoke_family(db, session.family_id, now)
        _clear_refresh_cookie(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token reuse detected",
        )

    if _normalize_dt(session.expires_at) <= now:
        session.revoked_at = now
        db.add(session)
        db.commit()
        _clear_refresh_cookie(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token expired",
        )

    new_refresh_token, new_jti, new_expire_at = rotate_refresh_session(
        subject=session.username,
        family_id=session.family_id,
    )

    session.rotated_at = now
    session.replaced_by_jti = new_jti
    db.add(session)

    next_session = RefreshSession(
        user_id=session.user_id,
        username=session.username,
        family_id=session.family_id,
        token_jti=new_jti,
        token_hash=hash_refresh_token(new_refresh_token),
        expires_at=new_expire_at,
        user_agent=(request.headers.get("user-agent") or "")[:255],
        ip_address=(request.client.host if request.client else None),
    )
    db.add(next_session)
    db.commit()

    access_token = create_access_token(subject=session.username)
    _set_refresh_cookie(response, new_refresh_token, new_expire_at)
    return TokenResponse(access_token=access_token)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(response: Response, request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get(settings.refresh_cookie_name)
    if token:
        token_hash = hash_refresh_token(token)
        session = db.query(RefreshSession).filter(RefreshSession.token_hash == token_hash).first()
        if session and session.revoked_at is None:
            session.revoked_at = datetime.now(timezone.utc)
            db.add(session)
            db.commit()

    _clear_refresh_cookie(response)
    response.status_code = status.HTTP_204_NO_CONTENT
    return response
