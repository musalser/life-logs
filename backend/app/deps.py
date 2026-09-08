
from datetime import datetime, timedelta, timezone
import hashlib
import uuid

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import settings
from .db import SessionLocal
from app.services.diary_service import DiaryService
from app.adapters.ollama_adapter import OllamaAdapter



def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


auth_scheme = HTTPBearer(auto_error=False)


def create_access_token(subject: str) -> str:
    expire_at = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_expire_minutes
    )
    payload = {"sub": subject, "exp": expire_at, "typ": "access"}
    return jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def create_refresh_token(subject: str, family_id: str, token_jti: str, expire_at: datetime) -> str:
    payload = {
        "sub": subject,
        "exp": expire_at,
        "typ": "refresh",
        "fid": family_id,
        "jti": token_jti,
    }
    return jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def decode_refresh_token(token: str) -> dict:
    payload = jwt.decode(
        token,
        settings.jwt_secret_key,
        algorithms=[settings.jwt_algorithm],
    )
    if payload.get("typ") != "refresh":
        raise jwt.InvalidTokenError("Invalid token type")
    if not payload.get("sub") or not payload.get("fid") or not payload.get("jti"):
        raise jwt.InvalidTokenError("Invalid refresh token payload")
    return payload


def generate_refresh_session(subject: str) -> tuple[str, str, str, datetime]:
    token_jti = uuid.uuid4().hex
    family_id = uuid.uuid4().hex
    expire_at = datetime.now(timezone.utc) + timedelta(days=settings.refresh_token_expire_days)
    token = create_refresh_token(subject, family_id, token_jti, expire_at)
    return token, family_id, token_jti, expire_at


def rotate_refresh_session(subject: str, family_id: str) -> tuple[str, str, datetime]:
    token_jti = uuid.uuid4().hex
    expire_at = datetime.now(timezone.utc) + timedelta(days=settings.refresh_token_expire_days)
    token = create_refresh_token(subject, family_id, token_jti, expire_at)
    return token, token_jti, expire_at


def hash_refresh_token(token: str) -> str:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return digest


def build_refresh_cookie_settings() -> dict:
    return {
        "httponly": True,
        "secure": settings.refresh_cookie_secure,
        "samesite": settings.refresh_cookie_samesite,
        "path": settings.refresh_cookie_path,
        "domain": settings.refresh_cookie_domain,
    }


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(auth_scheme),
) -> str:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        if payload.get("typ") != "access":
            raise jwt.InvalidTokenError("Invalid token type")
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    username = payload.get("sub")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return username


# _llama_service: LlamaService | None = None


# def get_llama_service() -> LlamaService | None:
# 	global _llama_service
# 	if _llama_service is not None:
# 		return _llama_service

# 	try:
# 		# _llama_service = LlamaService()
# 		return _llama_service
# 	except Exception:
# 		return None


# _ner_service: NERService | None = None
_diary_service: DiaryService | None = None


# def get_ner_service() -> NERService | None:
# 	global _ner_service
# 	if _ner_service is not None:
# 		return _ner_service
# 	try:
# 		_ner_service = NERService()
# 		return _ner_service
# 	except Exception as e:
# 		print(f"⚠ NERService не инициализирован: {e}")
# 		return None

_ollama_adapter: OllamaAdapter | None = None
def get_ollama_adapter() -> OllamaAdapter:
    global _ollama_adapter
    if _ollama_adapter is not None:
        return _ollama_adapter
    else:
        _ollama_adapter = OllamaAdapter()
        return _ollama_adapter

def get_diary_service() -> DiaryService:
    global _diary_service
    if _diary_service is not None:
        return _diary_service
    else:
        _diary_service = DiaryService(ai_adapter=get_ollama_adapter())
        return _diary_service