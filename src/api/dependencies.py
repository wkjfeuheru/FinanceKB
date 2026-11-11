"""JWT 单用户鉴权（P0 最小实现）+ 下游数据接口静态 Bearer 鉴权。

JWT 链路用于前端管理界面，密码使用 ``secrets.compare_digest`` 常量时间比较。
用户名/密码配置在 ``.env`` 的 ``JWT_ADMIN_USERNAME`` / ``JWT_ADMIN_PASSWORD``。
数据接口链路（``/api/v1/search/*``、MCP 服务）使用 ``DATA_API_TOKENS`` 静态令牌。

未配置 ``JWT_SECRET`` 或密钥长度不足时，鉴权请求一律拒绝。
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordBearer
from pydantic import BaseModel

from src.core.config import (
    DATA_API_TOKENS,
    JWT_ADMIN_PASSWORD,
    JWT_ADMIN_USERNAME,
    JWT_ALGORITHM,
    JWT_EXPIRE_MINUTES,
    JWT_SECRET,
)

# auto_error=False：未带 token 时返回 None 而非直接 403，
# 由 get_current_user 统一决定鉴权失败响应
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)
data_api_scheme = HTTPBearer(auto_error=False, scheme_name="DataApiBearer")


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = JWT_EXPIRE_MINUTES * 60


def authenticate_user(username: str, password: str) -> bool:
    """常量时间密码比较，防止时序攻击。"""
    user_ok = secrets.compare_digest(username, JWT_ADMIN_USERNAME)
    pwd_ok = secrets.compare_digest(password, JWT_ADMIN_PASSWORD)
    return user_ok and pwd_ok


def create_access_token(subject: str) -> str:
    """生成 JWT，包含签发时间与过期时间。"""
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> str:
    """解码并校验 JWT，返回用户名；失败抛 401。"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    username: str | None = payload.get("sub")
    if username is None:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    return username


async def get_current_user(
    token: Annotated[str | None, Depends(oauth2_scheme)],
) -> str:
    """FastAPI 依赖：校验 Bearer token，返回用户名。

    - 未配置或长度不足的 ``JWT_SECRET`` → 401。
    - 已配置但未带 token / token 非法 → 401。
    """
    if len(JWT_SECRET) < 32:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication is not configured",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_access_token(token)


def _data_api_unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_data_api_token(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(data_api_scheme)
    ],
) -> str:
    """FastAPI 依赖：校验下游数据接口的静态 Bearer Token。

    - 未配置 ``DATA_API_TOKENS`` → 401（宁可拒绝，也不能放行未鉴权访问）；
    - 未带 header / 非 Bearer / token 不匹配 → 统一 401；
    - 逐个 token 做常量时间比较，不使用短路，避免时序侧信道。
    """
    if not DATA_API_TOKENS:
        raise _data_api_unauthorized("Data API authentication is not configured")
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _data_api_unauthorized("Not authenticated")

    matched = False
    for expected in DATA_API_TOKENS:
        matched |= secrets.compare_digest(credentials.credentials, expected)
    if not matched:
        raise _data_api_unauthorized("Invalid token")
    return "data-api"
