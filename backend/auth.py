from passlib.context import CryptContext
from jose import jwt, JWTError
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from . import models, schemas
from .database import get_db
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
SECRET = "mysecretkey123"
ALGO = "HS256"

pwd = CryptContext(schemes=["pbkdf2_sha256"])
oauth2 = HTTPBearer()


# ✅ Register
def create_user(db: Session, user: schemas.UserCreate):
    existing = db.query(models.User).filter(models.User.email == user.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed = pwd.hash(user.password)

    db_user = models.User(
        username=user.username,   # ✅ ADD THIS
        email=user.email,
        hashed_password=hashed
    )

    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return db_user


# ✅ Login (FIXED)
def login_user(db: Session, username: str, password: str):
    # ✅ search by username (NOT email)
    db_user = db.query(models.User).filter(models.User.username == username).first()

    if not db_user:
        raise HTTPException(status_code=401, detail="User not found")

    if not pwd.verify(password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Wrong password")

    token = jwt.encode({"sub": db_user.email}, SECRET, algorithm=ALGO)

    return {
        "access_token": token,
        "token_type": "bearer"
    }

# ✅ Get current user
def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(oauth2),
    db: Session = Depends(get_db)
):
    try:
        token = credentials.credentials  # 🔥 extract token

        payload = jwt.decode(token, SECRET, algorithms=[ALGO])
        email = payload.get("sub")

        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        user = db.query(models.User).filter(models.User.email == email).first()

        if not user:
            raise HTTPException(status_code=401, detail="User not found")

        return user

    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")