from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from backend.database import get_db, engine
from backend import models, schemas, auth, ml_service

# ✅ Create tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Cervical Cancer Prediction API")

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Health check
@app.get("/health")
def health():
    return {"status": "ok"}


# ✅ Register
@app.post("/register", response_model=schemas.UserOut)
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    try:
        db_user = auth.create_user(db, user)
        return db_user
    except Exception as e:
        import traceback
        print("🔥 REGISTER ERROR:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ✅ Login
@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    return auth.login_user(
        db,
        username=form_data.username,
        password=form_data.password
    )


# ✅ Predict (protected)
@app.post("/predict", response_model=schemas.PredictionOut)
def predict(
    data: schemas.PredictionIn,
    db: Session = Depends(get_db),
    current_user=Depends(auth.get_current_user)
):
    try:
        # use internal schema names (underscore version)
        payload = data.model_dump()
        print("✅ PAYLOAD:", payload)

        result = ml_service.predict(payload)
        print("✅ ML RESULT:", result)

        pred = models.Prediction(
            user_id=current_user.id,
            input_data=str(payload),
            result=result.get("result"),
            confidence=result.get("confidence"),
        )

        db.add(pred)
        db.commit()
        db.refresh(pred)

        return schemas.PredictionOut(
            id=pred.id,
            result=pred.result,
            confidence=pred.confidence,
            user_id=pred.user_id,
        )

    except Exception as e:
        import traceback
        print("🔥 PREDICT ERROR:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ✅ Get user predictions
@app.get("/my-predictions")
def my_predictions(
    db: Session = Depends(get_db),
    current_user=Depends(auth.get_current_user)
):
    return db.query(models.Prediction).filter(
        models.Prediction.user_id == current_user.id
    ).all()