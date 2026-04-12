from pydantic import BaseModel
class PredictionOut(BaseModel):
    id: int
    result: str
    confidence: float
    user_id: int

    class Config:
        from_attributes = True

class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    email: str

    class Config:
        from_attributes = True
        
class PredictionIn(BaseModel):
    Age: float = 0
    Number_of_sexual_partners: float = 0
    First_sexual_intercourse: float = 0
    Num_of_pregnancies: float = 0
    Smokes: float = 0
    Smokes_years: float = 0
    Smokes_packs_per_year: float = 0
    Hormonal_Contraceptives: float = 0
    Hormonal_Contraceptives_years: float = 0
    IUD: float = 0
    IUD_years: float = 0
    STDs: float = 0
    STDs_number: float = 0