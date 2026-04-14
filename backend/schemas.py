from pydantic import BaseModel, Field, ConfigDict


class PredictionOut(BaseModel):
    id: int
    result: str
    confidence: float
    user_id: int

    model_config = ConfigDict(from_attributes=True)


class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    email: str

    model_config = ConfigDict(from_attributes=True)


class PredictionIn(BaseModel):
    Age: float = Field(0, ge=0)
    Number_of_sexual_partners: float = Field(0, ge=0, alias="Number of sexual partners")
    First_sexual_intercourse: float = Field(0, ge=0, alias="First sexual intercourse")
    Num_of_pregnancies: float = Field(0, ge=0, alias="Num of pregnancies")
    Smokes: float = Field(0, ge=0)
    Smokes_years: float = Field(0, ge=0, alias="Smokes (years)")
    Smokes_packs_per_year: float = Field(0, ge=0, alias="Smokes (packs/year)")
    Hormonal_Contraceptives: float = Field(0, ge=0, alias="Hormonal Contraceptives")
    Hormonal_Contraceptives_years: float = Field(0, ge=0, alias="Hormonal Contraceptives (years)")
    IUD: float = Field(0, ge=0)
    IUD_years: float = Field(0, ge=0, alias="IUD (years)")
    STDs: float = Field(0, ge=0)
    STDs_number: float = Field(0, ge=0, alias="STDs (number)")

    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
    )