# build a schema using pydantic
from typing import Text, Optional
from pydantic import BaseModel

class Admin(BaseModel):
 
    user_name : str
    pwd : str


    class Config:
        orm_mode = True

class Model(BaseModel):
    id : int
    name : str
    description : Text
    F1_score : str
    precision : str
    accuracy : str
    visibility : bool

    class Config:
        orm_mode = True
        
class Inputs(BaseModel):
    text: str
    selectedIndex: int

class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None
