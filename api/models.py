from email.policy import default
from sqlalchemy.sql.expression import null
from database import Base
from sqlalchemy import String, Boolean, Integer, Column, Text


class Admin(Base):
    __tablename__ = 'admin'
    id = Column(Integer, primary_key=True)
    user_name = Column(String(255), nullable=False, unique=True)
    pwd = Column(Text)


class Model(Base):
    __tablename__ = 'model'
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=False)
    F1_score = Column(String(255))
    precision = Column(String(255))
    accuracy = Column(String(255))
    visibility = Column(Boolean, server_default='True',nullable = False)
