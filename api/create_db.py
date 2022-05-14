
from database import Base,engine
from models import Admin,Model

print("Creating database ....")

Base.metadata.create_all(engine)