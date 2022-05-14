from click import echo
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# à mettre dans un fichier à part
DATABASE_URL = "postgresql://gjfzpkxaqnkczm:644db5ec9ba0f9a1bfb7f77366a19af01fd38f6dd1e3ad9fc10220be32919927@ec2-54-86-224-85.compute-1.amazonaws.com:5432/dms5ftsc3838v"
engine = create_engine(DATABASE_URL,echo= True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()