from sqlmodel import SQLModel, create_engine, Session
from .config import settings

engine = create_engine(
    settings.DATABASE_URL,
    echo=True,  # shows SQL in logs; turn off in production
)

def get_session():
    with Session(engine) as session:
        yield session

def init_db():
    #Import models here so SQLModel knows them
    from app import models
    SQLModel.metadata.create_all(bind=engine)