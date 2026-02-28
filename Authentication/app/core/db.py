from sqlmodel import SQLModel, create_engine, Session
from .config import settings

engine = create_engine(
    settings.DATABASE_URL,
    echo=True,  # shows SQL in logs; turn off in production
    pool_pre_ping=True,  # verify connections before using
    pool_size=5,  # connection pool size
    max_overflow=10,  # max connections beyond pool_size
)

def get_session():
    with Session(engine) as session:
        yield session

def init_db():
    #Import models here so SQLModel knows them
    from app import models
    SQLModel.metadata.create_all(bind=engine)