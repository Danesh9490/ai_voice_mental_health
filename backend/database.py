from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker

engine = create_engine("sqlite:///results.db")
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()


class Result(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, index=True)
    stress_level = Column(String)


Base.metadata.create_all(bind=engine)
