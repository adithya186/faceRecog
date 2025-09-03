import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, text
from sqlalchemy.orm import scoped_session, sessionmaker, declarative_base, relationship


DATABASE_URL = os.environ.get('ATTENDANCE_DB_URL', 'sqlite:///attendance.db')

engine = create_engine(
    DATABASE_URL,
    connect_args={'check_same_thread': False} if DATABASE_URL.startswith('sqlite') else {},
    echo=False
)
db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()
Base.query = db_session.query_property()


class Person(Base):
    __tablename__ = 'people'
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False, unique=True)
    role = Column(String(20), nullable=False, default='student')  # 'student' or 'teacher'
    password_hash = Column(String(255), nullable=True)
    attendances = relationship('Attendance', back_populates='person', cascade='all, delete-orphan')


class Attendance(Base):
    __tablename__ = 'attendance'
    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey('people.id'), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    person = relationship('Person', back_populates='attendances')


def init_db():
    Base.metadata.create_all(bind=engine)
    # Lightweight migration: ensure 'role' column exists on 'people'
    try:
        with engine.connect() as conn:
            res = conn.execute(text("PRAGMA table_info(people)"))
            cols = {row[1] for row in res}
            if 'role' not in cols:
                conn.execute(text("ALTER TABLE people ADD COLUMN role VARCHAR(20) NOT NULL DEFAULT 'student'"))
                conn.commit()
            if 'password_hash' not in cols:
                conn.execute(text("ALTER TABLE people ADD COLUMN password_hash VARCHAR(255)"))
                conn.commit()
    except Exception:
        # Best-effort; ignore if not SQLite or already added
        pass