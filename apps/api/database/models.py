import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, Integer, Float,
    DateTime, Boolean, ForeignKey
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content = Column(Text, nullable=False)
    source = Column(String(500), nullable=False)
    chunk_index = Column(Integer, default=0)
    doc_metadata = Column(JSONB, default={})
    embedding = Column(Vector(768))
    created_at = Column(DateTime, default=datetime.utcnow)


class Session(Base):
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow)


class Query(Base):
    __tablename__ = "queries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=True)
    original_question = Column(Text, nullable=False)
    rewritten_question = Column(Text, nullable=True)
    final_answer = Column(Text, nullable=True)
    confidence_score = Column(Float, default=0.0)
    retry_count = Column(Integer, default=0)
    hallucination_detected = Column(Boolean, default=False)
    regeneration_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_id = Column(UUID(as_uuid=True), ForeignKey("queries.id"), nullable=True)
    step_type = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    data = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
