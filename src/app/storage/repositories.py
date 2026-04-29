from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.schemas import AdvisorResponse, AnalystReport, AnalystTask, ClientProfile
from app.storage.models import (
    AnalystTaskORM,
    ClientProfileORM,
    ConversationSessionORM,
    ConversationTurnORM,
)


def upsert_client_profile(session: Session, profile: ClientProfile) -> ClientProfileORM:
    existing = session.execute(
        select(ClientProfileORM).where(ClientProfileORM.name == profile.name)
    ).scalar_one_or_none()
    if existing:
        existing.age = profile.age
        existing.risk_aversion = profile.risk_aversion
        existing.annual_income = profile.annual_income
        existing.liquid_assets = profile.liquid_assets
        existing.raw_profile_json = profile.model_dump_json()
        session.flush()
        return existing

    row = ClientProfileORM(
        name=profile.name,
        age=profile.age,
        risk_aversion=profile.risk_aversion,
        annual_income=profile.annual_income,
        liquid_assets=profile.liquid_assets,
        raw_profile_json=profile.model_dump_json(),
    )
    session.add(row)
    session.flush()
    return row


def create_session(session: Session, session_id: str, client_id: int) -> ConversationSessionORM:
    row = ConversationSessionORM(session_id=session_id, client_id=client_id)
    session.add(row)
    session.flush()
    return row


def get_or_create_session(
    session: Session, session_id: str, profile: ClientProfile
) -> ConversationSessionORM:
    existing = session.execute(
        select(ConversationSessionORM).where(ConversationSessionORM.session_id == session_id)
    ).scalar_one_or_none()
    if existing:
        return existing
    client_row = upsert_client_profile(session, profile)
    return create_session(session, session_id=session_id, client_id=client_row.id)


def add_turn(
    session: Session,
    session_row: ConversationSessionORM,
    turn_index: int,
    client_message: str,
    advisor_response: AdvisorResponse,
) -> ConversationTurnORM:
    row = ConversationTurnORM(
        session_id=session_row.id,
        turn_index=turn_index,
        client_message=client_message,
        advisor_response=advisor_response.message,
        recommendation=advisor_response.recommendation,
    )
    session.add(row)
    session.flush()
    return row


def add_analyst_report(
    session: Session, turn_row: ConversationTurnORM, task: AnalystTask, report: AnalystReport
) -> AnalystTaskORM:
    row = AnalystTaskORM(
        turn_id=turn_row.id,
        external_task_id=task.task_id,
        question=task.question,
        summary=report.summary,
        evidence_count=len(report.findings),
    )
    session.add(row)
    session.flush()
    return row


def list_turns(session: Session, session_id: str) -> Sequence[ConversationTurnORM]:
    return session.execute(
        select(ConversationTurnORM)
        .join(ConversationSessionORM, ConversationTurnORM.session_id == ConversationSessionORM.id)
        .where(ConversationSessionORM.session_id == session_id)
        .order_by(ConversationTurnORM.turn_index.asc())
    ).scalars().all()
