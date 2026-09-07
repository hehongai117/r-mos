"""训练步骤证据的归属与完整性校验。"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.evidence import EvidenceBundle, EvidenceItem


async def is_valid_training_step_evidence(
    db: AsyncSession,
    *,
    bundle_id: str | None,
    user_id: int,
    session_id: str,
    step_id: str,
) -> bool:
    """证据包必须存在、完整，并绑定当前学生、会话和步骤。"""
    if not bundle_id:
        return False

    result = await db.execute(
        select(EvidenceBundle)
        .join(EvidenceItem, EvidenceItem.bundle_id == EvidenceBundle.id)
        .where(
            EvidenceBundle.id == bundle_id,
            EvidenceBundle.created_by_user_id == user_id,
            EvidenceBundle.is_sealed.is_(True),
        )
        .limit(1)
    )
    bundle = result.scalar_one_or_none()
    if bundle is None:
        return False

    tags = {str(tag) for tag in (bundle.machine_tags or [])}
    return session_id in tags and step_id in tags
