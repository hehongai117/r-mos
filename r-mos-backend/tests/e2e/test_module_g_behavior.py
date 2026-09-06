"""RMOS-S3-007 模块 G：训练路由覆盖与当前行为缺陷记录。

本文件只建立第一步安全网：用真实 HTTP、运行时路由、ORM 元数据和服务调用
固定当前行为。疑似缺陷只记录、不在本批修改生产实现。
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import app.models as app_models  # noqa: F401  # ensure metadata is fully loaded
from app.core.database import get_db
from app.models.base import Base
from app.models.evidence import EvidenceBundle
from app.models.school import School
from app.models.skill_profile import StudentSkillProfile, StudentWeakStep
from app.models.teaching import Enrollment, TeachingClass
from app.models.training import SessionStepRecord, TrainingSession
from app.models.training_submission import TrainingSubmission
from app.services.memory.hub import MemoryHub
from app.services.training.project_generator import ProjectGenerator
from main import app


pytestmark = [pytest.mark.e2e, pytest.mark.characterization]

TEST_SCHOOL_NAME = "测试学校"
OTHER_SCHOOL_NAME = "模块G外校"

MODULE_G_ROUTES = {
    ("POST", "/api/v1/training/projects/generate"),
    ("POST", "/api/v1/training/workbench/draft"),
    ("POST", "/api/v1/training/workbench/evidence"),
    ("POST", "/api/v1/training/workbench/sessions/{session_id}/steps/{step_id}/submit"),
    ("POST", "/api/v1/training/workbench/ask"),
    ("POST", "/api/v1/training/sessions"),
    ("GET", "/api/v1/training/sessions/{session_id}"),
    ("GET", "/api/v1/training/sessions/{session_id}/detail"),
    ("PATCH", "/api/v1/training/sessions/{session_id}/pause"),
    ("PATCH", "/api/v1/training/sessions/{session_id}/resume"),
    ("PATCH", "/api/v1/training/sessions/{session_id}/abandon"),
    ("POST", "/api/v1/training/sessions/{session_id}/submit"),
    ("POST", "/api/v1/training/sessions/{session_id}/force-submit"),
    ("POST", "/api/v1/training/sessions/{session_id}/steps"),
    ("GET", "/api/v1/training/sessions/{session_id}/steps"),
    ("GET", "/api/v1/training/users/{user_id}/sessions"),
    ("GET", "/api/v1/training/users/{user_id}/active-session"),
    ("GET", "/api/v1/training/feedback/{session_id}"),
    ("GET", "/api/v1/students/{user_id}/profile"),
    ("GET", "/api/v1/students/{user_id}/weak-steps"),
}


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _register_and_login(
    client: TestClient,
    *,
    email_prefix: str,
    school_name: str,
    role: str = "teacher",
    teacher_id: int | None = None,
) -> tuple[int, str]:
    email = f"{email_prefix}_{uuid4().hex[:8]}@example.com"
    payload: dict[str, object] = {
        "email": email,
        "password": "StrongPass123",
        "full_name": "Module G User",
        "role": role,
        "school_name": school_name,
    }
    if teacher_id is not None:
        payload["teacher_id"] = teacher_id
    registered = client.post("/api/v1/auth/register", json=payload)
    assert registered.status_code == 201, registered.text
    logged_in = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "StrongPass123"},
    )
    assert logged_in.status_code == 200, logged_in.text
    return int(registered.json()["user_id"]), logged_in.json()["access_token"]


@pytest.fixture(scope="module")
def module_g_env() -> dict:
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    async def _init_models() -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
            await connection.execute(
                School.__table__.insert(),
                [{"name": TEST_SCHOOL_NAME}, {"name": OTHER_SCHOOL_NAME}],
            )

    asyncio.run(_init_models())
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _override_get_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    app.state.test_sessionmaker = session_factory
    client = TestClient(app)

    owner_teacher_id, owner_teacher_token = _register_and_login(
        client,
        email_prefix="module_g_owner_teacher",
        school_name=TEST_SCHOOL_NAME,
    )
    peer_teacher_id, peer_teacher_token = _register_and_login(
        client,
        email_prefix="module_g_peer_teacher",
        school_name=TEST_SCHOOL_NAME,
    )
    foreign_teacher_id, foreign_teacher_token = _register_and_login(
        client,
        email_prefix="module_g_foreign_teacher",
        school_name=OTHER_SCHOOL_NAME,
    )
    student_id, student_token = _register_and_login(
        client,
        email_prefix="module_g_student",
        school_name=TEST_SCHOOL_NAME,
        role="student",
        teacher_id=owner_teacher_id,
    )

    try:
        yield {
            "client": client,
            "session_factory": session_factory,
            "owner_teacher_id": owner_teacher_id,
            "owner_teacher_token": owner_teacher_token,
            "peer_teacher_id": peer_teacher_id,
            "peer_teacher_token": peer_teacher_token,
            "foreign_teacher_id": foreign_teacher_id,
            "foreign_teacher_token": foreign_teacher_token,
            "student_id": student_id,
            "student_token": student_token,
        }
    finally:
        client.close()
        app.dependency_overrides.clear()
        app.state.test_sessionmaker = None
        asyncio.run(engine.dispose())


async def _seed_training_bundle(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    user_id: int,
    project_snapshot: dict | None = None,
    with_submission: bool = True,
) -> dict[str, str]:
    session_id = str(uuid4())
    step_id = f"step-{uuid4().hex[:8]}"
    submission_id = str(uuid4())
    snapshot = project_snapshot or {
        "estimated_time": 30,
        "steps": [
            {
                "id": step_id,
                "title": "安全检查",
                "instruction": "确认工位安全",
                "evidence_hint": "上传现场图片",
                "tools": [],
            }
        ],
    }
    async with session_factory() as session:
        session.add(
            TrainingSession(
                session_id=session_id,
                project_id=str(uuid4()),
                user_id=user_id,
                status="active",
                project_snapshot=snapshot,
            )
        )
        session.add(
            SessionStepRecord(
                record_id=str(uuid4()),
                session_id=session_id,
                step_id=step_id,
                step_index=0,
                status="pending",
                attempt_count=0,
            )
        )
        if with_submission:
            session.add(
                TrainingSubmission(
                    submission_id=submission_id,
                    session_id=session_id,
                    user_id=user_id,
                    submit_type="manual",
                    submitted_at=datetime.now(timezone.utc),
                    payload={"steps_summary": [], "total_duration": 0},
                    score=Decimal("80.00"),
                    total_steps=1,
                    completed_steps=1,
                    failed_steps=0,
                    total_duration=0,
                    feedback={
                        "overall_score": 80.0,
                        "score_breakdown": {"total_score": 80.0},
                        "suggestions": ["继续练习"],
                    },
                )
            )
        await session.commit()
    return {
        "session_id": session_id,
        "step_id": step_id,
        "submission_id": submission_id,
    }


def test_module_g_route_census_uses_runtime_app() -> None:
    """从运行时应用枚举，锁定模块 G 当前注册的 20 条方法与路径组合。"""
    actual: set[tuple[str, str]] = set()
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        if route.endpoint.__module__ in {
            "app.api.v1.endpoints.training",
            "app.api.v1.endpoints.training_workbench",
        }:
            actual.update((method, route.path) for method in route.methods)
    assert actual == MODULE_G_ROUTES


def test_five_tables_have_direct_or_derived_owner() -> None:
    """四张表有直接 user_id；步骤表经非空外键回到训练会话。

    这是当前事实：已知事实中“5 张表全部有 user_id/student_id”的字面表述不成立，
    但 session_step_records 不会成为无归属行，待模块 G 结论中纠正口径。
    """
    for model in (
        TrainingSession,
        TrainingSubmission,
        StudentSkillProfile,
        StudentWeakStep,
    ):
        assert "user_id" in model.__table__.columns
        assert model.__table__.columns.user_id.nullable is False

    assert "user_id" not in SessionStepRecord.__table__.columns
    assert "student_id" not in SessionStepRecord.__table__.columns
    session_fk = SessionStepRecord.__table__.columns.session_id
    assert session_fk.nullable is False
    assert {fk.target_fullname for fk in session_fk.foreign_keys} == {
        "training_sessions.session_id"
    }


@pytest.mark.parametrize(
    "kind",
    ("detail", "feedback", "sessions", "profile", "weak-steps"),
)
def test_cross_school_teacher_is_rejected_by_scoped_reads(
    module_g_env: dict, kind: str
) -> None:
    """跨校教师读取会话、提交或画像必须 404；同一守卫当前能拒绝。"""
    seeded = asyncio.run(
        _seed_training_bundle(
            module_g_env["session_factory"], user_id=module_g_env["student_id"]
        )
    )
    paths = {
        "detail": f"/api/v1/training/sessions/{seeded['session_id']}/detail",
        "feedback": f"/api/v1/training/feedback/{seeded['session_id']}",
        "sessions": f"/api/v1/training/users/{module_g_env['student_id']}/sessions",
        "profile": f"/api/v1/students/{module_g_env['student_id']}/profile",
        "weak-steps": f"/api/v1/students/{module_g_env['student_id']}/weak-steps",
    }
    response = module_g_env["client"].get(
        paths[kind], headers=_auth(module_g_env["foreign_teacher_token"])
    )
    assert response.status_code == 404, response.text


@pytest.mark.parametrize(
    "kind",
    ("detail", "feedback", "sessions", "profile", "weak-steps"),
)
def test_same_school_teacher_is_allowed_by_scoped_reads(
    module_g_env: dict, kind: str
) -> None:
    """同校教师读取学生会话、提交或画像是既定设计，必须保持放行。"""
    seeded = asyncio.run(
        _seed_training_bundle(
            module_g_env["session_factory"], user_id=module_g_env["student_id"]
        )
    )
    paths = {
        "detail": f"/api/v1/training/sessions/{seeded['session_id']}/detail",
        "feedback": f"/api/v1/training/feedback/{seeded['session_id']}",
        "sessions": f"/api/v1/training/users/{module_g_env['student_id']}/sessions",
        "profile": f"/api/v1/students/{module_g_env['student_id']}/profile",
        "weak-steps": f"/api/v1/students/{module_g_env['student_id']}/weak-steps",
    }
    response = module_g_env["client"].get(
        paths[kind], headers=_auth(module_g_env["peer_teacher_token"])
    )
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("kind", ("session", "steps", "active-session"))
def test_unscoped_reads_leak_cross_school_current_behavior(
    module_g_env: dict, kind: str
) -> None:
    """这是当前行为，疑似缺陷 G-AUTH-01：三条读取路由未做归属校验，待模块 G 改造时处置。"""
    fresh_student_id, _ = _register_and_login(
        module_g_env["client"],
        email_prefix=f"module_g_leak_{kind}",
        school_name=TEST_SCHOOL_NAME,
        role="student",
        teacher_id=module_g_env["owner_teacher_id"],
    )
    seeded = asyncio.run(
        _seed_training_bundle(
            module_g_env["session_factory"], user_id=fresh_student_id
        )
    )
    paths = {
        "session": f"/api/v1/training/sessions/{seeded['session_id']}",
        "steps": f"/api/v1/training/sessions/{seeded['session_id']}/steps",
        "active-session": (
            f"/api/v1/training/users/{fresh_student_id}/active-session"
        ),
    }
    response = module_g_env["client"].get(
        paths[kind], headers=_auth(module_g_env["foreign_teacher_token"])
    )
    assert response.status_code == 200, response.text
    if kind == "steps":
        assert response.json()[0]["session_id"] == seeded["session_id"]
    else:
        assert response.json()["user_id"] == fresh_student_id


def test_active_session_crashes_with_multiple_active_rows_current_behavior(
    module_g_env: dict,
) -> None:
    """这是当前行为，疑似缺陷 G-DATA-03：同一学生多个进行中会话会令断点续训接口报错，待模块 G 改造时处置。"""
    fresh_student_id, fresh_student_token = _register_and_login(
        module_g_env["client"],
        email_prefix="module_g_multiple_active",
        school_name=TEST_SCHOOL_NAME,
        role="student",
        teacher_id=module_g_env["owner_teacher_id"],
    )
    asyncio.run(
        _seed_training_bundle(
            module_g_env["session_factory"],
            user_id=fresh_student_id,
            with_submission=False,
        )
    )
    asyncio.run(
        _seed_training_bundle(
            module_g_env["session_factory"],
            user_id=fresh_student_id,
            with_submission=False,
        )
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get(
            f"/api/v1/training/users/{fresh_student_id}/active-session",
            headers=_auth(fresh_student_token),
        )
    assert response.status_code == 500


@pytest.mark.parametrize(
    ("path", "payload"),
    (
        ("/api/v1/training/projects/generate", {"difficulty": "medium"}),
        ("/api/v1/training/workbench/draft", {"robot_model": ""}),
        (
            "/api/v1/training/workbench/sessions/missing/steps/step/submit",
            {"note": "missing step_index"},
        ),
        (
            "/api/v1/training/workbench/ask",
            {"session_id": "missing", "step_id": "step", "question": ""},
        ),
        ("/api/v1/training/sessions", {"project_snapshot": {}}),
        (
            "/api/v1/training/sessions/missing/submit",
            {"user_id": "not-an-integer"},
        ),
        (
            "/api/v1/training/sessions/missing/force-submit",
            {"teacher_id": "not-an-integer"},
        ),
        (
            "/api/v1/training/sessions/missing/steps",
            {"step_index": 0, "status": "pass"},
        ),
    ),
)
def test_json_write_routes_reject_invalid_input_with_422(
    module_g_env: dict, path: str, payload: dict
) -> None:
    response = module_g_env["client"].post(
        path,
        headers=_auth(module_g_env["student_token"]),
        json=payload,
    )
    assert response.status_code == 422, response.text


def test_create_and_generate_bind_requested_user_to_actor(
    module_g_env: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """创建会话与生成项目都拒绝冒用他人编号，同时允许本人。"""
    client = module_g_env["client"]
    student_headers = _auth(module_g_env["student_token"])
    foreign_id = module_g_env["foreign_teacher_id"]

    denied_create = client.post(
        "/api/v1/training/sessions",
        headers=student_headers,
        json={
            "user_id": foreign_id,
            "project_id": str(uuid4()),
            "project_snapshot": {},
        },
    )
    assert denied_create.status_code == 403

    allowed_create = client.post(
        "/api/v1/training/sessions",
        headers=student_headers,
        json={
            "user_id": module_g_env["student_id"],
            "project_id": str(uuid4()),
            "project_snapshot": {},
        },
    )
    assert allowed_create.status_code == 200
    assert allowed_create.json()["user_id"] == module_g_env["student_id"]

    denied_generate = client.post(
        "/api/v1/training/projects/generate",
        headers=student_headers,
        json={"user_id": foreign_id},
    )
    assert denied_generate.status_code == 403

    async def _fake_generate(self, intent, user_id):  # noqa: ANN001
        yield {
            "status": "completed",
            "project": SimpleNamespace(
                project_id="module-g-project",
                title="本人训练项目",
                description="identity-bound",
                estimated_time=30,
                difficulty_cap="medium",
            ),
        }

    monkeypatch.setattr(ProjectGenerator, "generate", _fake_generate)
    allowed_generate = client.post(
        "/api/v1/training/projects/generate",
        headers=student_headers,
        json={"user_id": module_g_env["student_id"]},
    )
    assert allowed_generate.status_code == 200
    assert "module-g-project" in allowed_generate.text


def test_workbench_session_writes_reject_non_owner(module_g_env: dict) -> None:
    """工作台步骤提交和追问拒绝其他用户；所有者正常路径已有原有测试覆盖。"""
    seeded = asyncio.run(
        _seed_training_bundle(
            module_g_env["session_factory"],
            user_id=module_g_env["student_id"],
            with_submission=False,
        )
    )
    headers = _auth(module_g_env["foreign_teacher_token"])
    submit = module_g_env["client"].post(
        (
            f"/api/v1/training/workbench/sessions/{seeded['session_id']}"
            f"/steps/{seeded['step_id']}/submit"
        ),
        headers=headers,
        json={"step_index": 0},
    )
    ask = module_g_env["client"].post(
        "/api/v1/training/workbench/ask",
        headers=headers,
        json={
            "session_id": seeded["session_id"],
            "step_id": seeded["step_id"],
            "question": "下一步是什么？",
        },
    )
    assert submit.status_code == 403
    assert ask.status_code == 403


@pytest.mark.parametrize("kind", ("submit", "ask"))
def test_workbench_session_writes_missing_id_returns_404(
    module_g_env: dict, kind: str
) -> None:
    missing = f"missing-{uuid4()}"
    if kind == "submit":
        response = module_g_env["client"].post(
            f"/api/v1/training/workbench/sessions/{missing}/steps/step/submit",
            headers=_auth(module_g_env["student_token"]),
            json={"step_index": 0},
        )
    else:
        response = module_g_env["client"].post(
            "/api/v1/training/workbench/ask",
            headers=_auth(module_g_env["student_token"]),
            json={
                "session_id": missing,
                "step_id": "step",
                "question": "下一步是什么？",
            },
        )
    assert response.status_code == 404, response.text


def test_update_step_missing_session_returns_404(module_g_env: dict) -> None:
    response = module_g_env["client"].post(
        f"/api/v1/training/sessions/missing-{uuid4()}/steps",
        headers=_auth(module_g_env["student_token"]),
        json={"step_id": "step", "step_index": 0, "status": "pass"},
    )
    assert response.status_code == 404, response.text


def test_fake_evidence_bundle_marks_step_pass_current_behavior(
    module_g_env: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """这是当前行为，疑似缺陷 G-EVID-01：任意非空证据编号可判步骤通过，待模块 G 改造时处置。"""
    from app.services.training.workbench_execution_service import (
        TrainingWorkbenchExecutionService,
    )

    async def _fake_explanation(self, **kwargs):  # noqa: ANN001
        return "当前规则判定通过"

    monkeypatch.setattr(
        TrainingWorkbenchExecutionService,
        "_generate_verdict_explanation",
        _fake_explanation,
    )
    seeded = asyncio.run(
        _seed_training_bundle(
            module_g_env["session_factory"],
            user_id=module_g_env["student_id"],
            with_submission=False,
        )
    )
    fake_bundle_id = f"does-not-exist-{uuid4()}"
    response = module_g_env["client"].post(
        (
            f"/api/v1/training/workbench/sessions/{seeded['session_id']}"
            f"/steps/{seeded['step_id']}/submit"
        ),
        headers=_auth(module_g_env["student_token"]),
        json={
            "step_index": 0,
            "evidence_bundle_id": fake_bundle_id,
            "tools_confirmed": [],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "pass"
    assert response.json()["evidence_bundle_id"] == fake_bundle_id

    async def _evidence_still_missing() -> bool:
        async with module_g_env["session_factory"]() as session:
            return await session.get(EvidenceBundle, fake_bundle_id) is None

    assert asyncio.run(_evidence_still_missing()) is True


def test_force_submit_uses_any_class_scope_current_behavior(module_g_env: dict) -> None:
    """这是当前行为，疑似缺陷 G-AUTH-02：force-submit 只问教师是否在任一班教过学生，待模块 G 改造时处置。"""
    async def _seed_classes_and_session() -> str:
        async with module_g_env["session_factory"]() as session:
            teacher_a_class = TeachingClass(
                name=f"教师A班-{uuid4().hex[:6]}",
                teacher_id=module_g_env["owner_teacher_id"],
            )
            teacher_b_class = TeachingClass(
                name=f"教师B班-{uuid4().hex[:6]}",
                teacher_id=module_g_env["peer_teacher_id"],
            )
            session.add_all([teacher_a_class, teacher_b_class])
            await session.flush()
            session.add_all(
                [
                    Enrollment(
                        class_id=teacher_a_class.id,
                        student_id=module_g_env["student_id"],
                    ),
                    Enrollment(
                        class_id=teacher_b_class.id,
                        student_id=module_g_env["student_id"],
                    ),
                ]
            )
            training_session = TrainingSession(
                session_id=str(uuid4()),
                project_id=str(uuid4()),
                user_id=module_g_env["student_id"],
                status="active",
                project_snapshot={
                    "class_id": teacher_b_class.id,
                    "steps": [],
                    "estimated_time": 30,
                },
            )
            session.add(training_session)
            await session.commit()
            return training_session.session_id

    session_id = asyncio.run(_seed_classes_and_session())
    owner_attempt = module_g_env["client"].post(
        f"/api/v1/training/sessions/{session_id}/force-submit",
        headers=_auth(module_g_env["student_token"]),
        json={},
    )
    assert owner_attempt.status_code == 403

    unrelated_teacher = module_g_env["client"].post(
        f"/api/v1/training/sessions/{session_id}/force-submit",
        headers=_auth(module_g_env["owner_teacher_token"]),
        json={},
    )
    assert unrelated_teacher.status_code == 200, unrelated_teacher.text
    assert unrelated_teacher.json()["submit_type"] == "teacher"


def test_feedback_role_query_grants_teacher_view_current_behavior(
    module_g_env: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """这是当前行为，疑似缺陷 G-AUTH-03：学生可用 role=teacher 切换教师反馈视角，待模块 G 改造时处置。"""
    seeded = asyncio.run(
        _seed_training_bundle(
            module_g_env["session_factory"], user_id=module_g_env["student_id"]
        )
    )

    async def _clear_feedback() -> None:
        async with module_g_env["session_factory"]() as session:
            result = await session.execute(
                select(TrainingSubmission).where(
                    TrainingSubmission.submission_id == seeded["submission_id"]
                )
            )
            result.scalar_one().feedback = None
            await session.commit()

    asyncio.run(_clear_feedback())

    async def _fake_generate(self, submission_id, role):  # noqa: ANN001
        return SimpleNamespace(
            overall_score=88.0,
            score_breakdown={"total_score": 88.0},
            step_analyses=[],
            suggestions=[],
            next_learning_plan="",
            teaching_diagnosis=("教师专属诊断" if role.value == "teacher" else None),
            ranking_percentile=90.0 if role.value == "teacher" else None,
            hint_level_suggestion=None,
        )

    monkeypatch.setattr(
        "app.api.v1.endpoints.training.FeedbackGenerator.generate", _fake_generate
    )
    response = module_g_env["client"].get(
        f"/api/v1/training/feedback/{seeded['session_id']}?role=teacher",
        headers=_auth(module_g_env["student_token"]),
    )
    assert response.status_code == 200, response.text
    assert response.json()["teaching_diagnosis"] == "教师专属诊断"
    assert response.json()["ranking_percentile"] == 90.0


def test_profile_get_creates_row_but_direct_profile_write_is_absent(
    module_g_env: dict,
) -> None:
    """这是当前行为，疑似缺陷 G-DATA-01：GET 会创建画像；没有直接画像写接口，待模块 G 改造时处置。"""
    async def _delete_and_count() -> int:
        async with module_g_env["session_factory"]() as session:
            result = await session.execute(
                select(StudentSkillProfile).where(
                    StudentSkillProfile.user_id == module_g_env["student_id"]
                )
            )
            profile = result.scalar_one_or_none()
            if profile is not None:
                await session.delete(profile)
                await session.commit()
            return 0

    asyncio.run(_delete_and_count())
    read = module_g_env["client"].get(
        f"/api/v1/students/{module_g_env['student_id']}/profile",
        headers=_auth(module_g_env["student_token"]),
    )
    assert read.status_code == 200
    assert read.json()["overall_level"] == 1

    direct_write = module_g_env["client"].patch(
        f"/api/v1/students/{module_g_env['student_id']}/profile",
        headers=_auth(module_g_env["student_token"]),
        json={"overall_level": 5, "score_safety": 100},
    )
    assert direct_write.status_code == 405


def test_student_self_report_updates_skill_profile_current_behavior(
    module_g_env: dict,
) -> None:
    """这是当前行为，疑似缺陷 G-DATA-02：学生自报 pass 且无证据也会写入技能画像，待模块 G 改造时处置。"""
    client = module_g_env["client"]
    headers = _auth(module_g_env["student_token"])
    created = client.post(
        "/api/v1/training/sessions",
        headers=headers,
        json={
            "user_id": module_g_env["student_id"],
            "project_id": str(uuid4()),
            "project_snapshot": {"estimated_time": 30, "steps": []},
        },
    )
    assert created.status_code == 200, created.text
    session_id = created.json()["session_id"]
    self_reported = client.post(
        f"/api/v1/training/sessions/{session_id}/steps",
        headers=headers,
        json={
            "step_id": "self-reported-step",
            "step_index": 0,
            "status": "pass",
            "attempt_count": 1,
            "tools_confirmed": [{"tool_id": "tool", "status": "confirmed"}],
        },
    )
    assert self_reported.status_code == 200, self_reported.text
    submitted = client.post(
        f"/api/v1/training/sessions/{session_id}/submit",
        headers=headers,
        json={"confirm_incomplete": True},
    )
    assert submitted.status_code == 200, submitted.text

    async def _load_profile() -> StudentSkillProfile:
        async with module_g_env["session_factory"]() as session:
            result = await session.execute(
                select(StudentSkillProfile).where(
                    StudentSkillProfile.user_id == module_g_env["student_id"]
                )
            )
            return result.scalar_one()

    profile = asyncio.run(_load_profile())
    assert profile.total_sessions >= 1
    assert float(profile.score_procedure) == 100.0


def test_project_generator_omits_knowledge_viewer_current_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """这是当前行为，疑似缺陷 G-KNOW-01：走统一知识入口但未传当前用户，校内私有知识被排除，待模块 G 改造时处置。"""
    captured: dict = {}

    class _FakeKnowledgeHub:
        async def search(self, **kwargs):
            captured.update(kwargs)
            return [{"title": "public", "content": "public", "score": 1.0}]

    async def _no_embedding(query):  # noqa: ANN001
        return None

    monkeypatch.setattr(
        "app.services.training.project_generator.query_embedding_service.embed_query",
        _no_embedding,
    )
    generator = ProjectGenerator(SimpleNamespace())
    generator.knowledge_hub = _FakeKnowledgeHub()
    result = asyncio.run(
        generator._retrieve_knowledge(
            SimpleNamespace(brand="ATOM", model="01", focus_areas=[])
        )
    )
    assert result[0]["title"] == "public"
    assert "viewer_user_id" not in captured


def test_short_term_fallback_is_not_user_scoped_current_behavior() -> None:
    """这是当前行为，疑似缺陷 G-MEM-01：fallback 可向另一 user_id 返回同一会话业务数据，待模块 G 改造时处置。"""
    async def _exercise() -> tuple[list, list]:
        hub = MemoryHub()
        hub.short_term._client = None
        await hub.write(
            session_id="shared-session-id",
            user_id="student-a",
            data={"type": "recommendation", "student_id": "student-a"},
        )
        owner_read = await hub.read(
            session_id="shared-session-id", user_id="student-a"
        )
        other_read = await hub.read(
            session_id="shared-session-id", user_id="student-b"
        )
        return owner_read, other_read

    owner_read, other_read = asyncio.run(_exercise())
    assert owner_read[0].data["student_id"] == "student-a"
    assert other_read[0].data["student_id"] == "student-a"
