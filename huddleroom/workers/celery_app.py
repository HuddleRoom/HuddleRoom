try:
    import logging
    from datetime import datetime, timezone
    import litellm
    litellm.drop_params = True
    from celery import Celery
    from huddleroom.config import settings, validate_supported_settings
    from huddleroom.services.llm_debug_logging import register_litellm_debug_logger

    _logger = logging.getLogger(__name__)

    validate_supported_settings(settings)

    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)
    register_litellm_debug_logger()

    if settings.redis_url is None:
        _logger.warning("Celery configured without REDIS_URL — using localhost fallback. Set HUDDLEROOM_REDIS_URL for production.")

    app = Celery("rally")    # pylint: disable=invalid-name
    app.conf.update(
        broker_url=settings.redis_url or "redis://localhost:6379/0",
        result_backend=settings.redis_url or "redis://localhost:6379/0",
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
     )
    app.autodiscover_tasks(["huddleroom.workers.session_tasks", "huddleroom.workers.trigger_tasks", "huddleroom.workers.orchestration_tasks", "huddleroom.workers.orchestration_recovery_tasks"])

    app.conf.beat_schedule = {
         "evaluate-cron-triggers": {
             "task": "rally.workers.trigger_tasks.evaluate_cron_triggers",
             "schedule": 60.0,
         },
         "supervise-orchestration": {
             "task": "rally.workers.orchestration_tasks.supervise_orchestration",
            "schedule": settings.orchestration_reconcile_interval_seconds,
         },
         "recover-orchestration": {
             "task": "rally.workers.orchestration_recovery_tasks.recover_orchestration",
             "schedule": settings.orchestration_reconcile_interval_seconds,
         },
     }

    import huddleroom.workers.meeting_tasks
    huddleroom.workers.meeting_tasks.register_tasks(app)

    from celery.signals import before_task_publish, worker_ready

    @before_task_publish.connect
    def _stamp_recovery_publish(sender=None, headers=None, **_kwargs):
        if sender == "rally.workers.orchestration_recovery_tasks.recover_orchestration" and headers is not None:
            headers["orchestration_recovery_ready_at"] = _utcnow().isoformat()

    @worker_ready.connect
    def _recover_orchestration_when_worker_ready(**_kwargs):
        app.send_task(
            "rally.workers.orchestration_recovery_tasks.recover_orchestration",
            args=[_utcnow().isoformat()],
        )

    def _unsupported_celery_service(*_args, **_kwargs):
        raise RuntimeError("Celery workers and beat are not yet supported; use huddleroom serve.")

    app.Worker = _unsupported_celery_service
    app.Beat = _unsupported_celery_service

except ImportError:
    app = None   # type: ignore[assignment]
