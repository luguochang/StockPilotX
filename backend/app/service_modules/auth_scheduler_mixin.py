from __future__ import annotations

from .shared import *

class AuthSchedulerMixin:
    def scheduler_run(self, job_name: str) -> dict[str, Any]:
        """Manually trigger one scheduler job."""
        return self.scheduler.run_once(job_name)

    def scheduler_status(self) -> dict[str, Any]:
        """Return status for all scheduler jobs."""
        return self.scheduler.list_status()

    # ---------- Prediction domain methods ----------

    def scheduler_pause(self, job_name: str) -> dict[str, Any]:
        return self.scheduler.pause(job_name)

    def scheduler_resume(self, job_name: str) -> dict[str, Any]:
        return self.scheduler.resume(job_name)

    # ---------- Web domain methods ----------

    def auth_register(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.web.auth_register(payload["username"], payload["password"], payload.get("tenant_name"))

    def auth_login(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.web.auth_login(payload["username"], payload["password"])

    def auth_me(self, token: str) -> dict[str, Any]:
        return self.web.auth_me(token)

