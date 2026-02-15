from __future__ import annotations

from typing import Any

from backend.app.prompt.registry import PromptRegistry

try:
    from langchain_core.prompts import ChatPromptTemplate

    LANGCHAIN_PROMPT_AVAILABLE = True
except Exception:  # pragma: no cover
    ChatPromptTemplate = None
    LANGCHAIN_PROMPT_AVAILABLE = False


class PromptRuntime:
    """Prompt 运行时组装器（System/Policy/Task 三层）。"""

    def __init__(self, registry: PromptRegistry) -> None:
        self.registry = registry

    def build(self, prompt_id: str, variables: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        prompt = self.registry.get_stable_prompt(prompt_id)
        self._validate_variables(prompt["variables_schema"], variables)
        assembled = self._build_with_langchain(prompt, variables) if LANGCHAIN_PROMPT_AVAILABLE else self._build_with_format(prompt, variables)
        return assembled, {
            "prompt_id": prompt["prompt_id"],
            "prompt_version": prompt["version"],
            "prompt_engine": "langchain_chat_prompt" if LANGCHAIN_PROMPT_AVAILABLE else "python_format",
        }

    # 中文注释：优先使用 LangChain 模板渲染，确保 prompt 工程化链路进入真实运行时。
    def _build_with_langchain(self, prompt: dict[str, Any], variables: dict[str, Any]) -> str:
        template = ChatPromptTemplate.from_messages(  # type: ignore[union-attr]
            [
                ("system", "[SYSTEM]\n{template_system}\n\n[POLICY]\n{template_policy}"),
                ("human", "[TASK]\n" + prompt["template_task"]),
            ]
        )
        prompt_value = template.invoke(
            {
                "template_system": prompt["template_system"],
                "template_policy": prompt["template_policy"],
                **variables,
            }
        )
        return "\n\n".join(getattr(m, "content", str(m)) for m in prompt_value.messages)

    def _build_with_format(self, prompt: dict[str, Any], variables: dict[str, Any]) -> str:
        task = prompt["template_task"].format(**variables)
        return (
            f"[SYSTEM]\n{prompt['template_system']}\n\n"
            f"[POLICY]\n{prompt['template_policy']}\n\n"
            f"[TASK]\n{task}"
        )

    def _validate_variables(self, schema: dict[str, Any], variables: dict[str, Any]) -> None:
        required = schema.get("required", [])
        for key in required:
            if key not in variables:
                raise ValueError(f"missing prompt variable: {key}")
