from __future__ import annotations

from typing import Any

from app.prompts.base import PromptTemplate, RenderedPrompt


class PromptRegistry:
    """用于管理版本化 Prompt 模板的内存注册表。"""

    def __init__(self) -> None:
        self._templates: dict[tuple[str, str], PromptTemplate] = {}
        self._latest_versions: dict[str, str] = {}

    def register(self, template: PromptTemplate) -> None:
        key = (template.name, template.version)
        self._templates[key] = template
        self._latest_versions[template.name] = template.version

    def register_many(self, templates: list[PromptTemplate]) -> None:
        for template in templates:
            self.register(template)

    def get(self, name: str, *, version: str | None = None) -> PromptTemplate:
        resolved_version = version or self._latest_versions.get(name)
        if not resolved_version:
            raise KeyError(f"prompt template '{name}' is not registered")

        template = self._templates.get((name, resolved_version))
        if template is None:
            raise KeyError(f"prompt template '{name}' with version '{resolved_version}' is not registered")
        return template

    def render(self, name: str, *, variables: dict[str, Any], version: str | None = None) -> RenderedPrompt:
        template = self.get(name, version=version)
        return RenderedPrompt(
            name=template.name,
            version=template.version,
            system_prompt=template.system_template.format(**variables),
            user_prompt=template.user_template.format(**variables),
            variables=variables,
        )

