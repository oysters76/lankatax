from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

_ALLOWED_ROLES = frozenset({"system", "human", "ai"})
_VARIABLE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_REQUIRED_KEYS = ("id", "version", "input_variables", "messages")
_MANIFEST_NAME = "manifest.yaml"


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of keeping the last."""


def _construct_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False):
    loader.flatten_mapping(node)
    seen: set[object] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


class PromptError(Exception):
    """Base class for prompt loading and rendering failures."""


class PromptNotFound(PromptError):
    """A prompt id, version or manifest could not be found."""


class PromptFormatError(PromptError):
    """A prompt file is malformed or internally inconsistent."""


class PromptVariableError(PromptError):
    """A prompt was rendered with missing, extra or invalid variables."""


@dataclass(frozen=True)
class Prompt:
    id: str
    version: str
    messages: tuple[tuple[str, str], ...]
    input_variables: tuple[str, ...]
    sha256: str
    path: Path

    def template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(list(self.messages))

    def format_messages(self, **values: str) -> list[BaseMessage]:
        declared = set(self.input_variables)
        provided = set(values)

        if provided != declared:
            problems = []
            missing = sorted(declared - provided)
            extra = sorted(provided - declared)
            if missing:
                problems.append(f"missing {missing}")
            if extra:
                problems.append(f"unexpected {extra}")
            raise PromptVariableError(f"{self.id}@{self.version}: {', '.join(problems)}")

        non_string = sorted(name for name, value in values.items() if not isinstance(value, str))
        if non_string:
            raise PromptVariableError(
                f"{self.id}@{self.version}: values must be strings: {non_string}"
            )

        try:
            return self.template().format_messages(**values)
        except KeyError as exc:
            raise PromptVariableError(f"{self.id}@{self.version}: missing {exc}") from exc

    @property
    def metadata(self) -> dict[str, str]:
        """Fields to attach to traces so runs can be grouped by prompt version."""
        return {
            "prompt_id": self.id,
            "prompt_version": self.version,
            "prompt_hash": self.sha256,
        }


def prompts_root() -> Path:
    """Directory holding prompt files; override with LANKATAX_PROMPTS_DIR."""
    override = os.environ.get("LANKATAX_PROMPTS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[4] / "prompts"


def get_prompt(prompt_id: str, *, version: str | None = None) -> Prompt:
    """Load a prompt by id and version.

    Version precedence: explicit argument, then LANKATAX_PROMPT_OVERRIDES,
    then the manifest default.
    """
    root = prompts_root() / prompt_id
    if not root.is_dir():
        raise PromptNotFound(f"Prompt {prompt_id!r} not found under {prompts_root()}")

    resolved = version or _env_override(prompt_id) or _manifest_default(prompt_id, root)
    path = root / f"{resolved}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in root.glob("*.yaml") if p.name != _MANIFEST_NAME)
        raise PromptNotFound(
            f"Prompt {prompt_id!r} version {resolved!r} not found in {root}; available: {available}"
        )

    return _build_prompt(_read_yaml(path), prompt_id=prompt_id, version=resolved, path=path)


def _env_override(prompt_id: str) -> str | None:
    raw = os.environ.get("LANKATAX_PROMPT_OVERRIDES", "")
    for entry in raw.split(","):
        name, separator, value = entry.partition("=")
        if separator and name.strip() == prompt_id:
            return value.strip()
    return None


def _manifest_default(prompt_id: str, root: Path) -> str:
    manifest = root / _MANIFEST_NAME
    if not manifest.is_file():
        raise PromptFormatError(f"Missing manifest for prompt {prompt_id!r}: {manifest}")

    default = _read_yaml(manifest).get("default")
    if not isinstance(default, str) or not default:
        raise PromptFormatError(f"{manifest}: 'default' must be a non-empty string")
    return default


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise PromptFormatError(f"Could not read {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise PromptFormatError(f"{path}: expected a YAML mapping")
    return payload


def _build_prompt(payload: dict[str, Any], *, prompt_id: str, version: str, path: Path) -> Prompt:
    missing = [key for key in _REQUIRED_KEYS if key not in payload]
    if missing:
        raise PromptFormatError(f"{path}: missing keys {missing}")

    if payload["id"] != prompt_id:
        raise PromptFormatError(f"{path}: id {payload['id']!r} != directory {prompt_id!r}")
    if payload["version"] != version:
        raise PromptFormatError(f"{path}: version {payload['version']!r} != file {version!r}")

    raw_variables = payload["input_variables"]
    if not isinstance(raw_variables, list) or not all(isinstance(v, str) for v in raw_variables):
        raise PromptFormatError(f"{path}: input_variables must be a list of strings")
    if len(set(raw_variables)) != len(raw_variables):
        raise PromptFormatError(f"{path}: duplicate input_variables")

    raw_messages = payload["messages"]
    if not isinstance(raw_messages, list) or not raw_messages:
        raise PromptFormatError(f"{path}: messages must be a non-empty list")

    messages: list[tuple[str, str]] = []
    for index, item in enumerate(raw_messages):
        if not isinstance(item, dict):
            raise PromptFormatError(f"{path}: message {index} must be a mapping")
        role, content = item.get("role"), item.get("content")
        if role not in _ALLOWED_ROLES:
            raise PromptFormatError(
                f"{path}: message {index} role {role!r} not in {sorted(_ALLOWED_ROLES)}"
            )
        if not isinstance(content, str):
            raise PromptFormatError(f"{path}: message {index} content must be a string")
        messages.append((role, content))

    try:
        found = list(ChatPromptTemplate.from_messages(messages).input_variables)
    except (ValueError, TypeError) as exc:
        raise PromptFormatError(f"{path}: invalid message template: {exc}") from exc

    invalid = sorted(name for name in found if not _VARIABLE_RE.match(name))
    if invalid:
        raise PromptFormatError(f"{path}: placeholders must be lowercase snake_case, got {invalid}")

    declared = set(raw_variables)
    if set(found) != declared:
        undeclared = sorted(set(found) - declared)
        unused = sorted(declared - set(found))
        raise PromptFormatError(
            f"{path}: input_variables mismatch (undeclared: {undeclared}, unused: {unused})"
        )

    digest = hashlib.sha256(
        json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return Prompt(
        id=prompt_id,
        version=version,
        messages=tuple(messages),
        input_variables=tuple(raw_variables),
        sha256=digest,
        path=path,
    )
