from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml
from langchain_core.load import dumpd
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from lankatax_ai.backend.types import PromptSource

_ALLOWED_ROLES = frozenset({"system", "human", "ai"})
_VARIABLE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_REQUIRED_KEYS = ("id", "version", "input_variables", "messages")
_MANIFEST_NAME = "manifest.yaml"

_hub_client: Any = None
_hub_client_lock = threading.Lock()


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of keeping the last."""


def _construct_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
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


class PromptSourceError(PromptError):
    """The prompt source is invalid or a hub prompt could not be pulled."""


@dataclass(frozen=True)
class Prompt:
    id: str
    version: str
    input_variables: tuple[str, ...]
    sha256: str
    source: PromptSource
    path: Path | None

    def template(self) -> ChatPromptTemplate:
        raise NotImplementedError

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
            "prompt_source": self.source.value,
        }


@dataclass(frozen=True)
class LocalPrompt(Prompt):
    messages: tuple[tuple[str, str], ...]

    def template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(list(self.messages))


@dataclass(frozen=True)
class HubPrompt(Prompt):
    chat_template: ChatPromptTemplate = field(repr=False, compare=False)

    def template(self) -> ChatPromptTemplate:
        return self.chat_template


def prompts_root() -> Path:
    """Directory holding prompt files; override with LANKATAX_PROMPTS_DIR."""
    override = _setting("LANKATAX_PROMPTS_DIR", "prompts_dir")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[4] / "prompts"


def _setting(env_name: str, field: str) -> str | None:
    """Read a prompt setting from the environment, then from Settings/.env."""
    value = os.environ.get(env_name)
    if value:
        return value

    from lankatax_ai.backend.config import settings

    value = getattr(settings, field, None)
    if isinstance(value, PromptSource):
        return value.value
    return value or None


def get_prompt(
    prompt_id: str,
    *,
    version: str | None = None,
    source: PromptSource | str | None = None,
) -> Prompt:
    """Load a prompt by id, version and source.

    Source precedence: explicit argument, then LANKATAX_PROMPT_SOURCE, then
    "local". Version precedence: explicit argument, then
    LANKATAX_PROMPT_OVERRIDES, then the manifest default (local files) or
    "latest" when a hub prompt has no local manifest.
    """
    resolved_source = _resolve_source(source)
    if resolved_source is PromptSource.hub:
        return _load_hub(prompt_id, version)
    return _load_local(prompt_id, version)


def _resolve_source(source: PromptSource | str | None) -> PromptSource:
    if source is None:
        source = _setting("LANKATAX_PROMPT_SOURCE", "prompt_source") or "local"
    if isinstance(source, PromptSource):
        return source

    try:
        return PromptSource(source.strip().lower())
    except ValueError as exc:
        allowed = [item.value for item in PromptSource]
        raise PromptSourceError(
            f"Unknown prompt source {source!r}; expected one of {allowed}"
        ) from exc


def _load_local(prompt_id: str, version: str | None) -> Prompt:
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


def _load_hub(prompt_id: str, version: str | None) -> Prompt:
    prefix = (_setting("LANKATAX_PROMPT_HUB_PREFIX", "prompt_hub_prefix") or "").strip().rstrip("/")
    if not prefix:
        raise PromptSourceError(
            "LANKATAX_PROMPT_HUB_PREFIX must be set (for example 'owner' or "
            "'owner/project') when the prompt source is 'hub'"
        )

    resolved = (
        version
        or _env_override(prompt_id)
        or _optional_manifest_default(prompt_id, prompts_root() / prompt_id)
        or "latest"
    )
    identifier = f"{prefix}/{prompt_id}"
    if resolved != "latest":
        identifier = f"{identifier}:{resolved}"

    try:
        template = _pull_from_hub(identifier)
    except PromptError:
        raise
    except Exception as exc:
        raise PromptSourceError(
            f"Could not pull prompt {identifier!r} from LangSmith Hub: {exc}"
        ) from exc

    if not isinstance(template, ChatPromptTemplate):
        raise PromptFormatError(
            f"{identifier}: expected a chat prompt, got {type(template).__name__}"
        )

    variables = tuple(template.input_variables)
    invalid = sorted(name for name in variables if not _VARIABLE_RE.match(name))
    if invalid:
        raise PromptFormatError(
            f"{identifier}: placeholders must be lowercase snake_case, got {invalid}"
        )
    if len(set(variables)) != len(variables):
        raise PromptFormatError(f"{identifier}: duplicate input_variables")

    return HubPrompt(
        id=prompt_id,
        version=resolved,
        input_variables=variables,
        sha256=_template_digest(template),
        source=PromptSource.hub,
        path=None,
        chat_template=template,
    )


def _hub_client_instance() -> Any:
    """Build the LangSmith client once, with credentials from Settings.

    Passing the key and endpoint explicitly means Hub pulls work even when
    tracing is disabled or bootstrap_tracing() has not exported them.
    """
    global _hub_client
    if _hub_client is None:
        with _hub_client_lock:
            if _hub_client is None:
                from langsmith import Client

                from lankatax_ai.backend.config import settings

                _hub_client = Client(
                    api_key=settings.langchain_api_key.get_secret_value(),
                    api_url=settings.langchain_endpoint,
                )
    return _hub_client


def _pull_from_hub(identifier: str) -> ChatPromptTemplate:
    """Pull one prompt, reusing one client and LangSmith's prompt cache.

    The client owns a thread-safe LRU cache (5 minute TTL by default) with
    stale-while-revalidate, so repeat loads do not make a network call. A
    single client also reuses its HTTP connection pool across pulls.
    """
    return cast(
        ChatPromptTemplate, _hub_client_instance().pull_prompt(identifier, skip_cache=False)
    )


def _template_digest(template: ChatPromptTemplate) -> str:
    payload = json.dumps(dumpd(template), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _env_override(prompt_id: str) -> str | None:
    raw = _setting("LANKATAX_PROMPT_OVERRIDES", "prompt_overrides") or ""
    for entry in raw.split(","):
        name, separator, value = entry.partition("=")
        if separator and name.strip() == prompt_id:
            return value.strip()
    return None


def _manifest_default(prompt_id: str, root: Path) -> str:
    default = _optional_manifest_default(prompt_id, root)
    if default is None:
        raise PromptFormatError(
            f"Missing manifest for prompt {prompt_id!r}: {root / _MANIFEST_NAME}"
        )
    return default


def _optional_manifest_default(prompt_id: str, root: Path) -> str | None:
    manifest = root / _MANIFEST_NAME
    if not manifest.is_file():
        return None

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
        template = ChatPromptTemplate.from_messages(messages)
        found = list(template.input_variables)
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

    digest = _template_digest(template)

    return LocalPrompt(
        id=prompt_id,
        version=version,
        messages=tuple(messages),
        input_variables=tuple(raw_variables),
        sha256=digest,
        source=PromptSource.local,
        path=path,
    )
