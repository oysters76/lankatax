from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from lankatax_ai.backend.rag import multi_query_retriever, reranker
from lankatax_ai.backend.rag import prompts as prompts_module
from lankatax_ai.backend.rag.multi_query_retriever import MultiQueryExpander
from lankatax_ai.backend.rag.prompts import (
    Prompt,
    PromptError,
    PromptFormatError,
    PromptNotFound,
    PromptSourceError,
    PromptVariableError,
    get_prompt,
)
from lankatax_ai.backend.rag.reranker import Reranker
from lankatax_ai.backend.types import PromptSource

pytestmark = pytest.mark.unit

REPO_PROMPTS = Path(__file__).resolve().parents[1] / "prompts"

DEFAULT_MESSAGES = (("human", "Question: {question}"),)


def _write_prompt(
    root: Path,
    prompt_id: str = "demo",
    version: str = "v1",
    *,
    variables: tuple[str, ...] = ("question",),
    messages: tuple[tuple[str, str], ...] = DEFAULT_MESSAGES,
) -> Path:
    directory = root / prompt_id
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": prompt_id,
        "version": version,
        "input_variables": list(variables),
        "messages": [{"role": role, "content": content} for role, content in messages],
    }
    (directory / f"{version}.yaml").write_text(json.dumps(payload), encoding="utf-8")
    return directory


def _write_manifest(root: Path, prompt_id: str = "demo", default: str = "v1") -> None:
    (root / prompt_id / "manifest.yaml").write_text(f"default: {default}\n", encoding="utf-8")


@pytest.fixture
def prompts_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from lankatax_ai.backend.config import settings

    monkeypatch.setenv("LANKATAX_PROMPTS_DIR", str(tmp_path))
    monkeypatch.delenv("LANKATAX_PROMPT_OVERRIDES", raising=False)
    monkeypatch.delenv("LANKATAX_PROMPT_SOURCE", raising=False)
    monkeypatch.delenv("LANKATAX_PROMPT_HUB_PREFIX", raising=False)
    monkeypatch.setattr(settings, "prompts_dir", None)
    monkeypatch.setattr(settings, "prompt_overrides", None)
    monkeypatch.setattr(settings, "prompt_source", PromptSource.local)
    monkeypatch.setattr(settings, "prompt_hub_prefix", None)
    return tmp_path


def _hub_template(content: str = "Question: {question}") -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("human", content)])


def _fake_hub(
    monkeypatch: pytest.MonkeyPatch, template, *, error: Exception | None = None
) -> list[str]:
    calls: list[str] = []

    def fake_pull(identifier: str):
        calls.append(identifier)
        if error is not None:
            raise error
        return template

    monkeypatch.setattr(prompts_module, "_pull_from_hub", fake_pull)
    return calls


def test_manifest_default_version_is_loaded(prompts_dir: Path):
    _write_prompt(prompts_dir)
    _write_manifest(prompts_dir)

    prompt = get_prompt("demo")

    assert isinstance(prompt, Prompt)
    assert prompt.id == "demo"
    assert prompt.version == "v1"
    assert prompt.input_variables == ("question",)
    assert prompt.source is PromptSource.local


def test_explicit_version_wins_over_manifest(prompts_dir: Path):
    _write_prompt(prompts_dir, version="v1")
    _write_prompt(prompts_dir, version="v2")
    _write_manifest(prompts_dir, default="v1")

    assert get_prompt("demo", version="v2").version == "v2"


def test_env_override_wins_over_manifest(prompts_dir: Path, monkeypatch: pytest.MonkeyPatch):
    _write_prompt(prompts_dir, version="v1")
    _write_prompt(prompts_dir, version="v2")
    _write_manifest(prompts_dir, default="v1")
    monkeypatch.setenv("LANKATAX_PROMPT_OVERRIDES", "demo=v2")

    assert get_prompt("demo").version == "v2"


def test_explicit_version_wins_over_env_override(
    prompts_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_prompt(prompts_dir, version="v1")
    _write_prompt(prompts_dir, version="v2")
    _write_manifest(prompts_dir, default="v1")
    monkeypatch.setenv("LANKATAX_PROMPT_OVERRIDES", "demo=v2")

    assert get_prompt("demo", version="v1").version == "v1"


def test_env_override_only_applies_to_the_named_prompt(
    prompts_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_prompt(prompts_dir, prompt_id="demo")
    _write_prompt(prompts_dir, prompt_id="other")
    _write_manifest(prompts_dir, prompt_id="demo")
    _write_manifest(prompts_dir, prompt_id="other")
    monkeypatch.setenv("LANKATAX_PROMPT_OVERRIDES", "demo=v2")

    assert get_prompt("other").version == "v1"


def test_unknown_prompt_id_raises_not_found(prompts_dir: Path):
    with pytest.raises(PromptNotFound, match="not found"):
        get_prompt("missing")


def test_unknown_version_lists_available_versions(prompts_dir: Path):
    _write_prompt(prompts_dir, version="v1")
    _write_manifest(prompts_dir)

    with pytest.raises(PromptNotFound, match=r"available: \['v1'\]"):
        get_prompt("demo", version="v9")


def test_missing_manifest_raises_format_error(prompts_dir: Path):
    _write_prompt(prompts_dir)

    with pytest.raises(PromptFormatError, match="Missing manifest"):
        get_prompt("demo")


def test_malformed_yaml_raises_format_error(prompts_dir: Path):
    directory = prompts_dir / "demo"
    directory.mkdir()
    (directory / "v1.yaml").write_text("id: [unclosed\n", encoding="utf-8")
    _write_manifest(prompts_dir)

    with pytest.raises(PromptFormatError, match="Could not read"):
        get_prompt("demo")


def test_duplicate_yaml_keys_are_rejected(prompts_dir: Path):
    directory = prompts_dir / "demo"
    directory.mkdir()
    (directory / "v1.yaml").write_text(
        "id: demo\nversion: v1\nid: demo\n"
        "input_variables: [question]\n"
        "messages:\n  - role: human\n    content: 'Hi {question}'\n",
        encoding="utf-8",
    )
    _write_manifest(prompts_dir)

    with pytest.raises(PromptFormatError, match="duplicate key"):
        get_prompt("demo")


def test_missing_required_key_raises_format_error(prompts_dir: Path):
    directory = prompts_dir / "demo"
    directory.mkdir()
    (directory / "v1.yaml").write_text(
        json.dumps({"id": "demo", "version": "v1"}), encoding="utf-8"
    )
    _write_manifest(prompts_dir)

    with pytest.raises(PromptFormatError, match="missing keys"):
        get_prompt("demo")


def test_id_must_match_directory(prompts_dir: Path):
    directory = prompts_dir / "demo"
    directory.mkdir()
    (directory / "v1.yaml").write_text(
        json.dumps(
            {
                "id": "other",
                "version": "v1",
                "input_variables": ["question"],
                "messages": [{"role": "human", "content": "{question}"}],
            }
        ),
        encoding="utf-8",
    )
    _write_manifest(prompts_dir)

    with pytest.raises(PromptFormatError, match="id 'other'"):
        get_prompt("demo")


def test_version_must_match_filename(prompts_dir: Path):
    _write_prompt(prompts_dir, version="v1")
    (prompts_dir / "demo" / "v2.yaml").write_text(
        json.dumps(
            {
                "id": "demo",
                "version": "v1",
                "input_variables": ["question"],
                "messages": [{"role": "human", "content": "{question}"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PromptFormatError, match="version 'v1'"):
        get_prompt("demo", version="v2")


def test_undeclared_placeholder_raises_variable_error(prompts_dir: Path):
    _write_prompt(
        prompts_dir,
        variables=("question",),
        messages=(("human", "{question} and {context}"),),
    )
    _write_manifest(prompts_dir)

    with pytest.raises(PromptFormatError, match="undeclared: \\['context'\\]"):
        get_prompt("demo")


def test_declared_but_unused_variable_raises_error(prompts_dir: Path):
    _write_prompt(
        prompts_dir,
        variables=("question", "context"),
        messages=(("human", "{question}"),),
    )
    _write_manifest(prompts_dir)

    with pytest.raises(PromptFormatError, match="unused: \\['context'\\]"):
        get_prompt("demo")


def test_attribute_style_placeholders_are_rejected(prompts_dir: Path):
    _write_prompt(
        prompts_dir,
        variables=("question.__class__",),
        messages=(("human", "{question.__class__}"),),
    )
    _write_manifest(prompts_dir)

    with pytest.raises(PromptFormatError, match="invalid message template"):
        get_prompt("demo")


def test_non_snake_case_placeholders_are_rejected(prompts_dir: Path):
    _write_prompt(
        prompts_dir,
        variables=("Question",),
        messages=(("human", "{Question}"),),
    )
    _write_manifest(prompts_dir)

    with pytest.raises(PromptFormatError, match="snake_case"):
        get_prompt("demo")


def test_unknown_message_role_is_rejected(prompts_dir: Path):
    _write_prompt(prompts_dir, messages=(("tool", "Hi {question}"),))
    _write_manifest(prompts_dir)

    with pytest.raises(PromptFormatError, match="role 'tool'"):
        get_prompt("demo")


def test_format_messages_keeps_roles_and_renders(prompts_dir: Path):
    _write_prompt(
        prompts_dir,
        variables=("question", "context"),
        messages=(
            ("system", "You cite sources."),
            ("human", "Question: {question}\nContext: {context}"),
        ),
    )
    _write_manifest(prompts_dir)

    messages = get_prompt("demo").format_messages(question="How is APIT calculated?", context="18%")

    assert [message.type for message in messages] == ["system", "human"]
    assert messages[1].content.endswith("Context: 18%")


def test_same_variable_renders_in_every_message(prompts_dir: Path):
    _write_prompt(
        prompts_dir,
        messages=(("system", "Query: {question}"), ("human", "Again: {question}")),
    )
    _write_manifest(prompts_dir)

    messages = get_prompt("demo").format_messages(question="APIT")

    assert [message.content for message in messages] == ["Query: APIT", "Again: APIT"]


def test_missing_render_value_raises(prompts_dir: Path):
    _write_prompt(prompts_dir)
    _write_manifest(prompts_dir)

    with pytest.raises(PromptVariableError, match="missing \\['question'\\]"):
        get_prompt("demo").format_messages()


def test_extra_render_value_raises(prompts_dir: Path):
    _write_prompt(prompts_dir)
    _write_manifest(prompts_dir)

    with pytest.raises(PromptVariableError, match="unexpected \\['context'\\]"):
        get_prompt("demo").format_messages(question="APIT", context="18%")


def test_non_string_render_value_raises(prompts_dir: Path):
    _write_prompt(prompts_dir)
    _write_manifest(prompts_dir)

    with pytest.raises(PromptVariableError, match="must be strings"):
        get_prompt("demo").format_messages(question=3)


def test_literal_braces_survive_rendering(prompts_dir: Path):
    _write_prompt(prompts_dir, messages=(("human", "Answer as {{answer}} to {question}"),))
    _write_manifest(prompts_dir)

    content = get_prompt("demo").format_messages(question="APIT")[0].content

    assert content == "Answer as {answer} to APIT"


def test_rendering_is_pure(prompts_dir: Path):
    _write_prompt(prompts_dir)
    _write_manifest(prompts_dir)
    prompt = get_prompt("demo")

    first = prompt.format_messages(question="APIT")
    second = prompt.format_messages(question="APIT")

    assert [message.content for message in first] == [message.content for message in second]


def test_hash_is_stable_and_metadata_is_exposed(prompts_dir: Path):
    _write_prompt(prompts_dir)
    _write_manifest(prompts_dir)

    first = get_prompt("demo")
    second = get_prompt("demo")

    assert first.sha256 == second.sha256
    assert first.metadata == {
        "prompt_id": "demo",
        "prompt_version": "v1",
        "prompt_hash": first.sha256,
        "prompt_source": "local",
    }


def test_hash_changes_when_a_role_changes(prompts_dir: Path):
    _write_prompt(prompts_dir, version="v1", messages=(("human", "Hi {question}"),))
    _write_prompt(prompts_dir, version="v2", messages=(("system", "Hi {question}"),))
    _write_manifest(prompts_dir)

    assert get_prompt("demo", version="v1").sha256 != get_prompt("demo", version="v2").sha256


def test_all_errors_share_a_base_class(prompts_dir: Path):
    _write_prompt(prompts_dir)

    with pytest.raises(PromptError):
        get_prompt("demo")


def test_existing_rag_modules_no_longer_hold_prompt_text():
    assert not hasattr(reranker, "RERANK_PROMPT")
    assert not hasattr(multi_query_retriever, "EXPAND_PROMPT")
    assert not hasattr(Reranker, "_build_prompt")
    assert not hasattr(MultiQueryExpander, "_build_prompt")


def test_every_repo_prompt_loads():
    prompt_dirs = sorted(path for path in REPO_PROMPTS.iterdir() if path.is_dir())
    assert prompt_dirs, f"no prompts found under {REPO_PROMPTS}"

    for directory in prompt_dirs:
        manifest = directory / "manifest.yaml"
        assert manifest.is_file(), f"{directory.name} has no manifest"

        versions = sorted(
            path.stem for path in directory.glob("v*.yaml") if path.name != "manifest.yaml"
        )
        assert versions, f"{directory.name} has no version files"

        default = get_prompt(directory.name, source="local")
        assert default.id == directory.name
        assert default.input_variables

        for version in versions:
            prompt = get_prompt(directory.name, version=version, source="local")
            assert prompt.version == version
            assert prompt.sha256


def test_environment_can_select_the_hub(prompts_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LANKATAX_PROMPT_SOURCE", "hub")
    monkeypatch.setenv("LANKATAX_PROMPT_HUB_PREFIX", "owner")
    calls = _fake_hub(monkeypatch, _hub_template())

    prompt = get_prompt("demo")

    assert calls == ["owner/demo"]
    assert prompt.source is PromptSource.hub


def test_explicit_source_beats_the_environment(prompts_dir: Path, monkeypatch: pytest.MonkeyPatch):
    _write_prompt(prompts_dir)
    _write_manifest(prompts_dir)
    monkeypatch.setenv("LANKATAX_PROMPT_SOURCE", "hub")
    monkeypatch.setattr(
        prompts_module,
        "_pull_from_hub",
        lambda identifier: pytest.fail(f"hub called with {identifier}"),
    )

    assert get_prompt("demo", source="local").source is PromptSource.local


def test_hub_uses_manifest_default_as_tag(prompts_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LANKATAX_PROMPT_HUB_PREFIX", "owner/project")
    (prompts_dir / "demo").mkdir()
    _write_manifest(prompts_dir)
    calls = _fake_hub(monkeypatch, _hub_template())

    prompt = get_prompt("demo", source="hub")

    assert calls == ["owner/project/demo:v1"]
    assert prompt.version == "v1"
    assert prompt.source is PromptSource.hub
    assert prompt.path is None
    assert prompt.format_messages(question="APIT")[0].content == "Question: APIT"


def test_hub_without_manifest_uses_latest(prompts_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LANKATAX_PROMPT_HUB_PREFIX", "owner")
    calls = _fake_hub(monkeypatch, _hub_template())

    prompt = get_prompt("demo", source="hub")

    assert calls == ["owner/demo"]
    assert prompt.version == "latest"


def test_hub_explicit_version_is_a_tag(prompts_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LANKATAX_PROMPT_HUB_PREFIX", "owner")
    calls = _fake_hub(monkeypatch, _hub_template())

    assert get_prompt("demo", version="prod", source="hub").version == "prod"
    assert calls == ["owner/demo:prod"]


def test_hub_prompt_override_is_a_tag(prompts_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LANKATAX_PROMPT_HUB_PREFIX", "owner")
    monkeypatch.setenv("LANKATAX_PROMPT_OVERRIDES", "demo=v2")
    calls = _fake_hub(monkeypatch, _hub_template())

    assert get_prompt("demo", source="hub").version == "v2"
    assert calls == ["owner/demo:v2"]


def test_unknown_source_raises(prompts_dir: Path):
    with pytest.raises(PromptSourceError, match="Unknown prompt source"):
        get_prompt("demo", source="s3")


def test_hub_requires_a_prefix(prompts_dir: Path):
    with pytest.raises(PromptSourceError, match="LANKATAX_PROMPT_HUB_PREFIX"):
        get_prompt("demo", source="hub")


def test_hub_pull_failure_is_wrapped(prompts_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LANKATAX_PROMPT_HUB_PREFIX", "owner")
    _fake_hub(monkeypatch, None, error=RuntimeError("network down"))

    with pytest.raises(PromptSourceError, match="network down"):
        get_prompt("demo", source="hub")


def test_hub_rejects_non_chat_prompts(prompts_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LANKATAX_PROMPT_HUB_PREFIX", "owner")
    _fake_hub(monkeypatch, PromptTemplate.from_template("{question}"))

    with pytest.raises(PromptFormatError, match="expected a chat prompt"):
        get_prompt("demo", source="hub")


def test_hub_rejects_invalid_placeholders(prompts_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LANKATAX_PROMPT_HUB_PREFIX", "owner")
    _fake_hub(monkeypatch, _hub_template("{Question}"))

    with pytest.raises(PromptFormatError, match="snake_case"):
        get_prompt("demo", source="hub")


def test_hub_keeps_strict_render_checks(prompts_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LANKATAX_PROMPT_HUB_PREFIX", "owner")
    _fake_hub(monkeypatch, _hub_template())
    prompt = get_prompt("demo", source="hub")

    with pytest.raises(PromptVariableError, match="missing"):
        prompt.format_messages()

    with pytest.raises(PromptVariableError, match="unexpected"):
        prompt.format_messages(question="APIT", context="18%")


def test_hub_metadata_and_hash_are_exposed(prompts_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LANKATAX_PROMPT_HUB_PREFIX", "owner")
    _fake_hub(monkeypatch, _hub_template())

    first = get_prompt("demo", source="hub")
    second = get_prompt("demo", source="hub")

    assert first.sha256 == second.sha256
    assert first.metadata == {
        "prompt_id": "demo",
        "prompt_version": "latest",
        "prompt_hash": first.sha256,
        "prompt_source": "hub",
    }


def test_identical_local_and_hub_prompts_hash_the_same(
    prompts_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_prompt(prompts_dir)
    _write_manifest(prompts_dir)
    local = get_prompt("demo", source="local")

    monkeypatch.setenv("LANKATAX_PROMPT_HUB_PREFIX", "owner")
    _fake_hub(monkeypatch, _hub_template())
    hub = get_prompt("demo", source="hub")

    assert local.sha256 == hub.sha256


def test_settings_are_used_when_environment_is_unset(
    prompts_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    from lankatax_ai.backend.config import settings

    monkeypatch.setattr(settings, "prompt_source", PromptSource.hub)
    monkeypatch.setattr(settings, "prompt_hub_prefix", "owner")
    calls = _fake_hub(monkeypatch, _hub_template())

    prompt = get_prompt("demo")

    assert calls == ["owner/demo"]
    assert prompt.source is PromptSource.hub


def test_hub_pull_reuses_one_client_and_uses_the_sdk_cache(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.calls = []

        def pull_prompt(self, identifier, *, skip_cache=True):
            self.calls.append((identifier, skip_cache))
            return _hub_template()

    client = FakeClient()
    monkeypatch.setattr(prompts_module, "_hub_client", client)

    prompts_module._pull_from_hub("owner/demo")
    prompts_module._pull_from_hub("owner/demo:latest")

    assert client.calls == [
        ("owner/demo", False),
        ("owner/demo:latest", False),
    ]
