import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ValidationError
from rich import print as rprint

from lankatax_ai.backend.types import DocumentKind

_HERE = Path(__file__).parent
_DEFAULT_REGISTRY = _HERE / "data" / "urls.json"


class URLCategory(Enum):
    tax_info = "tax_info"
    guide = "guide"
    circular = "circular"
    act = "act"
    gazette = "gazette"
    form = "form"
    schedule = "schedule"
    faq = "faq"
    notice = "notice"


class ScrapableURL(BaseModel):
    url: str
    category: URLCategory
    label: str
    kind: DocumentKind


class URLRegistry:
    def __init__(self, filepath: Path = _DEFAULT_REGISTRY):
        self.registry_map = {}
        self.registry_list = []
        self._load(filepath)

    def _load(self, filepath: Path) -> None:
        with open(filepath) as f:
            for entry in json.load(f):
                try:
                    url = ScrapableURL(**entry)
                    self.register(url)
                except ValidationError as e:
                    rprint(f"[bold yellow]⚠[/] Skipping bad entry: {e}")
                    continue

    def register(self, entry: ScrapableURL) -> None:
        self.registry_map.setdefault(entry.category, []).append(entry)
        self.registry_list.append(entry)

    def get_all(self) -> list[ScrapableURL]:
        return self.registry_list

    def get_by_category(self, categ: URLCategory) -> list[ScrapableURL]:
        return self.registry_map.get(categ, [])
