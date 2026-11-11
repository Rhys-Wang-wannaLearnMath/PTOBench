"""Independent publication review checks and Pi review contract."""
from __future__ import annotations
import json, re
from pathlib import Path
from .contracts import PublishedCard, PublishedReference, PublishedPack, ContractError
from .publish import validate_package
from .scheduler import validate_json

SOURCE_PATH = re.compile(r"(?:/(?:home|tmp|work|workspace|root|opt|usr|var|mnt|private|Users)/|[A-Za-z]:\\|pto_wiki/|(?:^|[\s\"'])modes/(?:cannbot|sluice)/|raw/patches/)", re.I)
HEX_IDENTITY = re.compile(r"(?<![A-Za-z0-9_])(?=[a-f0-9]{7,64}(?![A-Za-z0-9_]))[a-f0-9]*[a-f][a-f0-9]*(?![A-Za-z0-9_])", re.I)
# Timing estimates in documentation are still numerical performance data.
# Keep physical capacities, alignment and API dimension relations legal.
TIMING_VALUE = re.compile(r"\d+(?:\.\d+)?\s*(?:[nmu]s|[µμ]s|(?:nano|micro|milli)?seconds?|纳秒|微秒|毫秒|秒)(?![A-Za-z0-9_])", re.I)

def static_review(root):
    cards, refs, packs = validate_package(root, require_index=True)
    issues = []
    for path in sorted(Path(root).rglob('*.json')):
        text = path.name + "\n" + path.read_text(encoding='utf-8')
        if SOURCE_PATH.search(text) or HEX_IDENTITY.search(text) or TIMING_VALUE.search(text):
            issues.append({'object': str(path.relative_to(root)), 'code': 'purity_leak'})
    for card in cards.values():
        if not card.applicability or not card.principle:
            issues.append({'object': 'cards/' + card.id + '.json', 'code': 'missing_condition_or_principle'})
    return {'decision': 'accept' if not issues else 'revise', 'issues': issues}


def review_schema():
    return {'type':'object','additionalProperties':False,'required':['items','requests','omissions'],'properties':{'items':{'type':'array'},'requests':{'type':'array'},'omissions':{'type':'array'}}}
