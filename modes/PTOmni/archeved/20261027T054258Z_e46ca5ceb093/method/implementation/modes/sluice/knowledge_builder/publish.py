"""Pure-package validation and locked publication with crash recovery."""
from __future__ import annotations
import json
import re
import shutil
from pathlib import Path

from .contracts import PublishedCard, PublishedReference, PublishedPack
from .scheduler import atomic_json
from .bindings import reviewed_bindings


class PublishError(ValueError):
    pass


try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


SEMANTIC_ID = re.compile(r'[a-z][a-z0-9_]{2,80}')


def _registry_signals():
    # Keep question and signal definitions in the profiling registry. The
    # publication index stores only semantic links into that registry.
    from ..engine.profile_questions import QUESTIONS
    return {question: set(spec.expected_signal_keys) for question, spec in QUESTIONS.items()}


def _binding_rows(root, cards):
    path = Path(root) / 'index/mechanism_bindings.json'
    if not path.is_file():
        raise PublishError('mechanism binding index is missing')
    raw = json.loads(path.read_text(encoding='utf-8'))
    if type(raw) is not dict or set(raw) != {'schema_version', 'bindings'} or raw.get('schema_version') != 1 or type(raw.get('bindings')) is not list:
        raise PublishError('invalid mechanism binding index')
    registry = _registry_signals()
    bindings = {}
    for row in raw['bindings']:
        if type(row) is not dict or set(row) != {'mechanism_id', 'card_ref', 'evidence'}:
            raise PublishError('invalid mechanism binding')
        mechanism_id = row['mechanism_id']
        card_ref = row['card_ref']
        if (not isinstance(mechanism_id, str) or not SEMANTIC_ID.fullmatch(mechanism_id)
                or mechanism_id in bindings or card_ref != 'card:' + mechanism_id
                or mechanism_id not in cards or type(row['evidence']) is not list):
            raise PublishError('mechanism binding points to an unknown or duplicate card')
        evidence = []
        seen = set()
        for item in row['evidence']:
            if type(item) is not dict or set(item) != {'question_id', 'signal_keys'}:
                raise PublishError('invalid mechanism evidence binding')
            question = item['question_id']
            keys = item['signal_keys']
            if (not isinstance(question, str) or question not in registry
                    or type(keys) is not list or not keys
                    or any(not isinstance(key, str) or key not in registry[question] for key in keys)
                    or len(keys) != len(set(keys)) or (question, tuple(keys)) in seen):
                raise PublishError('mechanism binding references an unknown question or signal')
            seen.add((question, tuple(keys)))
            evidence.append({'question_id': question, 'signal_keys': list(keys)})
        if not evidence:
            raise PublishError('mechanism binding has no evidence mapping')
        bindings[mechanism_id] = {'mechanism_id': mechanism_id, 'card_ref': card_ref,
                                  'evidence': evidence}
    return bindings


def index_value(cards, packs=None, refs=None, bindings=None):
    packs = packs or {}
    refs = refs or {}
    bindings = bindings or {}
    items = [
        {'id': 'card:' + key, 'title': c.title, 'summary': c.summary, 'domain': c.domain,
         'aliases': list(dict.fromkeys(list(c.questions) +
             [keyword for _, pack in sorted(packs.items()) if key in pack.card_ids for keyword in pack.keywords] +
             [evidence['question_id'] for evidence in bindings.get(key, {}).get('evidence', [])])),
         'source_features': list(c.source_features),
         'preconditions': list(c.applicability),
         'expected_signals': list(dict.fromkeys(signal for evidence in bindings.get(key, {}).get('evidence', [])
                                                for signal in evidence['signal_keys'])),
         'risks': list(c.risks),
         'text': ' '.join((c.principle,) + c.applicability + c.directions + c.invariants + c.effects + c.risks)}
        for key, c in sorted(cards.items())]
    # References normally arrive with a card's requires closure. Independent
    # operation boundaries also need a searchable entry of their own.
    dependencies = {dep for obj in list(cards.values()) + list(refs.values()) for dep in obj.requires}
    items.extend({'id': 'ref:' + key, 'title': ref.title, 'summary': ref.title,
                  'domain': 'reference', 'aliases': [], 'source_features': [],
                  'preconditions': list(ref.applicability), 'expected_signals': [], 'risks': [],
                  'text': ref.body}
                 for key, ref in sorted(refs.items()) if 'ref:' + key not in dependencies)
    return {'schema_version': 1, 'items': items}


def validate_package(root, *, require_index=False):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise PublishError('knowledge package is missing or linked')
    directories = {'cards', 'reference', 'packs', 'index'}
    if {p.name for p in root.iterdir()} != directories:
        raise PublishError('knowledge package must contain exactly cards/reference/packs/index')
    for directory in root.iterdir():
        if directory.is_symlink() or not directory.is_dir():
            raise PublishError('knowledge directories must be real directories')
        for file in directory.iterdir():
            if file.is_symlink() or not file.is_file() or file.suffix != '.json':
                raise PublishError('unexpected knowledge entry')
            if directory.name == 'index':
                if file.name not in {'search.json', 'mechanism_bindings.json'}:
                    raise PublishError('unexpected knowledge index')
            elif not SEMANTIC_ID.fullmatch(file.stem):
                raise PublishError('invalid knowledge identity')
    def read(directory, cls):
        objects = {}
        for file in sorted((root / directory).iterdir()):
            value = cls.from_dict(json.loads(file.read_text(encoding='utf-8')))
            if value.id != file.stem:
                raise PublishError('knowledge filename mismatch')
            objects[value.id] = value
        return objects
    cards, refs, packs = read('cards', PublishedCard), read('reference', PublishedReference), read('packs', PublishedPack)
    if not cards or not refs or not packs:
        raise PublishError('empty knowledge package')
    if any(len(json.dumps(c.to_dict(), ensure_ascii=False).encode()) > 10240 for c in cards.values()):
        raise PublishError('card too large')
    valid = {'card:' + key for key in cards} | {'ref:' + key for key in refs}
    graph = {'card:' + key: list(c.requires) for key, c in cards.items()} | {'ref:' + key: list(r.requires) for key, r in refs.items()}
    for dependencies in graph.values():
        if len(set(dependencies)) != len(dependencies) or any(dep not in valid for dep in dependencies):
            raise PublishError('invalid knowledge dependency')
    for card in cards.values():
        if any(dep not in valid for dep in card.related):
            raise PublishError('invalid related reference')
    def visit(node, ancestors):
        if node in ancestors:
            raise PublishError('requires cycle')
        for dep in graph[node]:
            visit(dep, ancestors | {node})
    for node in graph:
        visit(node, set())
    for pack in packs.values():
        if len(set(pack.card_ids)) != len(pack.card_ids) or any(key not in cards for key in pack.card_ids):
            raise PublishError('invalid pack card membership')
    binding_path = root / 'index/mechanism_bindings.json'
    if not binding_path.is_file():
        raise PublishError('mechanism binding index is missing')
    bindings = _binding_rows(root, cards)
    index = root / 'index/search.json'
    if require_index and not index.is_file():
        raise PublishError('knowledge index is missing')
    if index.is_file() and json.loads(index.read_text(encoding='utf-8')) != index_value(cards, packs, refs, bindings):
        raise PublishError('knowledge index differs from published content')
    return cards, refs, packs


def build_index(root, cards):
    root = Path(root)
    packs = {p.stem: PublishedPack.from_dict(json.loads(p.read_text())) for p in (Path(root) / 'packs').glob('*.json')}
    refs = {p.stem: PublishedReference.from_dict(json.loads(p.read_text())) for p in (Path(root) / 'reference').glob('*.json')}
    bindings_path = root / 'index/mechanism_bindings.json'
    if not bindings_path.exists():
        atomic_json(bindings_path, reviewed_bindings(cards))
    bindings = _binding_rows(root, cards)
    atomic_json(root / 'index/search.json', index_value(cards, packs, refs, bindings))



def package_fingerprint(root):
    root = Path(root)
    files = {str(path.relative_to(root)): path.read_text(encoding='utf-8') for path in sorted(root.rglob('*.json'))}
    import hashlib
    return hashlib.sha256(json.dumps(files, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()

def _locations(out):
    return (out.with_name('.' + out.name + '.incoming'), out.with_name('.' + out.name + '.backup'),
            out.with_name('.' + out.name + '.install.json'))


def _binding_locations(out):
    return (out.with_name('.' + out.name + '.release.incoming'),
            out.with_name('.' + out.name + '.release.backup'))


def _recover(out):
    """Called with the publication lock; incomplete transactions roll back."""
    incoming, backup, journal = _locations(out)
    binding_path = out.with_name(out.name + '_release.json')
    binding_incoming, binding_backup = _binding_locations(out)
    if any(p.is_symlink() for p in (out, incoming, backup, journal, binding_path,
                                    binding_incoming, binding_backup)):
        raise PublishError('publication paths cannot be links')
    if not journal.exists():
        if backup.exists():
            raise PublishError('untracked publication backup needs operator inspection')
        if incoming.exists():
            shutil.rmtree(incoming)
        return
    record = json.loads(journal.read_text())
    if record.get('phase') == 'committed':
        validate_package(out, require_index=True)
        if record.get('bound_release'):
            binding = json.loads(binding_path.read_text(encoding='utf-8'))
            if binding != record.get('binding') or binding.get('knowledge_digest') != package_fingerprint(out):
                raise PublishError('committed knowledge release binding is inconsistent')
        if backup.exists():
            shutil.rmtree(backup)
        if binding_backup.exists():
            binding_backup.unlink()
    elif backup.exists():
        if out.exists():
            shutil.rmtree(out)
        backup.replace(out)
    elif not record.get('had_previous') and out.exists():
        shutil.rmtree(out)
    if record.get('bound_release'):
        if binding_backup.exists():
            binding_path.unlink(missing_ok=True)
            binding_backup.replace(binding_path)
        elif not record.get('had_binding'):
            binding_path.unlink(missing_ok=True)
    if incoming.exists():
        shutil.rmtree(incoming)
    if binding_incoming.exists():
        binding_incoming.unlink()
    journal.unlink()


def recover_install(out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / '.knowledge.install.lock').open('a+') as lock:
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        _recover(out)


def install(staging, out, *, binding=None):
    staging, out = Path(staging), Path(out)
    validate_package(staging, require_index=True)
    from .review import static_review
    if static_review(staging)['decision'] != 'accept':
        raise PublishError('staging package purity review rejected publication')
    out.parent.mkdir(parents=True, exist_ok=True)
    with (out.parent / '.knowledge.install.lock').open('a+') as lock:
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        _recover(out)
        incoming, backup, journal = _locations(out)
        binding_path = out.with_name(out.name + '_release.json')
        binding_incoming, binding_backup = _binding_locations(out)
        if binding is not None:
            if (type(binding) is not dict or set(binding) != {'environment','python_executable','knowledge_digest'}
                    or not all(isinstance(binding.get(key), str) and binding[key] for key in binding)):
                raise PublishError('invalid release binding')
            if binding['knowledge_digest'] != package_fingerprint(staging):
                raise PublishError('release binding does not match staged content')
        record = {'phase': 'preparing', 'had_previous': out.exists(),
                  'bound_release': binding is not None, 'had_binding': binding_path.exists(),
                  'binding': binding}
        atomic_json(journal, record)
        try:
            # Incoming is on the output filesystem; the final renames cannot cross devices.
            shutil.copytree(staging, incoming)
            validate_package(incoming, require_index=True)
            if binding is not None:
                atomic_json(binding_incoming, binding)
            record['phase'] = 'prepared'
            atomic_json(journal, record)
            if out.exists():
                out.replace(backup)
            if binding is not None and binding_path.exists():
                binding_path.replace(binding_backup)
            record['phase'] = 'backed_up'
            atomic_json(journal, record)
            incoming.replace(out)
            validate_package(out, require_index=True)
            if binding is not None:
                binding_incoming.replace(binding_path)
                if json.loads(binding_path.read_text(encoding='utf-8')) != binding:
                    raise PublishError('release binding changed during publication')
            record['phase'] = 'committed'
            atomic_json(journal, record)
            if backup.exists():
                shutil.rmtree(backup)
            if binding_backup.exists():
                binding_backup.unlink()
            journal.unlink()
        except BaseException:
            _recover(out)
            raise
