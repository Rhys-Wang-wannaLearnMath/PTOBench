"""Small durable DAG scheduler used by build/resume."""
from __future__ import annotations
import hashlib, json, sqlite3, time, os, math, re, tempfile, uuid, threading
from pathlib import Path
from .contracts import ContractError

try:
    import fcntl
except ImportError:  # pragma: no cover - the supported runner is POSIX
    fcntl = None


def validate_json(value, schema: dict, *, path: str = "$", refs: set[str] | None = None):
    """Validate the schema subset used by builder contracts, including private refs."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError(f"{path}: nonfinite number")
    if not isinstance(schema, dict):
        raise ContractError("invalid schema")
    typ = schema.get("type")
    checks = {"object": dict, "array": list, "string": str, "number": (int, float),
              "integer": int, "boolean": bool, "null": type(None)}
    types = typ if isinstance(typ, list) else [typ] if typ else []
    if types and not any(t in checks and isinstance(value, checks[t]) and
                         not (t in {"number", "integer"} and isinstance(value, bool)) for t in types):
        raise ContractError(f"{path}: expected {typ}")
    if "enum" in schema and value not in schema["enum"]:
        raise ContractError(f"{path}: invalid enum value")
    if isinstance(value, str):
        value.encode("utf-8", errors="strict")
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", float("inf")):
            raise ContractError(f"{path}: invalid string length")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise ContractError(f"{path}: invalid string pattern")
        if schema.get("x-ref") and refs is not None and value not in refs:
            raise ContractError(f"{path}: unknown reference {value}")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            raise ContractError(f"{path}: unknown field")
        for key in schema.get("required", []):
            if key not in value:
                raise ContractError(f"{path}: missing {key}")
        for key, item in value.items():
            validate_json(item, properties.get(key, {}), path=f"{path}.{key}", refs=refs)
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", float("inf")):
            raise ContractError(f"{path}: invalid array length")
        if schema.get("uniqueItems") and len({json.dumps(x, sort_keys=True, allow_nan=False) for x in value}) != len(value):
            raise ContractError(f"{path}: duplicate items")
        identity = schema.get("x-unique-by")
        if identity:
            identities = [x.get(identity) for x in value if isinstance(x, dict)]
            if len(identities) != len(set(identities)):
                raise ContractError(f"{path}: duplicate {identity}")
        for index, item in enumerate(value):
            validate_json(item, schema.get("items", {}), path=f"{path}[{index}]", refs=refs)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.parent.is_symlink():
        raise ContractError("job record path cannot be a symlink")
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _synchronized(method):
    """Serialize operations on a shared SQLite connection."""
    def wrapped(self, *args, **kwargs):
        with self._db_lock:
            return method(self, *args, **kwargs)
    wrapped.__name__ = method.__name__
    wrapped.__doc__ = method.__doc__
    return wrapped


class Scheduler:
    def __init__(self, work_dir: Path, *, read_only=False):
        self.work_dir=Path(work_dir)
        # Documentary Pi workers share one scheduler instance. SQLite serializes
        # writes, while this lock also protects the connection object itself.
        self._db_lock = threading.RLock()
        if read_only:
            self.db = sqlite3.connect((self.work_dir / 'state.sqlite').resolve().as_uri() + '?mode=ro', uri=True,
                                      check_same_thread=False)
            self.db.row_factory = sqlite3.Row
            self._lock_handle = None
            return
        self.work_dir.mkdir(parents=True,exist_ok=True)
        self._lock_handle = (self.work_dir / '.workspace.lock').open('a+', encoding='utf-8')
        if fcntl is not None:
            try:
                fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                self._lock_handle.close()
                raise RuntimeError('workspace is already locked') from exc
        self.db=sqlite3.connect(self.work_dir/'state.sqlite', check_same_thread=False); self.db.row_factory=sqlite3.Row
        self.db.executescript('''CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,stage TEXT,domain TEXT,input_key TEXT,status TEXT,attempts INTEGER,result_path TEXT,error TEXT); CREATE TABLE IF NOT EXISTS dependencies(parent TEXT,child TEXT,PRIMARY KEY(parent,child)); CREATE TABLE IF NOT EXISTS approvals(candidate_id TEXT,review_job TEXT,decision TEXT,PRIMARY KEY(candidate_id,review_job)); CREATE TABLE IF NOT EXISTS releases(id TEXT PRIMARY KEY,phase TEXT,path TEXT);'''); self.db.commit()

    def close(self):
        with self._db_lock:
            if getattr(self, 'db', None) is not None:
                self.db.close(); self.db = None
            if getattr(self, '_lock_handle', None) is not None:
                if fcntl is not None:
                    try: fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
                    except OSError: pass
                self._lock_handle.close(); self._lock_handle = None

    def __del__(self):
        try: self.close()
        except Exception: pass
    @_synchronized
    def add(self, job_id, stage, domain, input_value, depends=()):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", job_id):
            raise ContractError("invalid job id")
        depends = tuple(depends)
        if job_id in depends or any(self.db.execute("SELECT id FROM jobs WHERE id=?", (p,)).fetchone() is None for p in depends):
            raise ContractError("missing or invalid dependency")
        if len(depends) != len(set(depends)):
            raise ContractError('duplicate dependency')
        descendants = {r[0] for r in self.db.execute("""WITH RECURSIVE reachable(id) AS (
            SELECT child FROM dependencies WHERE parent=? UNION
            SELECT d.child FROM dependencies d JOIN reachable r ON d.parent=r.id
        ) SELECT id FROM reachable""", (job_id,))}
        if set(depends) & descendants:
            raise ContractError('dependency cycle')
        old_deps = {r[0] for r in self.db.execute("SELECT parent FROM dependencies WHERE child=?", (job_id,))}
        key=hashlib.sha256(json.dumps(input_value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        existing = self.db.execute("SELECT input_key,status FROM jobs WHERE id=?", (job_id,)).fetchone()
        if existing is None:
            self.db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?)",(job_id,stage,domain,key,"pending",0,None,None))
        elif existing['input_key'] != key or old_deps != set(depends):
            self.invalidate(job_id, 'input or dependencies changed')
            # A successful result is reusable only for the exact normalized
            # input envelope.  Invalidate stale state before the caller reads
            # it; the old result remains private evidence but is never reused.
            self.db.execute(
                "UPDATE jobs SET stage=?,domain=?,input_key=?,status='pending',attempts=0,result_path=NULL,error=? WHERE id=?",
                (stage, domain, key, 'cache invalidated: input identity changed', job_id),
            )
        self.db.execute("DELETE FROM dependencies WHERE child=?", (job_id,))
        for parent in depends:self.db.execute("INSERT OR IGNORE INTO dependencies VALUES(?,?)",(parent,job_id))
        self.db.commit()

    @_synchronized
    def write_envelope(self, job_id: str, envelope: dict, *, prompt: str, schema: dict) -> None:
        """Persist the private, auditable job inputs before any Agent call."""
        root = self.work_dir / 'jobs' / job_id
        row = self.get(job_id)
        revision = root / 'revisions' / row['input_key']
        revision.mkdir(parents=True, exist_ok=True)
        atomic_json(revision / 'input.json', envelope)
        atomic_json(revision / 'schema.json', schema)
        (revision / 'prompt.txt').write_text(prompt, encoding='utf-8')
        root.mkdir(parents=True, exist_ok=True)
        for path, value in (
            (root / 'input.json', envelope),
            (root / 'schema.json', schema),
        ):
            if path.is_symlink():
                raise RuntimeError('job record path cannot be a symlink')
            path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')
        prompt_path = root / 'prompt.txt'
        if prompt_path.is_symlink():
            raise RuntimeError('job record path cannot be a symlink')
        prompt_path.write_text(prompt, encoding='utf-8')

    @_synchronized
    def invalidate(self, job_id: str, reason: str = 'cached result failed validation') -> None:
        self.db.execute("""WITH RECURSIVE affected(id) AS (
            SELECT ? UNION SELECT d.child FROM dependencies d JOIN affected a ON d.parent=a.id
        ) UPDATE jobs SET status='pending',attempts=0,result_path=NULL,error=?
          WHERE id IN (SELECT id FROM affected)""", (job_id, reason))
        self.db.execute("""WITH RECURSIVE affected(id) AS (
            SELECT ? UNION SELECT d.child FROM dependencies d JOIN affected a ON d.parent=a.id
        ) DELETE FROM approvals WHERE review_job IN (SELECT id FROM affected)""", (job_id,))
        self.db.commit()
    @_synchronized
    def ready(self):
        rows=self.db.execute("SELECT * FROM jobs WHERE status='pending'").fetchall(); out=[]
        for row in rows:
            deps=self.db.execute("SELECT j.status FROM dependencies d LEFT JOIN jobs j ON j.id=d.parent WHERE d.child=?",(row['id'],)).fetchall()
            if any(d['status'] in ('failed','skipped') for d in deps): self.db.execute("UPDATE jobs SET status='skipped',error=? WHERE id=?",('dependency failed',row['id'])); continue
            if all(d['status']=='succeeded' for d in deps): out.append(dict(row))
        self.db.commit(); return out
    @_synchronized
    def start(self,j): self.db.execute("UPDATE jobs SET status='running',attempts=attempts+1 WHERE id=?",(j,)); self.db.commit()
    @_synchronized
    def finish(self,j,status,result=None,error=None):
        path=None
        if result is not None:
            row = self.get(j)
            path=self.work_dir/'jobs'/j/'results'/row['input_key']/(uuid.uuid4().hex + '.json'); atomic_json(path, result)
            atomic_json(self.work_dir/'jobs'/j/'result.json', result)
        self.db.execute("UPDATE jobs SET status=?,result_path=?,error=? WHERE id=?",(status,str(path) if path else None,error,j)); self.db.commit()
    @_synchronized
    def recover(self): self.db.execute("UPDATE jobs SET status='pending' WHERE status='running'"); self.db.commit()
    @_synchronized
    def summary(self): return [dict(r) for r in self.db.execute('SELECT * FROM jobs ORDER BY id')]
    @_synchronized
    def get(self, job_id):
        row=self.db.execute('SELECT * FROM jobs WHERE id=?',(job_id,)).fetchone()
        if not row:return None
        value=dict(row)
        if value.get('result_path') and Path(value['result_path']).is_file():
            try:value['result']=json.loads(Path(value['result_path']).read_text())
            except Exception: pass
        return value
