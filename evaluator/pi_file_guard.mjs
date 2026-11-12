import fs from 'node:fs';
import path from 'node:path';

const fileTools = new Set(['read', 'write', 'edit', 'grep', 'find', 'ls']);
const pathKeys = new Set(['path', 'paths', 'directory', 'directories']);

function within(candidate, root) {
  const relative = path.relative(root, candidate);
  return relative === '' || (!relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative));
}

function canonical(candidate) {
  let current = path.resolve(candidate);
  const suffix = [];
  while (!fs.existsSync(current)) {
    const parent = path.dirname(current);
    if (parent === current) return path.resolve(candidate);
    suffix.unshift(path.basename(current));
    current = parent;
  }
  try {
    return path.resolve(fs.realpathSync(current), ...suffix);
  } catch {
    return path.resolve(candidate);
  }
}

function collectPaths(value, key = '', result = []) {
  if (pathKeys.has(key)) {
    if (typeof value === 'string') result.push(value);
    else if (Array.isArray(value)) result.push(...value.filter((item) => typeof item === 'string'));
  }
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    for (const [childKey, child] of Object.entries(value)) collectPaths(child, childKey, result);
  }
  return result;
}

function inodeKey(stat) {
  return `${stat.dev}:${stat.ino}`;
}

function protectedInodes(roots) {
  const found = new Set();
  const visit = (entry) => {
    let stat;
    try {
      stat = fs.lstatSync(entry);
    } catch {
      return;
    }
    if (stat.isSymbolicLink()) return;
    if (stat.isFile()) found.add(inodeKey(stat));
    if (!stat.isDirectory()) return;
    let children;
    try {
      children = fs.readdirSync(entry);
    } catch {
      return;
    }
    for (const child of children) visit(path.join(entry, child));
  };
  for (const root of roots) visit(root);
  return found;
}

function pathContainsPrivateAlias(target, privateRoots, privateFiles) {
  const inspect = (entry) => {
    let stat;
    try {
      stat = fs.lstatSync(entry);
    } catch (error) {
      if (error.code === 'ENOENT' || error.code === 'ENOTDIR') return false;
      return true;
    }

    if (stat.isSymbolicLink()) {
      const destination = canonical(entry);
      return privateRoots.some((root) => within(destination, root) || within(root, destination));
    }
    if (stat.isFile()) return privateFiles.has(inodeKey(stat));
    if (!stat.isDirectory()) return false;

    let children;
    try {
      children = fs.readdirSync(entry);
    } catch {
      return true;
    }
    return children.some((child) => inspect(path.join(entry, child)));
  };
  return inspect(target);
}

function blocked(candidate, cwd, privateRoots, privateFiles) {
  const resolved = path.resolve(cwd, candidate);
  const real = canonical(resolved);
  if (privateRoots.some((root) => within(resolved, root) || within(root, resolved)
      || within(real, root) || within(root, real))) return true;
  return pathContainsPrivateAlias(resolved, privateRoots, privateFiles);
}

export default function (pi) {
  pi.on('tool_call', (event, context) => {
    if (!fileTools.has(event.toolName)) return undefined;

    const privateRoots = [process.env.PI_CODING_AGENT_DIR, process.env.PTOBENCH_PI_TOOLS_DIR,
      process.env.PTOBENCH_AGENT_SESSION_DIR]
      .filter((value) => typeof value === 'string' && value.length > 0)
      .map((value) => canonical(value));
    if (privateRoots.length === 0) {
      return { block: true, reason: 'Private Pi runtime boundary is unavailable' };
    }

    const privateFiles = protectedInodes(privateRoots);
    const cwd = context.cwd || process.cwd();
    for (const candidate of collectPaths(event.input)) {
      if (blocked(candidate, cwd, privateRoots, privateFiles)) {
        return { block: true, reason: 'Access to private Pi runtime files is not available to Agent file tools' };
      }
    }
    return undefined;
  });
}
