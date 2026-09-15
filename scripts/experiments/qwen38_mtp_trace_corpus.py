"""Build a bounded, private corpus from synthetic or explicitly selected Pi traces.

This module deliberately has no network, subprocess, scanner, or tokenizer dependency.
The ``extract`` command copies only normalized candidate records into a new private
staging directory.  The ``filter-scan`` command consumes a Gitleaks JSON report and
writes grouped train/dev/test JSONL after excluding flagged candidate files.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


MAX_MESSAGES = 64
MAX_CONTEXT_CHARS = 32_000
MIN_SUBSTANTIVE_USER_CHARS = 120
MAX_TOOL_FAMILY_OUTPUT_CHARS = 8_192
MAX_JSONL_LINE_BYTES = 2 * 1024 * 1024
MAX_SESSION_BYTES = 64 * 1024 * 1024
MAX_SESSION_ENTRIES = 100_000
RECENT_WINDOW = _dt.timedelta(hours=24)

_TOOL_ARGUMENTS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "bash": (frozenset({"command"}), frozenset({"timeout"})),
    "read": (frozenset({"path"}), frozenset({"offset", "limit"})),
    "write": (frozenset({"path", "content"}), frozenset()),
    "replace": (
        frozenset({"path", "remove_from", "remove_to", "replacement_lines"}),
        frozenset(),
    ),
}
_ALLOWED_ROLES = frozenset({"user", "assistant", "tool"})
_TEXT_BLOCK_TYPES = frozenset({"text", "input_text", "output_text"})
_THINKING_BLOCK_TYPES = frozenset({"thinking", "reasoning"})
_TOOL_CALL_BLOCK_TYPES = frozenset({"toolCall", "tool_call", "function_call"})
_TOOL_RESULT_ROLES = frozenset({"tool", "toolResult", "tool_result"})
_SENSITIVE_COMMAND_RE = re.compile(
    r"(?:^|[\s/])(?:\.env(?:[.\w-]*)?|\.ssh)(?:$|[\s/])"
    r"|\b(?:credentials?|auth(?:orization)?|passwords?|passwd|tokens?|api[_ -]?keys?|secrets?)\b"
    r"|\bpi-secret\b|\bprintenv\b|\b(?:env|export)\s+(?:-\w+\s+)*\w*"
    r"|\b(?:cat|head|tail|less|more|sed|awk|grep|rg|find|ls|read)\b[^\n]*"
    r"\b(?:sessions?|observability)\b",
    re.IGNORECASE,
)
_SENSITIVE_PATH_RE = re.compile(
    r"(?:^|[/\\])(?:\.env(?:[.\w-]*)?|\.ssh)(?:[/\\]|$)"
    r"|(?:^|[/\\])[^/\\]*(?:sessions?|observability)[^/\\]*(?:[/\\]|$)"
    r"|(?:^|[/\\])[^/\\]*(?:credentials?|auth(?:entication|orization)?|secrets?|tokens?|passwords?|passwd)[^/\\]*(?:[/\\]|$)"
    r"|\bpi-secret\b",
    re.IGNORECASE,
)


class CorpusError(RuntimeError):
    """An expected, fail-closed corpus input or output error."""


class SessionRejected(ValueError):
    """A selected session cannot safely produce candidates."""


@dataclass
class _Session:
    path: Path
    header: dict[str, Any]
    entries: dict[str, dict[str, Any]]
    order: tuple[str, ...]
    session_id: str
    parent_session: str | None
    timestamp: _dt.datetime
    group: str = ""
    lineage_root: str = ""


@dataclass
class _Call:
    original_id: str
    canonical_id: str
    name: str
    arguments: dict[str, Any]
    message_index: int
    sensitive: bool = False



def _abs(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _assert_no_symlink(path: Path, *, allow_missing_final: bool = False) -> None:
    """Reject symlinks in every existing component of a path."""
    path = _abs(path)
    current = Path(path.anchor)
    parts = path.parts[1:] if path.anchor else path.parts
    for index, part in enumerate(parts):
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            if allow_missing_final and index == len(parts) - 1:
                return
            raise CorpusError("path does not exist") from None
        if stat.S_ISLNK(mode):
            raise CorpusError("symlink path is not allowed")


def _existing_dir(path: Path | str, *, private: bool = False) -> Path:
    path = _abs(path)
    _assert_no_symlink(path)
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        raise CorpusError("directory does not exist") from None
    if not stat.S_ISDIR(mode):
        raise CorpusError("path is not a directory")
    if private and stat.S_IMODE(mode) & 0o077:
        raise CorpusError("private directory permissions are required")
    return path


def _existing_file(path: Path | str, *, private: bool = False) -> Path:
    path = _abs(path)
    _assert_no_symlink(path)
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        raise CorpusError("file does not exist") from None
    if not stat.S_ISREG(mode):
        raise CorpusError("regular file is required")
    if private and stat.S_IMODE(mode) & 0o077:
        raise CorpusError("private file permissions are required")
    return path


def _under_git(path: Path) -> bool:
    current = path if path.is_dir() else path.parent
    while True:
        marker = current / ".git"
        try:
            marker_mode = os.lstat(marker).st_mode
        except FileNotFoundError:
            marker_mode = 0
        if marker_mode:
            return True
        if current.parent == current:
            return False
        current = current.parent


def _validate_new_output(path: Path | str, *, forbidden_roots: Sequence[Path]) -> Path:
    """Validate and create a new private directory, never overwriting anything."""
    output = _abs(path)
    parent = _existing_dir(output.parent)
    if output.exists() or output.is_symlink():
        raise CorpusError("output directory must be new")
    for forbidden in forbidden_roots:
        forbidden = _abs(forbidden)
        if _is_within(output, forbidden) or _is_within(forbidden, output):
            raise CorpusError("output overlaps a protected tree")
    if _under_git(parent):
        raise CorpusError("output under a Git repository is not allowed")
    try:
        os.mkdir(output, 0o700)
    except FileExistsError:
        raise CorpusError("output directory must be new") from None
    os.chmod(output, 0o700)
    return output


def _mkdir_private(path: Path) -> Path:
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        raise CorpusError("private directory must be new") from None
    os.chmod(path, 0o700)
    return path


def _write_exclusive(path: Path, text: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        raise CorpusError("output file must be new") from None
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.chmod(path, 0o600)
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise


def _read_text_no_symlink(path: Path, *, max_bytes: int | None = None) -> str:
    _existing_file(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        if max_bytes is not None:
            data = os.read(fd, max_bytes + 1)
            if len(data) > max_bytes:
                raise SessionRejected("input file is too large")
            return data.decode("utf-8")
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            fd = -1
            return handle.read()
    finally:
        if fd != -1:
            os.close(fd)


def _parse_timestamp(value: Any) -> _dt.datetime:
    if isinstance(value, bool):
        raise SessionRejected("invalid timestamp")
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise SessionRejected("invalid timestamp")
        try:
            return _dt.datetime.fromtimestamp(value, tz=_dt.timezone.utc)
        except (OverflowError, OSError, ValueError) as exc:
            raise SessionRejected("invalid timestamp") from exc
    if not isinstance(value, str) or not value.strip():
        raise SessionRejected("invalid timestamp")
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = _dt.datetime.fromisoformat(raw)
    except ValueError as exc:
        raise SessionRejected("invalid timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.timezone.utc)
    return parsed.astimezone(_dt.timezone.utc)


def _json_line_objects(path: Path) -> list[dict[str, Any]]:
    try:
        data = _read_text_no_symlink(path, max_bytes=MAX_SESSION_BYTES)
    except (UnicodeDecodeError, OSError, CorpusError) as exc:
        raise SessionRejected("unreadable session") from exc
    rows: list[dict[str, Any]] = []
    for line in data.splitlines():
        if len(line.encode("utf-8")) > MAX_JSONL_LINE_BYTES:
            raise SessionRejected("session line is too large")
        if not line.strip():
            raise SessionRejected("blank session line")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SessionRejected("malformed JSONL") from exc
        if not isinstance(row, dict):
            raise SessionRejected("session row is not an object")
        rows.append(row)
        if len(rows) > MAX_SESSION_ENTRIES:
            raise SessionRejected("session is too large")
    if not rows:
        raise SessionRejected("empty session")
    return rows


def _load_session(path: Path) -> _Session:
    rows = _json_line_objects(path)
    header = rows[0]
    if header.get("type") != "session":
        raise SessionRejected("missing session header")
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise SessionRejected("invalid session id")
    cwd = header.get("cwd")
    if not isinstance(cwd, str) or not cwd:
        raise SessionRejected("missing session cwd")
    timestamp = _parse_timestamp(header.get("timestamp"))
    parent_session = header.get("parentSession")
    if parent_session is not None and (
        not isinstance(parent_session, str) or not parent_session
    ):
        raise SessionRejected("invalid parent session")

    entries: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows[1:]:
        entry_id = row.get("id")
        if not isinstance(entry_id, str) or not entry_id or entry_id == session_id:
            raise SessionRejected("invalid entry id")
        if entry_id in entries:
            raise SessionRejected("duplicate entry id")
        if "parentId" not in row or (
            row.get("parentId") is not None and not isinstance(row.get("parentId"), str)
        ):
            raise SessionRejected("invalid parent id")
        if not isinstance(row.get("type"), str) or not row.get("type"):
            raise SessionRejected("invalid entry type")
        entries[entry_id] = row
        order.append(entry_id)
    return _Session(
        path=path,
        header=header,
        entries=entries,
        order=tuple(order),
        session_id=session_id,
        parent_session=parent_session,
        timestamp=timestamp,
    )


def _session_recent(session: _Session, now: _dt.datetime) -> bool:
    active = session.header.get("active") is True or session.header.get("isActive") is True
    status = session.header.get("status")
    if isinstance(status, str) and status.lower() in {"active", "running", "open"}:
        active = True
    if active or now - session.timestamp < RECENT_WINDOW:
        return True
    try:
        mtime = _dt.datetime.fromtimestamp(session.path.stat().st_mtime, tz=_dt.timezone.utc)
    except OSError:
        return True
    return now - mtime < RECENT_WINDOW


def _walk_jsonl(root: Path) -> Iterator[Path]:
    """Walk without following symlinks; any symlink is a fail-closed input error."""
    stack = [root]
    while stack:
        current = stack.pop()
        _assert_no_symlink(current)
        try:
            entries = list(os.scandir(current))
        except OSError as exc:
            raise CorpusError("cannot inspect session root") from exc
        for entry in sorted(entries, key=lambda item: item.name, reverse=True):
            try:
                if entry.is_symlink():
                    raise CorpusError("symlink in session tree")
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False) and entry.name.endswith(".jsonl"):
                    yield Path(entry.path)
            except OSError as exc:
                raise CorpusError("cannot inspect session tree") from exc


def _assign_session_groups(sessions: Sequence[_Session]) -> None:
    counts: dict[str, int] = {}
    for session in sessions:
        counts[session.session_id] = counts.get(session.session_id, 0) + 1
    by_id: dict[str, _Session] = {
        session.session_id: session
        for session in sessions
        if counts[session.session_id] == 1
    }
    # Pi stores fork parents as session-file paths, not only session IDs.
    # Resolve exclusively against files already loaded from the allowlist.
    by_path = {str(session.path.absolute()): session.session_id for session in by_id.values()}

    visiting: set[str] = set()
    finished: dict[str, str] = {}
    cyclic: set[str] = set()

    def root_for(session_id: str) -> str:
        if session_id in finished:
            return finished[session_id]
        if session_id in visiting:
            cyclic.add(session_id)
            return ""
        session = by_id[session_id]
        visiting.add(session_id)
        if session.parent_session:
            parent = by_path.get(session.parent_session, session.parent_session)
            root = root_for(parent) if parent in by_id else ""
        else:
            root = session_id
        visiting.remove(session_id)
        if not root:
            cyclic.add(session_id)
        finished[session_id] = root
        return root

    for session in sessions:
        if session.session_id not in by_id:
            continue
        root = root_for(session.session_id)
        if not root or session.session_id in cyclic:
            session.lineage_root = ""
            continue
        session.lineage_root = root
        # Forks on another day/project remain in the root family's split.
        origin = by_id[root]
        day = origin.timestamp.date().isoformat()
        source_key = f"{root}\0{origin.header['cwd']}\0{day}".encode("utf-8")
        session.group = hashlib.sha256(source_key).hexdigest()[:32]


def _extract_text(content: Any, *, allow_tool_calls: bool = False) -> tuple[str, list[dict[str, Any]]]:
    """Return text and embedded calls, dropping only explicit thinking blocks."""
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        raise SessionRejected("unsupported message content")
    pieces: list[str] = []
    calls: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            raise SessionRejected("unsupported content block")
        block_type = block.get("type")
        if block_type in _TEXT_BLOCK_TYPES:
            text = block.get("text")
            if not isinstance(text, str):
                raise SessionRejected("invalid text block")
            pieces.append(text)
        elif block_type in _THINKING_BLOCK_TYPES:
            continue
        elif block_type in _TOOL_CALL_BLOCK_TYPES:
            if not allow_tool_calls:
                raise SessionRejected("unexpected tool call")
            calls.append(block)
        else:
            raise SessionRejected("unsupported content block")
    return "".join(pieces), calls


def _parse_tool_call(raw: Mapping[str, Any]) -> tuple[str, str, dict[str, Any]]:
    if not isinstance(raw, Mapping):
        raise SessionRejected("invalid tool call")
    if "function" in raw and raw.get("type") not in {None, "function"}:
        raise SessionRejected("unsupported tool call type")
    if "function" not in raw and "type" in raw and raw.get("type") not in _TOOL_CALL_BLOCK_TYPES | {"function"}:
        raise SessionRejected("unsupported tool call type")
    if "function" in raw:
        if set(raw) - {"id", "type", "function"}:
            raise SessionRejected("unsupported tool call fields")
    elif set(raw) - {"id", "type", "name", "arguments"}:
        raise SessionRejected("unsupported tool call fields")
    call_id = raw.get("id")
    if not isinstance(call_id, str) or not call_id:
        raise SessionRejected("tool call id is required")
    if "function" in raw:
        function = raw.get("function")
        if not isinstance(function, Mapping):
            raise SessionRejected("invalid function call")
        if set(function) != {"name", "arguments"}:
            raise SessionRejected("unsupported function fields")
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = raw.get("name")
        arguments = raw.get("arguments")
    if not isinstance(name, str) or name not in _TOOL_ARGUMENTS:
        raise SessionRejected("unsupported tool")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise SessionRejected("malformed tool arguments") from exc
    if not isinstance(arguments, dict):
        raise SessionRejected("tool arguments must be an object")
    required, optional = _TOOL_ARGUMENTS[name]
    keys = set(arguments)
    if not required.issubset(keys) or not keys.issubset(required | optional):
        raise SessionRejected("invalid tool argument keys")
    _validate_tool_arguments(name, arguments)
    return call_id, name, {key: arguments[key] for key in sorted(arguments)}


def _validate_tool_arguments(name: str, arguments: Mapping[str, Any]) -> None:
    if name == "bash":
        if not isinstance(arguments["command"], str):
            raise SessionRejected("bash command must be text")
        if "timeout" in arguments:
            timeout = arguments["timeout"]
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise SessionRejected("invalid bash timeout")
            if not math.isfinite(float(timeout)) or timeout <= 0:
                raise SessionRejected("invalid bash timeout")
    elif name == "read":
        if not isinstance(arguments["path"], str) or not arguments["path"]:
            raise SessionRejected("read path must be text")
        for key in ("offset", "limit"):
            if key in arguments:
                value = arguments[key]
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise SessionRejected("invalid read bound")
    elif name == "write":
        if not isinstance(arguments["path"], str) or not arguments["path"]:
            raise SessionRejected("write path must be text")
        if not isinstance(arguments["content"], str):
            raise SessionRejected("write content must be text")
    elif name == "replace":
        if not isinstance(arguments["path"], str) or not arguments["path"]:
            raise SessionRejected("replace path must be text")
        if not isinstance(arguments["remove_from"], str) or not isinstance(
            arguments["remove_to"], str
        ):
            raise SessionRejected("invalid replace range")
        lines = arguments["replacement_lines"]
        if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
            raise SessionRejected("invalid replacement lines")


def _sensitive_arguments(name: str, arguments: Mapping[str, Any]) -> bool:
    if name == "bash":
        return bool(_SENSITIVE_COMMAND_RE.search(str(arguments["command"])))
    if name in {"read", "write", "replace"}:
        values: list[str] = []
        for key, value in arguments.items():
            if isinstance(value, str):
                values.append(value)
            elif isinstance(value, list):
                values.extend(item for item in value if isinstance(item, str))
        return any(_SENSITIVE_PATH_RE.search(value) for value in values)
    return True




def _message(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    message = entry.get("message")
    if not isinstance(message, Mapping):
        raise SessionRejected("message row is missing message object")
    role = message.get("role")
    if not isinstance(role, str):
        raise SessionRejected("message role is missing")
    return message


def _raw_calls(message: Mapping[str, Any], embedded: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    calls: list[Mapping[str, Any]] = list(embedded)
    if "tool_calls" in message:
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            raise SessionRejected("tool_calls must be a list")
        calls.extend(tool_calls)
    return calls


def _tool_stdout(message: Mapping[str, Any]) -> str:
    content_text = ""
    if "content" in message:
        content_text, embedded = _extract_text(message.get("content"), allow_tool_calls=False)
        if embedded:
            raise SessionRejected("tool result contains a call")
    if "stdout" in message:
        stdout = message.get("stdout")
        if not isinstance(stdout, str):
            raise SessionRejected("stdout must be text")
        return stdout
    return content_text


def _thinking_value(entry: Mapping[str, Any]) -> bool | None:
    for obj in (entry, entry.get("message")):
        if not isinstance(obj, Mapping):
            continue
        if isinstance(obj.get("thinking_level_change"), bool):
            return bool(obj["thinking_level_change"])
        if obj.get("type") == "thinking_level_change":
            level = obj.get("thinkingLevel")
            if isinstance(level, bool):
                return level
            if isinstance(level, str) and level.strip():
                return level.strip().lower() not in {"off", "none", "disabled"}
    return None


def _is_assistant_entry(entry: Mapping[str, Any]) -> bool:
    if entry.get("type") != "message":
        return False
    message = entry.get("message")
    return isinstance(message, Mapping) and message.get("role") == "assistant"


def _chain(session: _Session, target_id: str) -> list[dict[str, Any]] | None:
    current: str | None = target_id
    seen: set[str] = set()
    reverse: list[dict[str, Any]] = []
    while current is not None and current != session.session_id:
        if current in seen:
            return None
        seen.add(current)
        entry = session.entries.get(current)
        if entry is None or "parentId" not in entry:
            return None
        reverse.append(entry)
        parent = entry["parentId"]
        if parent is None:
            current = None
        elif isinstance(parent, str):
            current = parent
        else:
            return None
    if current not in {None, session.session_id}:
        return None
    reverse.reverse()
    return reverse


def _normalize_candidate_context(
    context_entries: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    messages: list[dict[str, Any]] = []
    calls_by_original: dict[str, _Call] = {}
    canonical_ids: dict[str, str] = {}
    pending: dict[str, _Call] = {}
    used_results: set[str] = set()
    family_chars: dict[str, int] = {}
    tool_names: set[str] = set()

    def canonical_id(original: str) -> str:
        if original not in canonical_ids:
            canonical_ids[original] = f"tool_{len(canonical_ids) + 1:04d}"
        return canonical_ids[original]

    for entry in context_entries:
        if entry.get("type") != "message":
            continue
        message = _message(entry)
        role = message.get("role")
        if role == "user":
            if pending:
                raise SessionRejected("incomplete tool group")
            text, embedded = _extract_text(message.get("content"), allow_tool_calls=False)
            if embedded:
                raise SessionRejected("user tool call")
            messages.append({"role": "user", "content": text})
            continue
        if role == "assistant":
            if pending:
                raise SessionRejected("incomplete tool group")
            text, embedded = _extract_text(message.get("content"), allow_tool_calls=True)
            raw_calls = _raw_calls(message, embedded)
            normalized_calls: list[dict[str, Any]] = []
            for raw_call in raw_calls:
                original, name, arguments = _parse_tool_call(raw_call)
                if original in calls_by_original:
                    raise SessionRejected("duplicate tool call id")
                call = _Call(
                    original_id=original,
                    canonical_id=canonical_id(original),
                    name=name,
                    arguments=arguments,
                    message_index=len(messages),
                    sensitive=_sensitive_arguments(name, arguments),
                )
                calls_by_original[original] = call
                pending[original] = call
                tool_names.add(name)
                normalized_calls.append(
                    {
                        "id": call.canonical_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(
                                arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                            ),
                        },
                    }
                )
            if text or normalized_calls:
                assistant_message: dict[str, Any] = {"role": "assistant", "content": text}
                if normalized_calls:
                    assistant_message["tool_calls"] = normalized_calls
                messages.append(assistant_message)
            continue
        if role in _TOOL_RESULT_ROLES:
            if "tool_call_id" in message:
                original = message.get("tool_call_id")
            elif "toolCallId" in message:
                original = message.get("toolCallId")
            else:
                original = message.get("call_id")
            if not isinstance(original, str) or not original or original not in pending:
                raise SessionRejected("orphan tool result")
            call = pending.pop(original)
            result_name = message.get("name", message.get("toolName", message.get("tool_name")))
            if result_name is not None and result_name != call.name:
                raise SessionRejected("tool result name mismatch")
            stdout = _tool_stdout(message)
            family_chars[call.name] = family_chars.get(call.name, 0) + len(stdout)
            if family_chars[call.name] > MAX_TOOL_FAMILY_OUTPUT_CHARS:
                raise SessionRejected("tool output family cap exceeded")
            if call.sensitive:
                raise SessionRejected("sensitive tool context")
            if original in used_results:
                raise SessionRejected("duplicate tool result")
            used_results.add(original)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.canonical_id,
                    "name": call.name,
                    "content": stdout,
                }
            )
            continue
        raise SessionRejected("unsupported message role")

    if pending or not messages or messages[-1]["role"] not in {"user", "tool"}:
        raise SessionRejected("prompt does not end at a safe boundary")
    if len(messages) > MAX_MESSAGES:
        raise SessionRejected("message bound exceeded")
    chars = len(json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    if chars > MAX_CONTEXT_CHARS:
        raise SessionRejected("context bound exceeded")
    return messages, sorted(tool_names)


def _candidate_id(messages: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _candidate_for_target(session: _Session, target: Mapping[str, Any], chain: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    try:
        # The target response is not part of this prompt. Its future tool choice
        # must not filter an otherwise valid context or leak into the dataset.
        user_index = -1
        for index in range(len(chain) - 1, -1, -1):
            entry = chain[index]
            if entry.get("type") != "message":
                continue
            message = _message(entry)
            if message.get("role") != "user":
                continue
            text, embedded = _extract_text(message.get("content"), allow_tool_calls=False)
            if embedded:
                raise SessionRejected("user tool call")
            if len(text) >= MIN_SUBSTANTIVE_USER_CHARS:
                user_index = index
                break
        if user_index < 0:
            return None
        messages, tools = _normalize_candidate_context(chain[user_index:-1])
        thinking = _thinking_value(target)
        if thinking is None:
            for entry in reversed(chain[: len(chain) - 1]):
                thinking = _thinking_value(entry)
                if thinking is not None:
                    break
        return {
            "id": _candidate_id(messages),
            "source_group": session.group,
            "messages": messages,
            "tools": tools,
            "thinking": True if thinking is None else thinking,
        }
    except SessionRejected:
        return None


def _session_candidates(session: _Session) -> Iterator[dict[str, Any]]:
    if not session.lineage_root or not session.group:
        return
    for entry_id in session.order:
        entry = session.entries[entry_id]
        if not _is_assistant_entry(entry):
            continue
        chain = _chain(session, entry_id)
        if not chain or chain[-1].get("id") != entry_id:
            continue
        candidate = _candidate_for_target(session, entry, chain)
        if candidate is not None:
            yield candidate


def _safe_candidate_json(candidate: Mapping[str, Any]) -> str:
    return json.dumps(candidate, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def extract(
    session_root: Path | str,
    allow_roots: Sequence[Path | str],
    output: Path | str,
    *,
    now: _dt.datetime | None = None,
) -> dict[str, int]:
    """Extract normalized candidates into a new private staging directory."""
    root = _existing_dir(session_root)
    if not allow_roots:
        raise CorpusError("at least one allow root is required")
    allows = [_existing_dir(item) for item in allow_roots]
    scopes = [root] if any(_is_within(root, allow) for allow in allows) else allows
    if any(not _is_within(scope, root) for scope in scopes):
        raise CorpusError("allow root is outside the session root")

    now = now or _dt.datetime.now(tz=_dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=_dt.timezone.utc)
    now = now.astimezone(_dt.timezone.utc)
    sessions: list[_Session] = []
    skipped_recent = 0
    skipped_invalid = 0
    for path in sorted({path for scope in scopes for path in _walk_jsonl(scope)}):
        try:
            session = _load_session(path)
        except SessionRejected:
            skipped_invalid += 1
            continue
        if _session_recent(session, now):
            skipped_recent += 1
            continue
        sessions.append(session)
    _assign_session_groups(sessions)
    output_dir = _validate_new_output(output, forbidden_roots=(root,))
    candidate_dir = _mkdir_private(output_dir / "candidates")

    written: set[str] = set()
    candidates = 0
    for session in sessions:
        for candidate in _session_candidates(session):
            candidate_id = candidate["id"]
            if candidate_id in written:
                continue
            path = candidate_dir / f"{candidate_id}.json"
            _write_exclusive(path, _safe_candidate_json(candidate))
            written.add(candidate_id)
            candidates += 1
    return {
        "candidates": candidates,
        "sessions": len(sessions),
        "skipped_recent": skipped_recent,
        "skipped_invalid": skipped_invalid,
    }


def _candidate_messages_valid(candidate: Mapping[str, Any]) -> None:
    required = {"id", "source_group", "messages", "tools", "thinking"}
    if set(candidate) != required:
        raise CorpusError("candidate schema mismatch")
    if not isinstance(candidate["id"], str) or not candidate["id"]:
        raise CorpusError("candidate id is invalid")
    if not isinstance(candidate["source_group"], str) or not candidate["source_group"]:
        raise CorpusError("candidate group is invalid")
    if not isinstance(candidate["thinking"], bool):
        raise CorpusError("candidate thinking flag is invalid")
    messages = candidate["messages"]
    if not isinstance(messages, list) or not messages:
        raise CorpusError("candidate messages are invalid")
    if len(messages) > MAX_MESSAGES:
        raise CorpusError("candidate message bound exceeded")
    if not isinstance(messages[-1], dict) or messages[-1].get("role") not in {"user", "tool"}:
        raise CorpusError("candidate boundary is invalid")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in _ALLOWED_ROLES:
            raise CorpusError("candidate message is invalid")
        if not isinstance(message.get("content"), str):
            raise CorpusError("candidate content is invalid")
        if message["role"] == "assistant" and "tool_calls" in message:
            calls = message["tool_calls"]
            if not isinstance(calls, list):
                raise CorpusError("candidate tool calls are invalid")
            for call in calls:
                if not isinstance(call, dict) or set(call) != {"id", "type", "function"}:
                    raise CorpusError("candidate tool call is invalid")
                if not isinstance(call["id"], str) or call["type"] != "function":
                    raise CorpusError("candidate tool call is invalid")
                function = call["function"]
                if not isinstance(function, dict) or set(function) != {"name", "arguments"}:
                    raise CorpusError("candidate function call is invalid")
                name = function["name"]
                if not isinstance(name, str) or name not in _TOOL_ARGUMENTS:
                    raise CorpusError("candidate tool is invalid")
                args = function["arguments"]
                if not isinstance(args, str):
                    raise CorpusError("candidate arguments are invalid")
                try:
                    parsed = json.loads(args)
                except json.JSONDecodeError as exc:
                    raise CorpusError("candidate arguments are invalid") from exc
                if not isinstance(parsed, dict):
                    raise CorpusError("candidate arguments are invalid")
                required, optional = _TOOL_ARGUMENTS[name]
                if set(parsed) != set(parsed) & (required | optional) or not required.issubset(parsed):
                    raise CorpusError("candidate arguments are invalid")
                _validate_tool_arguments(name, parsed)
        if message["role"] == "tool":
            if not isinstance(message.get("tool_call_id"), str) or not isinstance(message.get("name"), str):
                raise CorpusError("candidate tool result is invalid")
    tools = candidate["tools"]
    if not isinstance(tools, list) or not all(isinstance(name, str) and name in _TOOL_ARGUMENTS for name in tools):
        raise CorpusError("candidate tools are invalid")
    if tools != sorted(set(tools)):
        raise CorpusError("candidate tools are not canonical")
    actual_tools = sorted(
        {
            call["function"]["name"]
            for message in messages
            if message["role"] == "assistant"
            for call in message.get("tool_calls", [])
        }
    )
    if tools != actual_tools:
        raise CorpusError("candidate tools do not match messages")
    chars = len(json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    if chars > MAX_CONTEXT_CHARS:
        raise CorpusError("candidate context bound exceeded")
    family_chars: dict[str, int] = {}
    pending: dict[str, str] = {}
    for message in messages:
        if message["role"] == "assistant":
            if pending:
                raise CorpusError("candidate tool group is incomplete")
            for call in message.get("tool_calls", []):
                call_id = call["id"]
                if call_id in pending:
                    raise CorpusError("candidate tool id is duplicated")
                pending[call_id] = call["function"]["name"]
        elif message["role"] == "tool":
            call_id = message["tool_call_id"]
            if call_id not in pending or pending.pop(call_id) != message["name"]:
                raise CorpusError("candidate tool result is orphaned")
            family = message["name"]
            family_chars[family] = family_chars.get(family, 0) + len(message["content"])
            if family_chars[family] > MAX_TOOL_FAMILY_OUTPUT_CHARS:
                raise CorpusError("candidate tool output cap exceeded")
        elif pending:
            raise CorpusError("candidate tool group is incomplete")
    if pending:
        raise CorpusError("candidate tool group is incomplete")


def _load_candidates(extract_root: Path) -> tuple[list[tuple[Path, dict[str, Any]]], dict[Path, Path]]:
    candidate_dir = _existing_dir(extract_root / "candidates", private=True)
    rows: list[tuple[Path, dict[str, Any]]] = []
    by_resolved: dict[Path, Path] = {}
    try:
        entries = sorted(os.scandir(candidate_dir), key=lambda entry: entry.name)
    except OSError as exc:
        raise CorpusError("cannot inspect candidates") from exc
    for entry in entries:
        path = Path(entry.path)
        if entry.is_symlink():
            raise CorpusError("candidate symlink is not allowed")
        if not entry.is_file(follow_symlinks=False):
            raise CorpusError("unexpected candidate entry")
        if not entry.name.endswith(".json"):
            raise CorpusError("unexpected candidate file")
        try:
            raw = _read_text_no_symlink(path, max_bytes=MAX_JSONL_LINE_BYTES)
            candidate = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CorpusError("malformed candidate") from exc
        if not isinstance(candidate, dict):
            raise CorpusError("malformed candidate")
        try:
            _candidate_messages_valid(candidate)
        except (CorpusError, SessionRejected, KeyError, IndexError, TypeError, ValueError) as exc:
            raise CorpusError("malformed candidate") from exc
        resolved = _abs(path)
        by_resolved[resolved] = path
        rows.append((path, candidate))
    if not rows:
        raise CorpusError("no candidates")
    return rows, by_resolved


def _resolve_report_candidate(
    file_value: str,
    *,
    extract_root: Path,
    report_path: Path,
    known: Mapping[Path, Path],
) -> Path:
    if not file_value or "\x00" in file_value:
        raise CorpusError("invalid scanner report path")
    raw = Path(file_value)
    if raw.is_absolute():
        options = [raw]
    else:
        options = [extract_root / raw, extract_root / "candidates" / raw, report_path.parent / raw, extract_root.parent / raw, Path.cwd() / raw]
    matches: list[Path] = []
    for option in options:
        option = _abs(option)
        if not os.path.lexists(option):
            continue
        try:
            _assert_no_symlink(option)
        except CorpusError:
            raise CorpusError("scanner report path is unsafe") from None
        if option in known:
            matches.append(option)
    if len(set(matches)) != 1:
        raise CorpusError("scanner report path is unknown")
    return known[matches[0]]


def _load_flagged(
    extract_root: Path,
    report: Path | str,
    known: Mapping[Path, Path],
) -> set[Path]:
    report_path = _existing_file(report)
    if _is_within(report_path, extract_root / "candidates"):
        raise CorpusError("scanner report overlaps candidates")
    try:
        raw = _read_text_no_symlink(report_path, max_bytes=16 * 1024 * 1024)
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorpusError("malformed scanner report") from exc
    if not isinstance(payload, list):
        raise CorpusError("scanner report must be a JSON array")
    flagged: set[Path] = set()
    for finding in payload:
        if not isinstance(finding, dict) or not isinstance(finding.get("File"), str):
            raise CorpusError("scanner report finding is malformed")
        flagged.add(_resolve_report_candidate(finding["File"], extract_root=extract_root, report_path=report_path, known=known))
    return flagged


def _context_key(candidate: Mapping[str, Any]) -> str:
    payload = json.dumps(candidate["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split_for_group(group: str) -> str:
    value = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 100
    if value < 80:
        return "train"
    if value < 90:
        return "dev"
    return "test"


def filter_scan(
    extract_root: Path | str,
    report: Path | str,
    output: Path | str,
) -> dict[str, int]:
    """Apply a pre-generated Gitleaks JSON report and write grouped JSONL splits."""
    extract_dir = _existing_dir(extract_root, private=True)
    rows, known = _load_candidates(extract_dir)
    flagged = _load_flagged(extract_dir, report, known)
    output_dir = _validate_new_output(output, forbidden_roots=(extract_dir,))

    # Deduplicate exact contexts before assigning any split.  A duplicate found in
    # different source groups is conservatively unified to the lexicographically
    # smallest opaque group, so it cannot cross a split boundary.  If one copy is
    # flagged, discard every exact copy rather than relying on scanner coverage.
    context_keys = {path: _context_key(candidate) for path, candidate in rows}
    flagged_contexts = {context_keys[path] for path in flagged}
    by_context: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path, candidate in rows:
        key = context_keys[path]
        if _abs(path) in flagged or key in flagged_contexts:
            continue
        previous = by_context.get(key)
        if previous is None or (candidate["source_group"], candidate["id"]) < (
            previous[1]["source_group"],
            previous[1]["id"],
        ):
            by_context[key] = (path, candidate)
    unified: list[dict[str, Any]] = []
    for _, candidate in sorted(by_context.values(), key=lambda item: (item[1]["source_group"], item[1]["id"])):
        unified.append(candidate)

    split_rows: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    for candidate in unified:
        split_rows[_split_for_group(candidate["source_group"])].append(_safe_candidate_json(candidate))
    for split, lines in split_rows.items():
        _write_exclusive(output_dir / f"{split}.jsonl", "".join(lines))

    # Delete only paths proven to be candidate files in the owned private output,
    # and only after all report validation and dataset writes succeeded.
    for candidate_path in flagged:
        candidate_path = _abs(candidate_path)
        if candidate_path not in known or not _is_within(candidate_path, extract_dir / "candidates"):
            raise CorpusError("refusing unsafe candidate deletion")
        _assert_no_symlink(candidate_path)
        try:
            os.unlink(candidate_path)
        except OSError as exc:
            raise CorpusError("cannot remove rejected candidate") from exc
    report_path = _abs(report)
    if _is_within(report_path, extract_dir) and not _is_within(report_path, extract_dir / "candidates"):
        _assert_no_symlink(report_path)
        try:
            os.unlink(report_path)
        except OSError as exc:
            raise CorpusError("cannot remove scanner report") from exc
    return {
        "kept": len(unified),
        "rejected": len(flagged),
        "train": len(split_rows["train"]),
        "dev": len(split_rows["dev"]),
        "test": len(split_rows["test"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract_parser = subparsers.add_parser("extract", help="extract normalized candidates")
    extract_parser.add_argument("--session-root", "--root", required=True, type=Path)
    extract_parser.add_argument("--allow-root", action="append", required=True, type=Path)
    extract_parser.add_argument("--output", "--out", required=True, type=Path)
    extract_parser.add_argument("--now", type=str, help="UTC ISO timestamp for deterministic tests")

    filter_parser = subparsers.add_parser("filter-scan", help="apply a Gitleaks JSON report")
    filter_parser.add_argument("--input", "--extract-root", required=True, type=Path)
    filter_parser.add_argument("--report", required=True, type=Path)
    filter_parser.add_argument("--output", "--out", required=True, type=Path)
    return parser


def _main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "extract":
            now = _parse_timestamp(args.now) if args.now else None
            result = extract(args.session_root, args.allow_root, args.output, now=now)
        else:
            result = filter_scan(args.input, args.report, args.output)
    except (CorpusError, SessionRejected, OSError, ValueError, TypeError):
        # Never echo source paths, report contents, or exception details.
        print("error: trace corpus operation rejected", file=sys.stderr)
        return 2
    print(" ".join(f"{key}={value}" for key, value in sorted(result.items())))
    return 0


def main() -> None:
    raise SystemExit(_main())


if __name__ == "__main__":
    main()
