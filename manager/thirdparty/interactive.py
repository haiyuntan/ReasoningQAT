import dataclasses
import json
import logging
import subprocess
import selectors
import signal
import time
from io import TextIOWrapper
from pathlib import Path

import conf.config as config
from manager.struct.structs import from_json, Goal, ProofGoal, ProofVariable
import os
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger(os.path.basename(__file__))
IS_BUILD = False


class InteractiveResponseTimeout(TimeoutError):
    """A bounded Lean response failure; consumes one search attempt."""
    def __init__(self, seconds, operation):
        self.seconds = seconds
        self.operation = operation
        super().__init__(f'Lean Interactive {operation} response exceeded {seconds}s; session terminated')


class InteractiveSessionError(Exception):
    """A failed Lean session, distinct from an ordinary invalid tactic."""
    def __init__(self, message, operation, returncode=None):
        self.operation = operation
        self.returncode = returncode
        super().__init__(message)

def get_project_toolchain(root: Path) -> str:
    with (root / "lean-toolchain").open() as fp:
        return fp.read()


def build_with_toolchain(root: Path, toolchain: str) -> Path:
    logger.info("Building interactive for %s", toolchain)
    if IS_BUILD:
        with (root / "lean-toolchain").open("w") as fp:
            fp.write(toolchain)
        # logger.info("Building interactive for %s", toolchain)
        if subprocess.call(
                ["lake", "build"],
                stdout=subprocess.DEVNULL,
                cwd=root) != 0:
            raise RuntimeError(f"Failed to build: {root} with {toolchain}")
    return root / ".lake" / "build" / "bin"


def build_interactive(toolchain: str) -> Path:
    bin_path = build_with_toolchain(config.interactive_path, toolchain)
    return bin_path / "interactive"


class Interactive:
    def __init__(self, root: Path, path: Path):
        toolchain = get_project_toolchain(root)
        interactive_bin = build_interactive(toolchain)
        args = ["lake", "env", str(interactive_bin), "-i", str(path)]
        self.proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # Inherit the campaign log: an unread stderr PIPE can fill and
            # deadlock Lean while Python waits for a stdout response.
            stderr=None,
            cwd=root,
            start_new_session=True,
        )
        self.read_from = self.proc.stdout
        self.write_to = TextIOWrapper(self.proc.stdin, line_buffering=True)
        self.id = 0
        self.tactic_mode = False
        self.response_timeout = 1200.0
        self._response_buffer = bytearray()
        self.operation = 'startup'
        os.set_blocking(self.read_from.fileno(), False)

    def read(self) -> dict:
        deadline = time.monotonic() + self.response_timeout
        with selectors.DefaultSelector() as selector:
            selector.register(self.read_from, selectors.EVENT_READ)
            while b'\n' not in self._response_buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    self.close()
                    raise InteractiveResponseTimeout(self.response_timeout, self.operation)
                chunk = os.read(self.read_from.fileno(), 65536)
                if not chunk:
                    code = self.proc.poll()
                    self.close()
                    raise InteractiveSessionError(
                        f'Lean Interactive stdout closed before a complete response (exit={code}); see stderr in execution log',
                        self.operation, code)
                self._response_buffer.extend(chunk)
        line, _, rest = self._response_buffer.partition(b'\n')
        self._response_buffer = bytearray(rest)
        line = line.strip()
        logger.debug("<-" + repr(line))
        try:
            return json.loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            self.close()
            raise InteractiveSessionError(f'Invalid Lean response: {exc}', self.operation, self.proc.returncode) from exc

    def write(self, data: dict):
        self.operation = data.get('method', 'openFile')
        data = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
        logger.debug("->" + data)
        try:
            print(data, file=self.write_to)
        except (BrokenPipeError, OSError, ValueError) as exc:
            code = self.proc.poll()
            self.close()
            raise InteractiveSessionError(f'Lean session write failed: {exc}', self.operation, code) from exc

    def open_file(self, path: Path, selectors: list[str | int | None]):
        assert not self.tactic_mode
        self.write({"filename": str(path), "selectors": selectors})

    def get_next_problem(self) -> str | None:
        assert not self.tactic_mode
        logger.debug("get_next_problem")
        response = self.read()
        decl_name = response.get("declName")
        self.tactic_mode = decl_name is not None
        return decl_name

    def request(self, method: str, params: dict):
        assert self.tactic_mode
        self.id += 1
        self.write({'id': self.id, 'method': method, 'params': params})
        response = self.read()
        try:
            return response['result']
        except KeyError:
            raise RuntimeError(response['error'])

    def run_tactic(self, sid: int, tactic: str, heartbeats: int = 200000000) -> int:
        return self.request('runTactic', {'sid': sid, 'tactic': tactic, 'heartbeats': heartbeats})

    def get_state(self, sid: int) -> list[Goal]:
        res = self.request('getState', {'sid': sid})
        return from_json(list[Goal], res)

    def get_messages(self, sid: int) -> list[str]:
        res = self.request('getMessages', {'sid': sid})
        return from_json(list[str], res)

    def resolve_name(self, sid: int, name: str) -> list[tuple[str, list[str]]]:
        return self.request('resolveName', {'sid': sid, 'name': name})

    def unify(self, sid: int, s1: str, s2: str) -> list[tuple[str, str | None]] | None:
        return self.request('unify', {'sid': sid, 's1': s1, 's2': s2})

    def new_state(self, state: list[ProofGoal]) -> int:
        return self.request('newState', {'state': [dataclasses.asdict(g) for g in state]})

    def get_position(self) -> dict:
        return self.request('getPosition', {})

    def give_up(self, sid: int) -> int:
        return self.request('giveUp', {'sid': sid})

    def commit(self, sid: int):
        self.request('commit', {'sid': sid})
        self.tactic_mode = False

    def close(self):
        """Reap lake and its Lean child, including timeout/error paths."""
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # A lake launcher may exit before its child. Always clean the group.
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.proc.wait(timeout=5)
        for stream in (self.write_to, self.read_from):
            try:
                stream.close()
            except (OSError, ValueError):
                pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    interactive = Interactive(Path(__file__).parent, Path("Example.lean"))
    sid = 0
    interactive.open_file(Path("Example.lean"), ["neg_is_some_none", "eq_trans_sym"])

    while True:
        problem = interactive.get_next_problem()
        if problem is None:
            break

        print("Solving problem", problem)
        sid = 0

        while True:
            state = interactive.get_state(sid)
            if not state:
                # proof complete
                interactive.commit(sid)
                break
            print("current state id:", sid)
            print(f"{len(state)} goals:")
            for goal in state:
                print(goal.pretty)

            command = input()
            try:
                if command.startswith(":s"):
                    sid = int(command.split()[1])
                elif command.startswith(":u"):
                    params = command.split()[1:]
                    result = interactive.unify(sid, params[0], params[1])
                    print(result)
                elif command.startswith(":n"):
                    sid = interactive.new_state([
                        ProofGoal(context=[ProofVariable("a", "Nat")], type="Nat"),
                        ProofGoal(context=[ProofVariable("b", "Nat")], type="Nat")
                    ])
                else:
                    sid = interactive.run_tactic(sid, command)
                messages = interactive.get_messages(sid)
                if messages:
                    print("Messages:", messages)
            except RuntimeError as e:
                err = e.args[0]
                print(f"Error {err['code']}: {err['message']}")
                print(err['data'])
