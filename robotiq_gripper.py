#!/usr/bin/env python3
import socket
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RobotiqGripper:
    host: str
    port: int = 63352
    timeout: float = 0.3          # 짧게: 응답 없는 커맨드에서 오래 안 기다리게
    recv_bytes: int = 4096
    activate_timeout_s: float = 2.0
    _activated: bool = field(default=False, init=False, repr=False)

    # ---- low-level ----
    def _send(self, cmd: str, expect_reply: bool) -> Optional[str]:
        """Send one command. If expect_reply=False, don't block waiting for recv."""
        if not cmd.endswith("\n"):
            cmd += "\n"

        with socket.create_connection((self.host, self.port), timeout=self.timeout) as s:
            s.settimeout(self.timeout)
            s.sendall(cmd.encode("ascii"))

            if not expect_reply:
                return None

            # Some servers reply quickly, some may send nothing for certain cmds.
            try:
                data = s.recv(self.recv_bytes)
            except socket.timeout:
                return None
            if not data:
                return None
            return data.decode("ascii", errors="ignore").strip()

    def _get_var(self, key: str) -> Optional[int]:
        """
        Parse integer reply for "GET <key>".
        Robust to multi-token replies like "... POS 123 ...".
        """
        resp = self._send(f"GET {key}", expect_reply=True)
        if not resp:
            return None
        tokens = resp.replace("\r", " ").replace("\n", " ").split()
        k = key.upper()
        for i in range(len(tokens) - 1):
            if tokens[i].upper() == k:
                try:
                    return int(tokens[i + 1])
                except ValueError:
                    return None
        return None

    def wait_until_stopped(self, timeout_s: float = 5.0, poll_dt: float = 0.05,
                           stable_time: float = 0.2, eps: int = 1) -> bool:
        """
        Returns True if gripper position stops changing for stable_time within timeout.
        eps: allowable pos jitter.
        """
        t0 = time.time()
        last = self.get_pos()
        if last is None:
            last = -999
        stable_start = None

        while time.time() - t0 < timeout_s:
            p = self.get_pos()
            if p is None:
                time.sleep(poll_dt)
                continue

            if abs(p - last) <= eps:
                if stable_start is None:
                    stable_start = time.time()
                elif time.time() - stable_start >= stable_time:
                    return True
            else:
                stable_start = None

            last = p
            time.sleep(poll_dt)

        return False

    # ---- queries ----
    def get_pos_raw(self) -> Optional[str]:
        """Return raw response, typically like 'POS 40'."""
        return self._send("GET POS", expect_reply=True)

    def get_pos(self) -> Optional[int]:
        """Return position as int [0..255] if parseable."""
        return self._get_var("POS")

    def get_status(self) -> Optional[int]:
        """Robotiq STA (typically: 0=reset, 1=activating, 3=active)."""
        return self._get_var("STA")

    def get_object_status(self) -> Optional[int]:
        """Robotiq OBJ status (object detection / motion state)."""
        return self._get_var("OBJ")

    def get_fault(self) -> Optional[int]:
        """Robotiq FLT fault code."""
        return self._get_var("FLT")

    def wait_until_active(self, timeout_s: Optional[float] = None, poll_dt: float = 0.05) -> bool:
        t0 = time.time()
        deadline = timeout_s if timeout_s is not None else self.activate_timeout_s
        while time.time() - t0 < deadline:
            sta = self.get_status()
            if sta == 3:
                return True
            time.sleep(poll_dt)
        return False

    # ---- commands (fire-and-forget by default) ----
    def activate(self, force: bool = False) -> bool:
        """
        Activate gripper in a protocol-compatible way.
        - Legacy servers: "ACT"
        - Robotiq socket protocol: "SET ACT 1" + "SET GTO 1"
        """
        if self._activated and not force:
            return True

        sta = self.get_status()
        if sta == 3:
            self._activated = True
            # Ensure go-to mode in case controller rebooted.
            self._send("SET GTO 1", expect_reply=False)
            return True

        # Some environments only support one variant. Send both safely.
        self._send("SET ACT 1", expect_reply=False)
        self._send("SET GTO 1", expect_reply=False)
        self._send("ACT", expect_reply=False)
        self._send("SET GTO 1", expect_reply=False)

        # Prefer explicit activation check when supported.
        if self.wait_until_active(timeout_s=self.activate_timeout_s):
            self._activated = True
            return True

        # Fallback: if POS can be read, communication is alive.
        if self.get_pos() is not None:
            self._activated = True
            return True

        return False

    def set_speed(self, speed: int) -> None:
        speed = max(0, min(255, int(speed)))
        self._send(f"SET SPE {speed}", expect_reply=False)

    def set_force(self, force: int) -> None:
        force = max(0, min(255, int(force)))
        self._send(f"SET FOR {force}", expect_reply=False)

    def goto(self, pos: int, speed: Optional[int] = None, force: Optional[int] = None,
             wait: bool = True, tol: int = 3, timeout_s: float = 5.0) -> bool:
        """
        Move to pos (0=open .. 255=close typical).
        Returns True if reached within tol (when wait=True), else False.
        """
        if not self.activate():
            raise RuntimeError("Robotiq gripper activation failed before motion command")

        pos = max(0, min(255, int(pos)))

        # Optional params first
        if speed is not None:
            self.set_speed(speed)
        if force is not None:
            self.set_force(force)

        # Command movement
        self._send(f"SET POS {pos}", expect_reply=False)

        if not wait:
            return True

        return self.wait_target(pos, tol=tol, timeout_s=timeout_s)

    def open(self, speed: int = 255, force: int = 255, wait: bool = True,
             tol: int = 3, timeout_s: float = 5.0) -> bool:
        return self.goto(0, speed=speed, force=force, wait=wait, tol=tol, timeout_s=timeout_s)

    def close(self, speed: int = 255, force: int = 255, wait: bool = True,
              tol: int = 3, timeout_s: float = 5.0) -> bool:
        if not self.activate():
            raise RuntimeError("Robotiq gripper activation failed before motion command")
        self.set_speed(speed)
        self.set_force(force)
        self._send("SET POS 255", expect_reply=False)

        if not wait:
            return True

        # Gripper가 완전히 닫히지 않을 수도 있으니, "정지"를 완료 조건으로 사용
        return self.wait_until_stopped(timeout_s=timeout_s)

    def wait_target(self, target: int, tol: int = 3, timeout_s: float = 5.0, poll_dt: float = 0.05) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            p = self.get_pos()
            if p is not None and abs(p - target) <= tol:
                return True
            time.sleep(poll_dt)
        return False


if __name__ == "__main__":
    g = RobotiqGripper("192.168.0.43")

    print("Current:", g.get_pos_raw())  # e.g., POS 40

    ok = g.open(wait=True)
    print("Open reached:", ok, "pos:", g.get_pos())

    time.sleep(0.1)

    ok = g.close(wait=True)
    print("Close reached:", ok, "pos:", g.get_pos())

    time.sleep(0.1)

    ok = g.goto(80, speed=200, force=40, wait=True)
    print("Goto 80 reached:", ok, "pos:", g.get_pos())
