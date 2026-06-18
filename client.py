from __future__ import annotations

import json
import socket
import time

from agent import RacingBrain
from config import load_settings
from protocol import RequestData, ResponseData


class UnityTcpClient:
    def __init__(
        self,
        host: str,
        port: int,
        brain: RacingBrain,
        reconnect_delay: float,
    ) -> None:
        self.host = host
        self.port = port
        self.brain = brain
        self.reconnect_delay = reconnect_delay
        self._buffer = b""
        self._requests = 0
        self._responses = 0
        self._rate_timer = time.monotonic()

    def run(self) -> None:
        while True:
            try:
                if self._run_session():
                    return
            except KeyboardInterrupt:
                raise
            except ConnectionResetError:
                print("\nConnection reset by Unity.")
            except OSError as exc:
                print(f"\nConnection error: {exc}")

            print(f"\nRetrying in {self.reconnect_delay:.0f}s...")
            time.sleep(self.reconnect_delay)

    def _run_session(self) -> bool:
        self._buffer = b""
        with socket.create_connection((self.host, self.port), timeout=30) as sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"Connected to Unity at {self.host}:{self.port}")

            while True:
                line = self._read_line(sock)
                if line is None:
                    print("\nUnity closed the connection.")
                    return False

                self._requests += 1
                payload = json.loads(line)
                request = RequestData.from_json(payload)
                response = self.brain.handle_request(request)
                self._send_response(sock, response)
                self._responses += 1
                self._maybe_print_rate()

                if response.is_done:
                    print(f"\nTraining complete after episode {request.episode}.")
                    return True

        return False

    def _maybe_print_rate(self) -> None:
        now = time.monotonic()
        if now - self._rate_timer < 1.0:
            return
        print(f"\r{self._requests}/s {self._responses}/s", end="", flush=True)
        self._requests = 0
        self._responses = 0
        self._rate_timer = now

    def _read_line(self, sock: socket.socket) -> str | None:
        while b"\n" not in self._buffer:
            chunk = sock.recv(65536)
            if not chunk:
                if self._buffer:
                    line = self._buffer.decode("utf-8")
                    self._buffer = b""
                    return line
                return None
            self._buffer += chunk

        newline_index = self._buffer.index(b"\n")
        line = self._buffer[:newline_index].decode("utf-8")
        self._buffer = self._buffer[newline_index + 1 :]
        return line

    def _send_response(self, sock: socket.socket, response: ResponseData) -> None:
        payload = json.dumps(response.to_json_dict(), separators=(",", ":")) + "\n"
        sock.sendall(payload.encode("utf-8"))


def main() -> None:
    settings = load_settings()

    brain = RacingBrain(
        train=settings.train,
        use_baseline=settings.baseline,
        epsilon=settings.epsilon,
        epsilon_min=settings.epsilon_min,
        epsilon_decay=settings.epsilon_decay,
        model_name=settings.model_name,
        models_dir=settings.models_dir,
        load_episode=settings.load_episode,
        load_latest=settings.load_latest,
        max_episodes=settings.max_episodes,
        save_every_episodes=settings.save_every_episodes,
        evaluation_dir=settings.evaluation_dir,
        train_every_n_steps=settings.train_every_n_steps,
    )

    client = UnityTcpClient(
        settings.host,
        settings.port,
        brain,
        settings.reconnect_delay,
    )

    try:
        client.run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        brain.save_model()
        brain.shutdown()


if __name__ == "__main__":
    main()
