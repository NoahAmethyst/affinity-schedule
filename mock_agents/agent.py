import ctypes
import multiprocessing
import socket
import threading
import time

from prometheus_client import start_http_server, Summary


PORT = 11111
METRICS_PORT = 11112
SOCKET_TIMEOUT_SECONDS = 1.0


class Agent:
    def __init__(
        self,
        cpu: int,
        memory: int,
        frequency: float,
        package: int,
        target: str,
        amount: int,
    ) -> None:
        if cpu < 0 or memory < 0 or package < 0 or amount < 0:
            raise ValueError("CPU、内存、包大小和发送次数不能为负数")
        if target and frequency <= 0:
            raise ValueError("存在通信目标时，通信频率必须大于 0")

        self.cpu = cpu
        self.memory = memory
        self.frequency = frequency
        self.package = package
        self.target = target
        self.amount = amount
        self.stop_event = multiprocessing.Event()
        self.busy_processes = new_busy_tasks(cpu, self.stop_event)
        self.busy_memory = new_busy_memory(memory)
        self.listen_process = multiprocessing.Process(target=self._listen_messages)
        self.send_process = (
            multiprocessing.Process(target=self._send_messages)
            if self.target
            else None
        )

    def run(self) -> None:
        print("start to run agent")
        for process in self.busy_processes:
            process.start()
        self.listen_process.start()
        if self.send_process is not None:
            self.send_process.start()

        try:
            while not self.stop_event.wait(SOCKET_TIMEOUT_SECONDS):
                if not self.listen_process.is_alive():
                    raise RuntimeError("socket server process stopped unexpectedly")
        finally:
            self.stop()
            self._join_processes()

    def stop(self, *_signal_args) -> None:
        self.stop_event.set()

    def _join_processes(self):
        processes = [
            *self.busy_processes,
            self.listen_process,
            *([self.send_process] if self.send_process is not None else []),
        ]
        for process in processes:
            if process.pid is not None:
                process.join(timeout=3)
        for process in processes:
            if process.pid is not None and process.is_alive():
                process.terminate()
                process.join(timeout=1)

    def _send_messages(self) -> None:
        start_http_server(port=METRICS_PORT)
        latency_summary = Summary(
            "request_latency_seconds",
            "Time taken for requests",
        )
        packet_size_bytes = int(self.package * 1024 * 1024)
        packet = b"x" * packet_size_bytes + b"e"

        while not self.stop_event.is_set():
            try:
                with socket.create_connection(
                    (self.target, PORT),
                    timeout=SOCKET_TIMEOUT_SECONDS,
                ) as sock:
                    sock.settimeout(SOCKET_TIMEOUT_SECONDS)
                    print(
                        f"connected to {self.target}:{PORT}; "
                        f"frequency={self.frequency}/s, package={self.package} MB"
                    )
                    for _ in range(self.amount):
                        if self.stop_event.is_set():
                            return
                        started_at = time.perf_counter()
                        sock.sendall(packet)
                        response = sock.recv(1024)
                        if response != b"ACK":
                            raise ConnectionError(
                                f"unexpected response from {self.target}: {response!r}"
                            )
                        latency = time.perf_counter() - started_at
                        latency_summary.observe(latency)
                        if self.stop_event.wait(1 / self.frequency):
                            return
                    return
            except (ConnectionError, OSError, socket.timeout) as exc:
                print(f"send messages failed: {exc}")
                if self.stop_event.wait(SOCKET_TIMEOUT_SECONDS):
                    return

    def _process_received_message(self, conn: socket.socket, addr) -> None:
        conn.settimeout(SOCKET_TIMEOUT_SECONDS)
        try:
            while not self.stop_event.is_set():
                try:
                    data = conn.recv(1024)
                except socket.timeout:
                    continue
                if not data:
                    return
                if b"e" in data:
                    conn.sendall(b"ACK")
        except OSError as exc:
            print(f"process received data from {addr} failed: {exc}")
        finally:
            conn.close()

    def _listen_messages(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", PORT))
            sock.listen(5)
            sock.settimeout(SOCKET_TIMEOUT_SECONDS)
            print(f"start to listen on 0.0.0.0:{PORT}")

            while not self.stop_event.is_set():
                try:
                    conn, addr = sock.accept()
                except socket.timeout:
                    continue
                thread = threading.Thread(
                    target=self._process_received_message,
                    args=(conn, addr),
                    daemon=True,
                )
                thread.start()


def new_busy_tasks(n: int, stop_event) -> list[multiprocessing.Process]:
    if n < 0:
        raise ValueError("CPU 核数不能为负数")
    return [
        multiprocessing.Process(target=cpu_intensive_task, args=(stop_event,))
        for _ in range(n)
    ]


def cpu_intensive_task(event):
    while not event.is_set():
        for _ in range(10_000_000):
            if event.is_set():
                return


def new_busy_memory(n: int):
    if n < 0:
        raise ValueError("内存大小不能为负数")
    if n == 0:
        return None
    target_memory_size_in_bytes = n * 1024 * 1024 * 1024
    return ctypes.create_string_buffer(target_memory_size_in_bytes)


if __name__ == "__main__":
    stop_event = multiprocessing.Event()
    tasks = new_busy_tasks(4, stop_event)
    for task in tasks:
        task.start()
    time.sleep(30)
    stop_event.set()
    for task in tasks:
        task.join()
