import logging
import multiprocessing
import threading
import time
import socket
import ctypes
from prometheus_client import start_http_server, Summary

PORT = 11111


class Agent:
    def __init__(self, cpu:int, memory:int, frequency:float, package:int, target:str, amount:int, logger: logging.Logger) -> None:
        self.cpu = cpu
        self.memory = memory
        self.frequency = frequency
        self.package = package
        self.target = target
        self.amount = amount
        self.logger = logger
        self.stop_event = multiprocessing.Event()

        self.busy_processes = new_busy_tasks(cpu, self.stop_event)
        logger.info("init the busy processes successfully")

        self.busy_memory = new_busy_memory(memory)
        logger.info("init the busy memory successfully")

        self.new_listen_process()
        logger.info("init the socket server successfully")

        if len(self.target) > 0:
            self.new_seed_process()
            logger.info("init the socket client successfully")

        
    
    def run(self) -> None:
        self.logger.info("start to run agent")

        for process in self.busy_processes:
            process.start()
        
        self.listen_process.start()
        if len(self.target) > 0:
            self.send_process.start()


        if len(self.target) > 0:
            self.send_process.join()
        
        self.listen_process.join()
        



    def stop(self) -> None:
        if not self.stop_event.is_set():
            self.stop_event.set()
    

    def new_seed_process(self)->None:
        def send_messages():
            start_http_server(11112)
            latency_summary = Summary('request_latency_seconds', 'Time taken for requests')

            while True:
                if self.stop_event.is_set():
                    break
                
                try:
                    packet_size_bytes = int(self.package * 1024 * 1024)
                    
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect((self.target, PORT))

                    self.logger.info(f"connected and start to send messages to {self.target}:{PORT} at {self.frequency} package/s, package size {self.package} MB.")

                    packet = b'x' * packet_size_bytes + b'e'

                    for i in range(self.amount):
                        if self.stop_event.is_set():
                            break

                        start_time = time.time()
                        sock.sendall(packet)
                        sock.recv(1024)
                        end_time = time.time()
                        
                        latency = end_time-start_time
                        latency_summary.observe(latency)
                        # self.logger.info(f'count: {latency_summary._count.get()}, sum: {latency_summary._sum.get()}')
                        # print(latency_summary._count.get())
                        # print(latency_summary._sum.get())
                        
                        time.sleep(1 / self.frequency)  # 根据频率控制发送间隔
                    
                    if not self.stop_event.is_set():
                        self.logger.info("agent has finished send task")
                        break


                except Exception as e:
                    self.logger.error(f'seed messsages failed: {e}')
                finally:
                    sock.close()
                    time.sleep(1)
        
        self.send_process = multiprocessing.Process(target=send_messages)

    def new_listen_process(self)->None:
        def process_reveived_message(conn:socket.socket, addr):
            try:
                while True:
                        data = conn.recv(1024)
                        # logger.info(f"receive from {addr} : {data.decode('utf-8')}")

                        if b'e' in data:
                            conn.sendall(b'ACK')
                        
                        if len(data) == 0 or self.stop_event.is_set():
                            break

            except Exception as e:
                self.logger.error(f'process received data failed: {e}')
            finally:
                conn.close()
        
        def listen_messages():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(("0.0.0.0", PORT))
                sock.listen(5)  # 开始监听，允许最多5个连接同时等待
                
                self.logger.info(f'start to listen on 0.0.0.0:{PORT}')

                while True:
                    if self.stop_event.is_set():
                        break

                    conn, addr = sock.accept()
                    
                    self.logger.info(f'get connected from {addr}')
        
                    client_thread = threading.Thread(target=process_reveived_message, args=(conn,addr))
                    client_thread.start()

            except Exception as e:
                self.logger.error(f'start socket server failed: {e}')
            finally:
                sock.close()
            
        
        self.listen_process = multiprocessing.Process(target=listen_messages)
        


def new_busy_tasks(n:int, stop_event)->list:
    processes = []

    if n > 2:
        for _ in range(n-2):
            p = multiprocessing.Process(target=cpu_intensive_task, args=(stop_event,))
            processes.append(p)
    return processes


def cpu_intensive_task(event):
    while True:
        result = 0

        for _ in range(10000000):
            result += 1
        
        if event.is_set():
            break

def new_busy_memory(n:int):
    target_memory_size_in_bytes = (n-1) * 1024 * 1024 * 1024 if n > 1 else 1024 * 1024 * 1024 * 1
    allocated_memory = ctypes.create_string_buffer(target_memory_size_in_bytes)
    return allocated_memory








if __name__ == '__main__':

    stop_event = multiprocessing.Event()
    tasks = new_busy_tasks(4, stop_event)
    for task in tasks:
        task.start()

    time.sleep(30)
    stop_event.set()
