import math
import pandas as pd
import numpy as np


class BaseObject:
    static_columns = []
    name = ""
    cpu = 0
    mem = 0
    gpu = 0
    disk = 0

    def __init__(self, name="", cpu=0, mem=0, gpu=0, disk=0):
        self.name = name
        self.cpu = cpu
        self.mem = mem
        self.gpu = gpu
        self.disk = disk

    def __str__(self):
        return f"{self.name}"

    def to_string(self):
        return f"{self.name},cpu:{self.cpu:.2f},mem:{self.mem:.2f},gpu:{self.gpu:.2f},disk:{self.disk:.2f}"

    def __add__(self, other):
        return BaseObject(
            "",
            self.cpu + other.cpu,
            self.mem + other.mem,
            self.gpu + other.gpu,
            self.disk + other.disk
        )

    def __sub__(self, other):
        return BaseObject(
            self.name,
            self.cpu - other.cpu,
            self.mem - other.mem,
            self.gpu - other.gpu,
            self.disk - other.disk
        )

    def __ge__(self, other):
        """ >= """
        if self.cpu < other.cpu:
            return False
        if self.mem < other.mem:
            return False
        if self.gpu < other.gpu:
            return False
        if self.disk < other.disk:
            return False
        return True

    def is_not_empty(self) -> bool:
        """ 资源值是否全部大于等于0 """
        return self.cpu >= 0 and self.gpu >= 0 and self.mem >= 0 and self.disk >= 0

    @classmethod
    def from_dataframe(cls, data: pd.Series):
        return cls(*[data[idx] for idx in cls.static_columns])


class BasePod(BaseObject):
    static_columns = ["name", "cpu", "mem", "gpu", "disk", "platform"]
    affinity_weight = [100, 1, 1, 1]

    def __init__(self, name="", cpu=0, mem=0, gpu=0, disk=0, platform=""):
        super().__init__(name, cpu, mem, gpu, disk)
        self.platform = platform

    def __str__(self):
        return super().__str__()

    def get_data(self) -> []:
        return [self.name, self.cpu, self.mem, self.gpu, self.disk, self.platform]

    def get_data_without_name(self) -> []:
        return [self.cpu, self.mem, self.gpu, self.disk, self.platform]

    def __hash__(self):
        return hash(self.name)

    def __add__(self, other):
        result = super().__add__(other)
        result.__class__ = BasePod
        return result

    def __sub__(self, other):
        result = super().__sub__(other)
        result.__class__ = BasePod
        return result

    @classmethod
    def get_columns(cls) -> list[str]:
        return cls.static_columns

    @classmethod
    def race_affinity(cls, x, y) -> float:
        x_data = x.get_data_without_name()
        y_data = y.get_data_without_name()
        result = 0.0
        for i, x, y in zip(cls.affinity_weight, x_data, y_data):
            result += i * (x * y) / (x + y + 0.01)
        return result


class BaseNode(BaseObject):
    static_columns = ['name', 'cpu', 'memory', 'gpu', 'disk', 'net']
    net = 0

    def __init__(self, name, cpu, mem, gpu, disk, net):
        super().__init__(name, cpu, mem, gpu, disk)
        self.net = net

    def __hash__(self):
        return hash(self.name)

    def get_data(self) -> []:
        return [self.name, self.cpu, self.mem, self.gpu, self.disk, self.net]

    @classmethod
    def get_columns(cls) -> list[str]:
        return cls.static_columns

    def is_not_empty(self) -> bool:
        """ 资源值是否全部大于等于0 """
        return super().is_not_empty() and self.net >= 0

    def max_usage(self, obj: BaseObject) -> float:
        """ pod 在 node 中的最大资源占比 """
        v = 0
        if self.cpu != 0:
            tmp = obj.cpu / self.cpu
            if tmp > v:
                v = tmp
        if self.gpu != 0:
            tmp = obj.gpu / self.gpu
            if tmp > v:
                v = tmp
        elif obj.gpu > 0:
            return math.inf
        if self.mem != 0:
            tmp = obj.mem / self.mem
            if tmp > v:
                v = tmp
        if self.disk != 0:
            tmp = obj.disk / self.disk
            if tmp > v:
                v = tmp
        return v

    def min_usage(self, used: BaseObject) -> float:
        v = math.inf
        if self.cpu != 0:
            tmp = used.cpu / self.cpu
            if tmp < v:
                v = tmp
        if self.gpu != 0:
            tmp = used.gpu / self.gpu
            if tmp < v:
                v = tmp
        elif used.gpu > 0:
            return math.inf
        if self.mem != 0:
            tmp = used.mem / self.mem
            if tmp < v:
                v = tmp
        if self.disk != 0:
            tmp = used.disk / self.disk
            if tmp < v:
                v = tmp
        return v

    def usage(self, used):
        result = BaseObject("", 0, 0, 0, 0)
        if self.cpu != 0:
            result.cpu = used.cpu / self.cpu
        if self.gpu != 0:
            result.gpu = used.gpu / self.gpu
        elif result.gpu > 0:
            result.gpu = math.inf
        if self.mem != 0:
            result.mem = used.mem / self.mem
        if self.disk != 0:
            result.disk = used.disk / self.disk
        return result

    def __add__(self, other):
        result = super().__add__(other)
        result.__class__ = BaseNode
        return result

    def __sub__(self, other):
        result = super().__sub__(other)
        result.__class__ = BaseNode
        return result


class BasePlatform:
    static_columns = ["name", "parent"]

    def __init__(self, name: str, parent=None):
        self.parent = parent
        self.name = name
        self.children = {}
        self.pods = []

    def __str__(self):
        return f"{self.name}"

    def add_parent(self, platform):
        self.parent = platform

    def add_child(self, platform):
        self.children[platform.name] = platform

    def add_pod(self, pod: BasePod):
        self.pods.append(pod)

    @classmethod
    def get_columns(cls) -> list[str]:
        return cls.static_columns

    def get_data(self):
        return [self.name, self.parent]

    @classmethod
    def from_dataframe(cls, data: pd.Series):
        p = cls(*[data[idx] for idx in cls.static_columns])
        if p.parent is np.nan:
            p.parent = None
        return p


class Communication:
    static_columns = ['target', 'source', 'frequency', 'package', 'count']
    src_pod = None
    tgt_pod = None
    freq = None
    package = None
    count = None

    def __init__(self, src, tgt, freq, pak, cnt):
        self.src_pod = src
        self.tgt_pod = tgt
        self.freq = freq
        self.package = pak
        self.count = cnt

    def get_data(self) -> []:
        return [self.tgt_pod, self.src_pod, self.freq, self.package, self.count]

    def to_string(self) -> str:
        return f"{self.freq}:{self.package}:{self.count}"

    @classmethod
    def get_columns(cls) -> list[str]:
        return cls.static_columns

    @classmethod
    def from_dataframe(cls, data: pd.Series):
        return Communication(*[data[idx] for idx in cls.static_columns])


class SingleSchedulerPlan:
    def __init__(self, pod: str, scheduled_node):
        self.pod = pod
        self.scheduled_node = scheduled_node

