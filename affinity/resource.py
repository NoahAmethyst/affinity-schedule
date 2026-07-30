import math
import pandas as pd


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
        if not isinstance(other, BaseObject):
            return NotImplemented
        return BaseObject(
            "",
            self.cpu + other.cpu,
            self.mem + other.mem,
            self.gpu + other.gpu,
            self.disk + other.disk
        )

    def __sub__(self, other):
        if not isinstance(other, BaseObject):
            return NotImplemented
        return BaseObject(
            self.name,
            self.cpu - other.cpu,
            self.mem - other.mem,
            self.gpu - other.gpu,
            self.disk - other.disk
        )

    def __ge__(self, other):
        """ >= """
        if not isinstance(other, BaseObject):
            return NotImplemented
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
        if not isinstance(other, BaseObject):
            return NotImplemented
        return BasePod(
            "",
            self.cpu + other.cpu,
            self.mem + other.mem,
            self.gpu + other.gpu,
            self.disk + other.disk,
            "",
        )

    def __sub__(self, other):
        if not isinstance(other, BaseObject):
            return NotImplemented
        return BasePod(
            self.name,
            self.cpu - other.cpu,
            self.mem - other.mem,
            self.gpu - other.gpu,
            self.disk - other.disk,
            self.platform,
        )

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
        return max(
            self._usage_ratio(self.cpu, obj.cpu),
            self._usage_ratio(self.mem, obj.mem),
            self._usage_ratio(self.gpu, obj.gpu),
            self._usage_ratio(self.disk, obj.disk),
        )

    def min_usage(self, used: BaseObject) -> float:
        ratios = [
            self._usage_ratio(self.cpu, used.cpu),
            self._usage_ratio(self.mem, used.mem),
            self._usage_ratio(self.gpu, used.gpu),
            self._usage_ratio(self.disk, used.disk),
        ]
        if math.inf in ratios:
            return math.inf
        finite_ratios = [
            ratio
            for capacity, ratio in zip(
                (self.cpu, self.mem, self.gpu, self.disk),
                ratios,
            )
            if capacity > 0
        ]
        return min(finite_ratios, default=0.0)

    def usage(self, used):
        return BaseObject(
            "",
            self._usage_ratio(self.cpu, used.cpu),
            self._usage_ratio(self.mem, used.mem),
            self._usage_ratio(self.gpu, used.gpu),
            self._usage_ratio(self.disk, used.disk),
        )

    @staticmethod
    def _usage_ratio(capacity: float, used: float) -> float:
        if capacity > 0:
            return used / capacity
        return math.inf if used > 0 else 0.0

    def __add__(self, other):
        if not isinstance(other, BaseObject):
            return NotImplemented
        return BaseNode(
            "",
            self.cpu + other.cpu,
            self.mem + other.mem,
            self.gpu + other.gpu,
            self.disk + other.disk,
            self.net + getattr(other, "net", 0),
        )

    def __sub__(self, other):
        if not isinstance(other, BaseObject):
            return NotImplemented
        return BaseNode(
            self.name,
            self.cpu - other.cpu,
            self.mem - other.mem,
            self.gpu - other.gpu,
            self.disk - other.disk,
            self.net - getattr(other, "net", 0),
        )


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
        if pd.isna(p.parent):
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
