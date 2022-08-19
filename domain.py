from typing import List


class Technique:
    def __init__(self, name: str):
        self.name: str = name
        self.title: str = ''
        self.controls: List['Control'] = []
        self.scCount: int = 0
        self.tactics: List['Tactic'] = []
        self.years: List[int] = []
        self.prob: float = 0
        self.occurrence: int = 0
        self.severity: float = 0
        self.support: float = 0
        self.hf: float = 0
        self.ec: float = 0
        self.timesUnmitigated: int = 0
        self.lk: float = 0
        self.sv: float = 0
        self.risk: float = 0


class Tactic:
    def __init__(self, name):
        self.name: str = name
        self.title: str = ''
        self.techniques: List['Technique'] = []
        self.controls: List['Control'] = []


class Procedure:
    def __init__(self, id):
        self.id: str = id
        self.name: str = ''
        self.type: str = ''
        self.year: int = 0
        self.ttp: Technique


class Group:
    def __init__(self, name):
        self.name: str = name
        self.ttps: List['Technique'] = []
        self.controls: List['Control'] = []
        self.procedures: List['Procedure'] = []


class Software:
    def __init__(self, name):
        self.name: str = name
        self.ttps: List['Technique'] = []
        self.controls: List['Control'] = []
        self.procedures: List['Procedure'] = []


class Control:
    def __init__(self, name):
        self.name: str = name
        self.title: str = ''
        self.ttps: List['Technique'] = []
        self.tec: int = 0
        self.cat: str = ''
        self.baseline: int
        self.ti: float = 0
        self.ri: float = 0
        self.mtac: float = 0
        self.prob: float = 0
        self.tac: List[float] = []
        self.red: float = 0
        self.advCov: float = 0
        self.advTecCov: float = 0
        self.advTacCov: List[float] = [] 
        self.risk: float = 0
        self.rulesCov: float = 0
