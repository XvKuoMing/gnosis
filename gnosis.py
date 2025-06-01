from typing import List, Dict, Callable, Optional

from rebuilder import Rebuilder
from dataclasses import dataclass
from abc import ABC, abstractmethod



@dataclass(frozen=True)
class BaseTerminal(ABC):
    name: str
    
    def __post_init__(self):
        assert self.name.upper() == self.name, "name must be uppercase"

    @property
    @abstractmethod
    def as_terminal(self) -> str:
        pass


@dataclass(frozen=True)
class Class(BaseTerminal):
    name: str
    values: List[str] # a list of keywords/keyphrases
    fuzzy_temperature: float = 0.5

    def __post_init__(self):
        assert self.fuzzy_temperature >= 0. and self.fuzzy_temperature <= 1., "fuzzy_temperature must be between 0 and 1"
        super().__post_init__()

    def __calculate_fuzzy_temperature(self, value: str):
        return max(0, int(len(value) * self.fuzzy_temperature - 1))

    @property
    def as_terminal(self):
        _values = []
        for value in self.values:
            temp = self.__calculate_fuzzy_temperature(value)
            fuzzy_postfix = ""
            if temp > 0:
                fuzzy_postfix = "{" + f"e<={temp}" + "}"
            value = f"(?:{value}){fuzzy_postfix}"
            _values.append(value)
        return f"{self.name}: /{'|'.join(_values)}/i"


class Gnosis:


    @staticmethod
    def setup_grammar(start: str, schema: str, terminals: List[BaseTerminal]):
        _terminals = "\n".join([terminal.as_terminal for terminal in terminals])
        return f"""
start: {start}
{schema}
{_terminals}
"""

    def __init__(self, 
                 start: str, 
                 schema: str, 
                 terminals: List[BaseTerminal],
                 term_subs: Optional[Dict[str, str]] = None,
                 token_transformer: Optional[Callable[[str, str, bool], str]] = None):
        # in theory, terminls must be anything that transform something into terminal def
        self.__start = start
        self.__schema = schema
        self.__terminals = terminals
        self.__grammar = self.__class__.setup_grammar(start, schema, terminals)
        self.__rebuilder = Rebuilder(
            grammar=self.__grammar, 
            term_subs=term_subs, 
            token_transformer=token_transformer)
    
    @property
    def terminals(self):
        return self.__terminals

    @property
    def start(self):
        return self.__start
    
    @property
    def schema(self):
        return self.__schema
    
    @property
    def grammar(self):
        return self.__grammar
    
    def repair(self, input_: str, *args, **kwargs):
        return self.__rebuilder.repair(input_, *args, **kwargs)



class Classifier(Gnosis):

    SEPARATOR = ">>"
    MISSING_CLASS = "MISSING_CLASS"

    @staticmethod
    def missing_class(token_name: str, token_def: str, is_regex: bool):
        if is_regex:
            return Classifier.MISSING_CLASS
        else:
            return token_def
    

    @staticmethod
    def conditional_rule(name: str):
        return f'{name} "{Classifier.SEPARATOR}" "{name.lower()}"'

    def __init__(self, classes: List[Class]):
        super().__init__(
            start="class",
            schema=f"class: {' | '.join([self.__class__.conditional_rule(class_.name) for class_ in classes])}",
            terminals=classes,
            token_transformer=self.__class__.missing_class)
    
    def repair(self, input_: str, *args, **kwargs):
        result = super().repair(input_, *args, **kwargs)
        input_, output = result.split(Classifier.SEPARATOR)
        if input_ == Classifier.MISSING_CLASS:
            return Classifier.MISSING_CLASS
        return output





