from valor.symbolic.modifiers import Variable
from valor.symbolic.atomics import Float

class Area(Variable):
    
    @property
    def area(self):
        if self.is_value():
            raise ValueError
        return Float(name=self._name, attribute="area")