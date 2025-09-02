from lambeq.backend.quantum import Diagram as Circuit
from lambeq.backend.grammar import Ty, Box, Word, Functor, grammar
from lambeq.backend.quantum import Discard
from lambeq.ansatz import Sim4Ansatz

from utils.following.generate_stories import NAMES


# Anonymise the characters by replacing their names with a 'Person' box.
def person_ar(self, box):
    n = Ty("n")
    person = Word("person", n)
    if box.name in NAMES:
        return person
    return box


PersonFunctor = Functor(grammar, lambda self, o: o, person_ar)


class DBox(Box):
    """Daggerable box. Implement this at the diagram level so that the axioms
    can all be implemented in the same place. Later we adapt the ansatz to
    handle this specific case.
    """
    def __init__(self, *args, dagger: bool = False, **kwargs):
        self.dagger = dagger
        super().__init__(*args, **kwargs)

    def __str__(self):
        return f"{super().__str__()}{'.dg' if self.dagger else ''}"

    def __repr__(self):
        return f"{super().__repr__()}{'.dg' if self.dagger else ''}"


class DaggerableSim4Ansatz(Sim4Ansatz):
    def _ar(self, _: Functor, box: Box) -> Circuit:
        # Wrap the default implementation to automatically convert
        # custom discard and daggered boxes.
        if box.name == "DISCARD":
            return Discard()

        circuit = super()._ar(_, box)
        if isinstance(box, DBox) and box.dagger:
            circuit = circuit.dagger()
        return circuit

