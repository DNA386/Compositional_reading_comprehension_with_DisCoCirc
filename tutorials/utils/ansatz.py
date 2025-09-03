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


class DiscardSim4Ansatz(Sim4Ansatz):
    def _ar(self, _: Functor, box: Box) -> Circuit:
        # Wrap the default implementation to automatically convert
        # custom discard and daggered boxes.
        if box.name == "DISCARD":
            return Discard()

        circuit = super()._ar(_, box)
        return circuit

