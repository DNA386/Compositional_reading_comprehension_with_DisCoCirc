"""Parse babi diagrams directly without relying on the pipeline."""

from typing import List, Tuple
from lambeq.backend.grammar import Box, Ty, Id, Diagram, Frame, Functor, grammar
from utils.babi6.vocabulary import PEOPLE, LOCATIONS, OBJECTS, ITV, TV, PREP


def get_coref_type(coref: str | None = None) -> Ty:
    # Supress coreferencing and handle manually
    return Ty("n")


def noun_state(name) -> Box:
    return Box(name, Ty(), get_coref_type(name))


def itv(name, subj: str | None) -> Box:
    dom = get_coref_type(subj)
    return Box(name, dom, dom)


def tv(name, subj: str | None, obj: str | None = None) -> Box:
    dom = get_coref_type(subj) @ get_coref_type(obj)
    return Box(name, dom, dom)


def frame(
        name,
        nouns: List[str | None],
        insides: List[Box | Frame]
) -> Frame:
    dom = Ty().tensor(*[get_coref_type(noun) for noun in nouns])
    return Frame(name, dom, dom, components=insides)


class bAbI6Parser():
    n = Ty("n")

    # Sentence schemas
    sentence_schemas = {
        "PERSON TV the OBJECT": lambda s, v, o: (tv(v, s, o), [s, o]),
        "PERSON TV the OBJECT there": lambda s, v, o: (frame(
            "there", [s, o], insides=[tv(v, s, o)]
        ), [s, o]),
        "PERSON TV PREP the OBJECT": lambda s, v, p, o: (frame(
            p, [s, o], insides=[tv(v, s, o)],
        ), [s, o]),
        "PERSON TV PREP the OBJECT there": lambda s, v, p, o: (frame(
            "there", [s, o], insides=[frame(
                p, [s, o], insides=[tv(v, s, o)],
            )]
        ), [s, o]),
        "PERSON ITV to the LOCATION": lambda s, v, o: (frame(
            "to", [s, o], insides=[itv(v, s)]
        ), [s, o]),
        "PERSON ITV back to the LOCATION": lambda s, v, o: (frame(
            "to", [s, o], insides=[
                frame("back", [s], insides=[itv(v, s)])
            ]
        ), [s, o]),
    }
    question_schemas = {
        # PERSON is in the LOCATION
        True: lambda p, l: noun_state(p) @ noun_state(l) >> frame(
            "is", [p, l], insides=[itv("in", l)]
        ),
        # PERSON is not in the LOCATION
        False: lambda p, l: noun_state(p) @ noun_state(l) >> frame(
            "not", [p, l], insides=[frame(
                "is", [p, l], insides=[itv("in", l)]
            )]
        )
    }

    def get_sentence_format(self, sentence) -> Tuple[str, List[str]]:
        """Extract the sentence schema, and its variables."""
        replacements = {
            "PERSON": PEOPLE,
            "LOCATION": LOCATIONS,
            "OBJECT": OBJECTS,
            "ITV": ITV,
            "TV": TV,
            "PREP": PREP,
        }

        def replace(w: str) -> str:
            for replacement, check in replacements.items():
                if w in check:
                    return replacement
            return w

        schema = " ".join([replace(word) for word in sentence.split(" ")])
        words = [
            word for word in sentence.split(" ")
            if word in [r for rep in replacements.values() for r in rep]
        ]
        return schema, words

    def babi_text_to_diagram(self, context_str: str, person, location) -> Diagram:
        context = [c for c in context_str.split(". ") if len(c) > 0]
        # First add in all the nouns, so we can be sure they come first.
        # Add them in the order they are introduced in the text
        nouns = []
        for noun in [w for s in context for w in s.split(" ") if w in (PEOPLE + OBJECTS + LOCATIONS)]:
            if noun not in nouns:
                nouns.append(noun)
        context_diag = Id().tensor(*[noun_state(noun) for i, noun in enumerate(nouns)])
        for sentence in context:
            # Get the sentence diagram to append
            schema, words = self.get_sentence_format(sentence)
            if schema not in self.sentence_schemas:
                print(f"SENTENCE NOT IN SCHEMAS! '{sentence}' -> '{schema}'")
                continue
            sentence_diag, sentence_nouns = self.sentence_schemas[schema](*words)

            # apply permutations. Assume the context (co)domain is always in the order specified by 'nouns_order'
            # don't care about swap depth; put the new box on the left and pass all other wires to the right.
            other_nouns = [n for n in nouns if n not in sentence_nouns]
            perm = [nouns.index(n) for n in sentence_nouns + other_nouns]
            perm_diag = context_diag.permutation(context_diag.cod, perm)

            context_diag >>= perm_diag
            context_diag >>= sentence_diag @ (
                Ty(objects=[self.n for _ in other_nouns])
                if len(other_nouns) != 1
                else self.n
            )
            inv_perm_diag = perm_diag.dagger()
            context_diag >>= inv_perm_diag

        # Discard the nouns not involved in the question and permute into the
        # desired order.
        target_cod = [person, location]
        discards = Id().tensor(*[
            Box("DISCARD", self.n, Ty())
            if name not in target_cod
            else self.n
            for name in nouns
        ]) if len(context_diag.cod) > 1 else Id(context_diag.cod)
        context_diag >>= discards
        context_diag = context_diag.permuted([
            target_cod.index(name)
            for name in nouns
            if name in target_cod
        ])

        return context_diag

    def babi_question_to_diagram(self, question: str, positive=False) -> Diagram:
        # Remove trailing space and '?'
        question = question.split("?")[0].strip(" ")

        schema, words = self.get_sentence_format(question)

        question_diag = self.question_schemas[positive](*words)
        return question_diag


def babi_sandwich_ar(self, box):
    # Very hacky frame expansion that appies correctly for the shapes encountered in bAbI 6
    if isinstance(box, Frame):
        top = Box(f"[{box.name}]", box.dom, box.dom)
        # There's only one hole, which applies to the subject first.
        inside = babiSandwichFunctor(box.components[0])
        if len(inside.dom) < len(box.dom):
            inside = inside @ box.dom[len(inside.dom)]
        bottom = Box(f"[\\{box.name}]", box.cod, box.cod)
        return top >> inside >> bottom

    return box

babiSandwichFunctor = Functor(grammar, lambda self, o: o, babi_sandwich_ar)
