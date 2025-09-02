from lambeq.backend.grammar import Ty, Box, Word, Id, Diagram, Swap

from utils.ansatz import DBox


def find_wire(diagram: Diagram, name: str) -> int:
    index = 0
    box = diagram.boxes[index]
    while box.dom == Ty():
        if box.name == name:
            return diagram.offsets[index]

        index += 1
        box = diagram.boxes[index]

    raise ValueError(f'wire "{name}" not found')


class FollowingParser:
    def __init__(self, with_axioms=False, n_directions=4):
        self.with_axioms = with_axioms
        self.n_directions = n_directions

        self.n = Ty("n")

        self.person = Word("person", self.n)
        self.walks_north = DBox("walks_north", self.n, self.n)
        self.walks_south = DBox("walks_south", self.n, self.n)
        self.walks_east = DBox("walks_east", self.n, self.n)
        self.walks_west = DBox("walks_west", self.n, self.n)
        self.follows = DBox("follows", self.n**2, self.n**2)
        if self.with_axioms:
            self.turns_left = DBox("turns_left", self.n, self.n)
            self.turns_right = DBox("turns_left", self.n, self.n, dagger=True)
            if self.n_directions == 4:
                self.turns_around = self.turns_left >> self.turns_left
            else:
                self.turns_around = DBox("turns_around", self.n, self.n)
            self.goes_opdir = self.follows >> self.turns_around @ Id(self.n)
        else:
            self.turns_left = DBox("turns_left", self.n, self.n)
            self.turns_right = DBox("turns_right", self.n, self.n)
            self.turns_around = DBox("turns_around", self.n, self.n)
            self.goes_opdir = DBox("opp_dir", self.n**2, self.n**2)

        self.discard = DBox("DISCARD", self.n, Ty())

    def get_ques_diag(self, negative=False, nouns=False, higher_order=True):
        n = self.n
        diag = Box("goes_in_same_direction_as", n ** 2, n ** 2)

        if negative:
            if higher_order:
                nn0 = Box("[not]", n ** 2, n ** 2)
                nn1 = Box("[\\not]", n ** 2, n ** 2)
                diag = nn0 >> diag >> nn1
            else:
                diag = Box("goes_not_in_same_direction_as", n ** 2, n ** 2)

        if nouns:
            noun = self.person
            diag = noun @ noun >> diag

        return diag

    def parse(self, sents, actor1, actor2):
        nouns = []
        for sent in sents:
            words = sent.split()
            if words[0] not in nouns:
                nouns.append(words[0])

        diag = Word(nouns[0], self.n)
        for noun in nouns[1:]:
            diag = diag @ Word(noun, self.n)

        for sent in sents:
            sent = sent.replace(".", "")
            words = sent.split()

            if "north" in sent:
                act = find_wire(diag, words[0])
                diag = diag >> Id(self.n**act) @ self.walks_north @ Id(
                    self.n ** (len(diag.cod) - act - 1)
                )

            elif "east" in sent:
                act = find_wire(diag, words[0])
                diag = diag >> Id(self.n**act) @ self.walks_east @ Id(
                    self.n ** (len(diag.cod) - act - 1)
                )

            elif "south" in sent:
                act = find_wire(diag, words[0])
                diag = diag >> Id(self.n**act) @ self.walks_south @ Id(
                    self.n ** (len(diag.cod) - act - 1)
                )

            elif "west" in sent:
                act = find_wire(diag, words[0])
                diag = diag >> Id(self.n**act) @ self.walks_west @ Id(
                    self.n ** (len(diag.cod) - act - 1)
                )

            elif "right" in sent:
                act = find_wire(diag, words[0])
                diag = diag >> Id(self.n**act) @ self.turns_right @ Id(
                    self.n ** (len(diag.cod) - act - 1)
                )

            elif "left" in sent:
                act = find_wire(diag, words[0])
                diag = diag >> Id(self.n**act) @ self.turns_left @ Id(
                    self.n ** (len(diag.cod) - act - 1)
                )

            elif "around" in sent:
                act = find_wire(diag, words[0])
                diag = diag >> Id(self.n**act) @ self.turns_around @ Id(
                    self.n ** (len(diag.cod) - act - 1)
                )

            if "follows" in sent or "opposite" in sent:
                act1 = find_wire(diag, words[0])
                act2 = find_wire(diag, words[-1])

                p1 = [i for i in range(act1 + 1)]
                if act2 in p1:
                    p1.remove(act2)
                p2 = [i for i in range(act1 + 1, len(diag.cod))]
                if act2 in p2:
                    p2.remove(act2)

                perm = p1 + [act2] + p2
                swaps = diag.permutation(diag.cod, perm)

                diag = diag >> swaps

                if "follows" in sent:
                    diag = diag >> Id(self.n ** perm.index(act1)) @ self.follows @ Id(
                        self.n ** (len(diag.cod) - perm.index(act1) - 2)
                    )
                elif "opposite" in sent:
                    diag = diag >> Id(
                        self.n ** perm.index(act1)
                    ) @ self.goes_opdir @ Id(
                        self.n ** (len(diag.cod) - perm.index(act1) - 2)
                    )

                diag = diag >> swaps.dagger()

        # At the end of the diagram, discard all but the final actors
        actor_ids = [
            find_wire(diag, actor1),
            find_wire(diag, actor2),
        ]
        discards = Id(Ty()).tensor(*[
            self.discard if noun_wire_id not in actor_ids else Id(self.n)
            for noun_wire_id in range(len(diag.cod))
        ])
        diag >>= discards

        # Ensure the final actors are output in the expected order.
        if actor_ids[0] > actor_ids[1]:
            diag >>= Swap(self.n, self.n)

        return diag