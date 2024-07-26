from discopy import Ty, Word, Box, Id, Functor
import discopy.monoidal

names = [
    "Bob",
    "Alice",
    "Daniel",
    "Dorothy",
    "Paul",
    "Helen",
    "Jason",
    "Ruth",
    "Michael",
    "Linda",
    "Brian",
    "Donna",
    "Matthew",
    "Betty",
    "Charles",
    "Patricia",
    "James",
    "Susan",
    "George",
    "Sarah",
    "Richard",
    "Karen",
    "Christopher",
    "Nancy",
    "Steven",
    "Carol",
    "Kevin",
    "Anna",
    "Edward",
    "Lisa",
    "Eric",
    "Michelle",
    "Timothy",
    "Jennifer",
    "Robert",
    "Kimberly",
    "Mark",
    "Jessica",
    "David",
    "Laura",
    "Joseph",
    "Maria",
    "John",
    "Sharon",
    "William",
    "Elizabeth",
    "Andrew",
    "Emily",
    "Thomas",
    "Sandra",
    "Kenneth",
    "Mary",
    "Ben",
    "Margaret",
    "Jack",
    "Paula",
    "Ethan",
    "Natalie",
    "Peter",
    "Victoria",
    "Charlie",
]


def find_wire(diagram: discopy.monoidal.Diagram, name: str) -> int:
    index = 0
    box = diagram.boxes[index]
    while box.dom == Ty():
        if box.name == name:
            return diagram.offsets[index]

        index += 1
        box = diagram.boxes[index]

    raise ValueError(f'wire "{name}" not found')


def get_ques_diag(neg=False, nouns=False, ho=True):
    n = Ty("n")
    diag = Box("goes_in_same_direction_as", n**2, n**2)

    if neg:
        if ho:
            nn0 = Box("[not]", n**2, n**2)
            nn1 = Box("[\\not]", n**2, n**2)
            diag = nn0 >> diag >> nn1
        else:
            diag = Box("goes_not_in_same_direction_as", n**2, n**2)

    if nouns:
        noun = Word("person", n)
        diag = noun @ noun >> diag

    return diag


def person_ar(box):
    n = Ty("n")
    person = Word("person", n)
    if box.name in names:
        return person
    return box

PersonFunctor = Functor(lambda o: o, person_ar)


class FollowingParserNoHO:
    def __init__(self):
        self.n = Ty("n")

        self.turns_left = Box("turns_left", self.n, self.n)
        self.turns_right = Box("turns_right", self.n, self.n)
        self.turns_around = Box("turns_around", self.n, self.n)

        self.follows = Box("follows", self.n**2, self.n**2)

        self.goes_opdir = Box("opp_dir", self.n**2, self.n**2)

        self.walks_north = Box("walks_north", self.n, self.n)
        self.walks_south = Box("walks_south", self.n, self.n)
        self.walks_east = Box("walks_east", self.n, self.n)
        self.walks_west = Box("walks_west", self.n, self.n)

    def parse(self, sents):
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

                diag = diag.permute(*perm, inverse=True)

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

                diag = diag.permute(*perm, inverse=False)

        return diag
