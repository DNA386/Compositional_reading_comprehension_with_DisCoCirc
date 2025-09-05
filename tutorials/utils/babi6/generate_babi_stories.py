from typing import List, Dict, Set
from dataclasses import dataclass, field
from enum import Enum
import random


@dataclass
class BaseState():
    location: str | None = None
    supporting_facts: List[int] = field(default_factory=lambda: [])

    def __repr__(self):
        return f"In {self.location} ({self.supporting_facts})"


@dataclass
class ObjectState(BaseState):
    carried: str | None = None

    def __repr__(self):
        return f"In {self.location} with {self.carried} ({self.supporting_facts})"


@dataclass
class PersonState(BaseState):
    carrying: Set[str] = field(default_factory=lambda: set())

    def __repr__(self):
        return f"In {self.location} with {self.carrying} ({self.supporting_facts})"


@dataclass
class WorldState:
    people: List[str] = field(default_factory=lambda: [])
    objects: List[str] = field(default_factory=lambda: [])
    locations: List[str] = field(default_factory=lambda: [])
    state: Dict[str, ObjectState] = field(default_factory=lambda: {})

    def get_n_nouns(self):
        return len(self.people) + len(self.objects) + len(self.locations)

    def add_person(self, p):
        self.people.append(p)
        self.state[p] = PersonState()

    def add_object(self, o):
        self.objects.append(o)
        self.state[o] = ObjectState()

    def add_location(self, l):
        self.locations.append(l)

    def get_people_with_locations(self):
        return [p for p in self.people if self.state[p].location is not None]


class Answers(Enum):
    yes = 'yes'
    no = 'no'

move = lambda p, m, l, _:     f"{p} {m} to the {l}."
grab = lambda p, g, o, there: f"{p} {g} the {o}{' there' if there else ''}."
drop = lambda p, d, o, there: f"{p} {d} the {o}{' there' if there else ''}."


def update_location(p, l, world_state, i):
    if p not in world_state.people:
        world_state.add_person(p)
    if l not in world_state.locations:
        world_state.add_location(l)
    
    world_state.state[p].location = l
    world_state.state[p].supporting_facts = [i]
    for o in world_state.state[p].carrying:
        world_state.state[o].location = l
        world_state.state[o].supporting_facts.append(i)


def update_carrying(p, o, is_carrying, world_state, i):
    if p not in world_state.people:
        world_state.add_person(p)
    if o not in world_state.objects:
        world_state.add_object(o)

    # person and object must be in same location to interact.
    if world_state.state[p].location:
        world_state.state[o].location = world_state.state[p].location
    elif world_state.state[o].location:
        world_state.state[p].location = world_state.state[o].location
    
    if is_carrying:
        world_state.state[p].carrying.add(o)
        world_state.state[o].carried = p
        world_state.state[o].supporting_facts.append(i)
    elif o in world_state.state[p].carrying:
        world_state.state[p].carrying.remove(o)
        world_state.state[o].carried = None
        world_state.state[o].supporting_facts.append(i)
    else:
        world_state.state[o].supporting_facts.append(i)


def generate(
        people, objects, locations, movements, grabs, drops,
        depth=1, max_nouns=None
):
    attempts = 0
    while attempts < 200:
        attempts += 1
        try:
            world_state = WorldState()
            sentences = []
            actions = {
                move: movements,
                grab: grabs,
                drop: drops,
            }
            
            for i in range(depth):
                att = 0
                p = None
                o = None
                l = None
                there = None
                verb = None
                
                while att < 50:
                    accept = False
                    att += 1
                    # reset
                    p = None
                    o = None
                    l = None
                    there = None
                    verb = None
                    try:
                        sent_schema = random.choice([move, move, grab, drop])
                        verb = random.choice(actions[sent_schema])
                        there = random.choice([True, False])
                        p = random.choice(
                            people 
                            if max_nouns is None or world_state.get_n_nouns() < max_nouns
                            else world_state.people
                        )
                        o = random.choice([
                            ob 
                            for ob in (
                                objects 
                                if max_nouns is None or world_state.get_n_nouns() < max_nouns
                                else world_state.objects
                            )
                            if (sent_schema == grab and (ob not in world_state.state or world_state.state[ob].carried is None))
                            or ((sent_schema == drop) and ob in world_state.state and world_state.state[ob].carried == p)
                            or sent_schema == move
                        ])
                        l = random.choice([
                            loc
                            for loc in (
                                locations 
                                if max_nouns is None or world_state.get_n_nouns() < max_nouns
                                else world_state.locations
                            )
                            if loc not in world_state.state or not world_state.state[p].location == loc
                        ])
                        accept = True
                    except Exception as e:
                        pass

                if not accept:
                    raise ValueError("Ran out of attempts")

                sent = sent_schema(p, verb, l if sent_schema == move else o, there)
                sentences.append(sent)

                if sent_schema == move:
                    update_location(p, l, world_state, i)
                else:
                    update_carrying(p, o, sent_schema == grab, world_state, i)

            q_person = random.choice(world_state.get_people_with_locations())
            q_location = random.choice(world_state.locations)
        
            ans_class = Answers.yes if world_state.state[q_person].location == q_location else Answers.no
            question = f"Is {q_person} in the {q_location}?"

            return {
                "sentences": sentences,
                "question": question,
                "answer": ans_class.value,
                "support": ' '.join(str(i) for i in world_state.state[q_person].supporting_facts),
                "n_sentences": len(sentences),
                "n_nouns": world_state.get_n_nouns(),
                "question_nouns": (q_person, q_location),
            }, world_state
        except Exception as e:
            pass
    raise ValueError("Ran out of attempts")
