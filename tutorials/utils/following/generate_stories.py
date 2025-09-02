import random
from tqdm import tqdm


NAMES = [
    'Bob', 'Alice', 'Daniel', 'Dorothy', 'Paul', 'Helen', 'Jason', 'Ruth', 'Michael', 'Linda', 'Brian', 'Donna', 'Matthew',
    'Betty', 'Charles', 'Patricia', 'James', 'Susan', 'George', 'Sarah', 'Richard', 'Karen', 'Christopher', 'Nancy', 'Steven',
    'Carol', 'Kevin', 'Anna', 'Edward', 'Lisa', 'Eric', 'Michelle', 'Timothy', 'Jennifer', 'Robert', 'Kimberly', 'Mark',
    'Jessica', 'David', 'Laura', 'Joseph', 'Maria', 'John', 'Sharon', 'William', 'Elizabeth', 'Andrew', 'Emily', 'Thomas',
    'Sandra', 'Kenneth', 'Mary', 'Ben', 'Margaret', 'Jack', 'Paula', 'Ethan', 'Natalie', 'Peter', 'Victoria'
]


class Actor:
    def __init__(self, name, direction=None, n_directions=4):
        self.name = name
        # The following properties are used to calculate whether two actors have entangled directions.
        self.tracker = 0  # Track the number of direction resets.
        self.depends_on = []  # Track the actors whose direction determine this one.
        self.direction_choices = {
            2: ["north", "south"],
            4: ["north", "east", "south", "west"],
        }
        if direction is not None:
            self.direction = direction
        else:
            self.direction = random.choice(self.direction_choices[n_directions])

        self.turn_dict = {
            "right": {
                "north": "east",
                "east": "south",
                "south": "west",
                "west": "north",
            },
            "around": {
                "north": "south",
                "south": "north",
                "east": "west",
                "west": "east",
            },
            "left": {
                "north": "west",
                "west": "south",
                "south": "east",
                "east": "north",
            },
        }

    @property
    def current_name(self):
        return f"{self.name}{self.tracker}"

    def follows(self, actor):
        self.direction = actor.direction
        self.tracker += 1
        self.depends_on = actor.depends_on + [actor.current_name]
        return f"{self.name} follows {actor.name}. "

    def opposite_direction_of(self, actor):
        self.direction = self.turn_dict["around"][actor.direction]
        self.tracker += 1
        self.depends_on = actor.depends_on + [actor.current_name]
        return f"{self.name} goes in the opposite direction of {actor.name}. "

    def turns(self, turn_direction):
        self.direction = self.turn_dict[turn_direction][self.direction]
        return f"{self.name} turns {turn_direction}. "


class Story:
    def __init__(self, actors, n_sentences, n_directions=4):
        self.actor_init = actors
        self.n_actors = len(actors)
        self.n_sentences = n_sentences
        self.events = ["follows", "turns", "op_dir_of"]
        self.turn_direction_choices = {
            2: ["around"],
            4: ["left", "right", "around"],
        }
        self.active_actors = []
        self.story = []
        self.n_directions = n_directions

    def init_actor(self, actor):
        if self.actor_init:
            self.active_actors.append(actor)
            self.story.append(f"{actor.name} walks {actor.direction}. ")

    def event(self, p_2qb):
        ev = random.choices(self.events, weights=[p_2qb / 2, 1 - p_2qb, p_2qb / 2])[0]

        if ev == "follows":
            act1, act2 = random.sample(self.active_actors, 2)
            if self.story[-1] != f"{act1.name} follows {act2.name}. ":
                self.story.append(act1.follows(act2))
            else:
                self.event(p_2qb)

        elif ev == "turns":
            act = random.choice(self.active_actors)
            turn_direction = random.choice(self.turn_direction_choices[self.n_directions])
            self.story.append(act.turns(turn_direction))

        elif ev == "op_dir_of":
            act1, act2 = random.sample(self.active_actors, 2)
            if (
                self.story[-1]
                != f"{act1.name} goes in the opposite direction of {act2.name}. "
            ):
                self.story.append(act1.opposite_direction_of(act2))
            else:
                self.event(p_2qb)

            return None

    def generate_dense(self, n_single_ev=None):
        """Generate stories with a controlled proportion of two actor vs single
        actor sentences. Two-actor interactions are sampled without replacement
        from all possible pairs.

        params
        ------
            n_single_ev: The number of sentences concerning only a single actor,
            not including the initializations.
        """
        for act in self.actor_init:
            self.init_actor(act)

        if n_single_ev is None:
            n_single_ev = self.n_actors

        ev_tuples = []
        for i in range(n_single_ev):
            turn_direction = random.choice(self.turn_direction_choices[self.n_directions])
            act = random.choice(self.active_actors)
            ev_tuples.append((turn_direction, act))

        for i, act1 in enumerate(self.active_actors):
            for act2 in self.active_actors[i + 1:]:
                ent_ev = random.choice(["follows", "op_dir_of"])
                actor1, actor2 = random.sample([act1, act2], 2)

                ev_tuples.append((ent_ev, actor1, actor2))

        random.shuffle(ev_tuples)

        for ev in ev_tuples[:(self.n_sentences-self.n_actors)]:
            if ev[0] == "right":
                self.story.append(ev[1].turns("right"))

            elif ev[0] == "left":
                self.story.append(ev[1].turns("left"))

            elif ev[0] == "around":
                self.story.append(ev[1].turns("around"))

            elif ev[0] == "follows":
                self.story.append(ev[1].follows(ev[2]))

            elif ev[0] == "op_dir_of":
                self.story.append(ev[1].opposite_direction_of(ev[2]))

        return self.story

    def generate(self, p_2qb=0.7):
        """Generate a story with the specified probability of sampling a two actor sentence.
        Each event is sampled independently."""
        for act in self.actor_init:
            self.init_actor(act)

        while len(self.story) < self.n_sentences:
            self.event(p_2qb)

        return self.story


def generate_stories(
    min_actors, max_actors,
    min_sentences, max_sentences,
    n_samples,
    density,
    n_directions,
):
    """Generate stories with the requested shapes.
    The stories are sorted by shape (number of actors, number of sentences),
    and answer class (positive or negative), and checked to avoid duplicates.
    We additionally record the final directions faced by the actors, as well
    as whether their directions are `entangled` (ie depend on one another,
    possibly via a chain of intermediate actors rather than being causally
    separate).
    """
    used_names = NAMES[:max_actors]
    n_single_ev = {
        "simple": 0,
        "less-dense": 2,
        "dense": 1,
        "superdense": 0,
    }[density]

    stories = {}
    for n_act in tqdm(range(min_actors, max_actors + 1), desc=f"Generating stories"):
        for n_sents in range(
            max(n_act, min_sentences), min(n_act * 6 + 1, max_sentences + 1)
        ):
            key = (n_act, n_sents)
            if key not in stories:
                stories[key] = {"pos": [], "neg": []}

            while (
                len(stories[key]["pos"]) < n_samples
                or len(stories[key]["neg"]) < n_samples
            ):
                actors = [Actor(name=name, n_directions=n_directions) for name in used_names]
                story = Story(actors[:n_act], n_sents, n_directions=n_directions)
                if density in ["less-dense", "dense", "superdense"]:
                    s = story.generate_dense(n_single_ev=n_single_ev*n_act)
                elif density in ["simple"]:
                    s = story.generate()
                else:
                    raise NotImplementedError(
                        f"{density} not recognized. "
                        "Please select one of simple, less-dense, dense, superdense."
                    )

                act1, act2 = random.sample(story.active_actors, 2)
                story_dict = {
                    "story": s,
                    "actor1": act1.name,
                    "actor2": act2.name,
                    "answer": act1.direction == act2.direction,
                    "dir1": act1.direction,
                    "dir2": act2.direction,
                    "entangled": act1.current_name in act2.depends_on # direct dependence
                                 or act2.current_name in act1.depends_on
                                 # indirect via a third actor
                                 or len(set(act1.depends_on).intersection(set(act2.depends_on))) > 0,
                }
                if act1.direction == act2.direction:
                    if (
                        story_dict not in stories[key]["pos"]
                        and len(stories[key]["pos"]) < n_samples
                    ):
                        stories[key]["pos"].append(story_dict)

                else:
                    if (
                        story_dict not in stories[key]["neg"]
                        and len(stories[key]["neg"]) < n_samples
                    ):
                        stories[key]["neg"].append(story_dict)

    return stories