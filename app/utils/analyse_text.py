import re

MODIFIERS = {
    "follows": 0,
    "goes": 2,  # opposite

    "left": 3,
    "right": 1,
    "around": 2,

    "north": 0,
    "east": 1,
    "south": 2,
    "west": 3,
}
INV_MODIFIERS = {
    # inverses for intial directions
    0: "north",
    1: "east",
    2: "south",
    3: "west",
}


def decompose_sent(sent):
    # trim trailing full stops and spaces.
    sent = re.sub('\.*\s*$', '', sent)
    words = sent.split(' ')
    subj = words[0]
    verb = words[1]
    obj = None if verb == "turns" or verb == "walks" else words[-1]
    direction = words[-1] if verb == "turns" or verb == "walks" else None
    bottom = verb == "walks"
    mod = MODIFIERS[verb if verb in MODIFIERS else direction]
    return subj, verb, obj, mod, bottom


def update_state(tracker, bot, mod, steps, entangle_depth, modifier, obj, bottom):
    # If already bottomed out, do not update.
    if bot:
        return tracker, bot, mod, steps, entangle_depth

    steps += 1
    mod = (mod + modifier) % 4
    if obj is not None:
        tracker = obj
        entangle_depth += 1

    return tracker, (bot or bottom), mod, steps, entangle_depth


def get_inf_steps(story, char1, char2, ans=None):
    """
    To solve, we need to find out which direction each person is facing, by working backwards.
    We can also shortcut if both of the characters we are currently tracking get directly related to one another.
    Confirm solution found against ans
    """
    same_dir = None
    steps1, steps2 = 0, 0
    separated = True
    entangle_depth1 = 0
    entangle_depth2 = 0
    # track the names we currently care about
    tracker1 = char1
    tracker2 = char2
    # track whether the path has bottomed out
    bottom1 = False
    bottom2 = False
    # track the modifier to the current location (mod4)
    mod1 = 0
    mod2 = 0
    # track the sentence index at which the characters become entangled
    resume_at = None
    try:
        for i, sent in enumerate(story[::-1]):
            subj, verb, obj, modifier, bottom = decompose_sent(sent)

            # update trackers and modifiers
            if tracker1 == subj:
                tracker1, bottom1, mod1, steps1, entangle_depth1 = update_state(
                    tracker1, bottom1, mod1, steps1, entangle_depth1, modifier, obj, bottom
                )
            elif tracker2 == subj:
                tracker2, bottom2, mod2, steps2, entangle_depth2 = update_state(
                    tracker2, bottom2, mod2, steps2, entangle_depth2, modifier, obj, bottom
                )

            if tracker1 == tracker2:
                separated = False  # Characters are 'entangled'.
                # Make sure we got the correct answer if supplied
                same_dir = mod1 == mod2
                assert ans is None or same_dir == ans, "Did not match at common ancestor"
                # Record the details we need to search for the initial directions later.
                resume_at = i
                break

        if separated:
            # Characters are tensor separated; confirm we've bottomed out to the solution
            assert bottom1, f"Undefined direction for {char1}"
            assert bottom2, f"Undefined direction for {char2}"

            same_dir = mod1 == mod2
            assert ans is None or same_dir == ans, "Did not match at initialisations"
        else:
            # Need to continue searching for the initial directions. Both trackers point to the same name.
            for sent in story[:resume_at:-1]:
                subj, verb, obj, mod, bottom = decompose_sent(sent)

                if tracker1 == subj:
                    tracker1, bottom1, mod1, _, _ = update_state(tracker1, bottom2, mod1, 0, 0, mod, obj, bottom)
                    tracker2, bottom2, mod2, _, _ = update_state(tracker2, bottom2, mod2, 0, 0, mod, obj, bottom)

    except AssertionError as err:
        print(err)
        raise ValueError("Could not solve context: " + err.__repr__()) from err

    return {
        "steps": (steps1, steps2),
        "entangle_depth": (entangle_depth1, entangle_depth2),
        "same_dir": same_dir,
        "final_dir": (
            INV_MODIFIERS[mod1] if bottom1 else None,
            INV_MODIFIERS[mod2] if bottom2 else None,
        ),
        "separated": separated,
    }
