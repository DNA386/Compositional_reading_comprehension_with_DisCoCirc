import os.path
import base64
from uuid import uuid4
from pathlib import Path
from discopy.quantum import Discard, Swap, CircuitFunctor, Ket
from discopy.rigid import Ty, Id
import pickle
import torch
from app.utils.parsing import get_ques_diag, find_wire, FollowingParserNoHO, PersonFunctor
from app.utils.analyse_text import get_inf_steps

CONTEXT_IMG_PATH = path=Path(os.path.dirname(__file__)).parent / 'temp'

class ShowcaseModel(FollowingParserNoHO):
    model_functor: CircuitFunctor
    model_path: str
    boxes_path: str
    name: str

    def __init__(self):
        super(ShowcaseModel, self).__init__()
        with open(self.model_path, "rb") as f:
            state_dict = torch.load(f)
        self.state_dict = state_dict

        with open(self.boxes_path, "rb") as f:
            self.boxes = pickle.load(f)

        self.free_syms = list(set(f for box in self.boxes.values() for f in box.free_symbols))
        self.sub_params = [self.state_dict["model"][f"params.{param}"].item() for param in self.free_syms]

        self.subbed_boxes = {
            k: v.lambdify(*self.free_syms)(*self.sub_params)
            for k, v in self.boxes.items()
        }

        def ar(a):
            # Remove explicit names
            a = PersonFunctor(a)
            # Convert to circuit
            if a.name in self.boxes:
                return self.subbed_boxes[a.name]

            raise ValueError("no circuit found for", a.name)

        self.model_functor = CircuitFunctor(ob={self.n: 1}, ar=ar)

    def build(self, context, actor1, actor2, filename):
        context_diag = self.parse(context)
        context_diag.draw(path=CONTEXT_IMG_PATH / filename)

        yes_question = get_ques_diag(nouns=True, ho=False)
        no_question = get_ques_diag(neg=True, nouns=True, ho=False)
        actor_pos = [
            find_wire(context_diag, actor1),
            find_wire(context_diag, actor2),
        ]

        # Connect the questions and context
        yes_circ = self.model_functor(yes_question).dagger()
        no_circ = self.model_functor(no_question).dagger()

        noun_wire = self.model_functor(self.n)

        discards = Id(Ty()).tensor(*[
            Discard(noun_wire) if i not in actor_pos else Id(noun_wire)
            for i in range(len(context_diag.cod))
        ])
        swaps = Swap(noun_wire, noun_wire) if actor_pos[0] < actor_pos[1] else Id(noun_wire @ noun_wire)
        context_circ = self.model_functor(context_diag) >> discards >> swaps

        if not context_circ.is_mixed:
            context_circ = context_circ @ (Ket(0) >> Discard(noun_wire))

        yes_circ = context_circ >> yes_circ
        no_circ = context_circ >> no_circ

        return yes_circ, no_circ

    def __call__(self, context, actor1, actor2):
        context_filename = f"context_{uuid4()}.png"
        metadata = get_inf_steps(context, actor1, actor2)

        ans = "Yes" if metadata["same_dir"] else "No"

        circuits = self.build(context, actor1, actor2, context_filename)
        overlaps = [c.eval().array.item().real for c in circuits]
        prob_yes = torch.softmax(torch.tensor(overlaps), dim=0)[0]
        result = "Yes" if prob_yes > 0.5 else "No"

        with open(CONTEXT_IMG_PATH / context_filename, "rb") as f:
            context_image = base64.b64encode(f.read())
        os.remove(CONTEXT_IMG_PATH / context_filename)

        return {
            "model": self.name,
            "context": context,
            "context_diag": context_image,
            "question": f"Is {actor1} going in the same direction as {actor2}?",
            "result": result,
            "target": ans,
            "correct": ans == result,
            "yes_prob": prob_yes.item(),
            "overlaps_real": overlaps,
            **metadata,
        }


MODEL_PATH = Path(os.path.dirname(__file__)).parent / "models"


class Showcase2dirModel(ShowcaseModel):
    name = "2Dir"
    boxes_path = MODEL_PATH / "2DIR_BOX_CIRCUITS.pkl"
    model_path = MODEL_PATH / "ext_following_model_taskextfoll3-2dir_QF-Binary-Statement-PostSelect0_"\
                              "TF-group-nouns_HO-quantum_HOQ-name-only_Sim4Following-3-1-mixed-dA_3.pkl"


class Showcase4dirModel(ShowcaseModel):
    name = "4Dir"
    boxes_path = MODEL_PATH / "4DIR_BOX_CIRCUITS.pkl"
    model_path = MODEL_PATH / "extfoll3_model_taskextfoll3-axioms-anc_QF-Binary-Statement-PostSelect0_"\
                              "TF-remove-the_HO-no-discards_HOQ-name-only_Sim4Ansatz-3-1-pure-dA_12.pkl"
