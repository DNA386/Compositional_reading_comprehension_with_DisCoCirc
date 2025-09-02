import torch

from lambeq.backend.quantum import Diagram as Circuit
from lambeq import Dataset, PytorchQuantumModel


class QADataset(Dataset):
    """Custom dataset class to play nice with pytorch tensors."""
    def __iter__(self):
        new_data, new_targets = self.data, self.targets

        if self.shuffle:
            new_data, new_targets = self.shuffle_data(new_data, new_targets)

        for start_idx in range(0, len(self.data), self.batch_size):
            yield (
                new_data[start_idx: start_idx+self.batch_size],
                torch.stack(new_targets[start_idx: start_idx+self.batch_size])
            )


class PytorchQAModel(PytorchQuantumModel):
    """Extend the default model to allow question answering via multiple circuits."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.softmax = torch.nn.Softmax(dim=1)

    @classmethod
    def from_diagrams(cls, diagrams, **kwargs):
        return super().from_diagrams(
            diagrams=[circ for circs in diagrams for circ in circs],
            **kwargs
        )

    def forward(self, x: list[tuple[Circuit, Circuit]]) -> torch.Tensor:
        """Input circuits are positive, negative question pairs. Combine these into a probability distribution."""
        outputs = []
        for pos, neg in x:
            output = self.get_diagram_output([pos, neg])
            outputs.append(output)
        outputs = torch.stack(outputs)
        outputs = self.softmax(outputs)
        return outputs
