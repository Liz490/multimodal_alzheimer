import torch
import torchvision
import matplotlib.pyplot as plt
import io
import matplotlib.pyplot as plt
from PIL import Image


class IntHandler:
    """
    See https://stackoverflow.com/a/73388839
    """

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        text = plt.matplotlib.text.Text(x0, y0, str(orig_handle))
        handlebox.add_artist(text)
        return text


def generate_loggable_confusion_matrix(self, outs):
    """
    See https://stackoverflow.com/a/73388839
    """

    outputs = torch.cat([tmp['outputs'] for tmp in outs])
    labels = torch.cat([tmp['labels'] for tmp in outs])

    fig = confusion_matrix(outputs,
                           labels,
                           self.hparams['n_classes'],
                           self.self.label_ind_by_names)

    buf = io.BytesIO()

    fig.savefig(buf, format='jpeg', bbox_inches='tight')
    plt.close('all')
    buf.seek(0)
    with Image.open(buf) as im:
        return torchvision.transforms.ToTensor()(im)


def confusion_matrix(outputs: torch.Tensor,
                     labels: torch.Tensor,
                     n_classes: int,
                     label_idx_by_name: dict[str, int]) -> plt.figure.Figure:
    """
    Implemented in majority voting.
    TODO move it here
    """
    pass

