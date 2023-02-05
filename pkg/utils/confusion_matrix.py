import torch
import torchmetrics
import torchvision
import matplotlib.pyplot as plt
import io
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import seaborn as sns
import pandas as pd


class IntHandler:
    """
    See https://stackoverflow.com/a/73388839
    """

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        text = plt.matplotlib.text.Text(x0, y0, str(orig_handle))
        handlebox.add_artist(text)
        return text


def generate_loggable_confusion_matrix(
        outs,
        label_idx_by_name: dict['str', 'int'],
        normalize: bool = False,
        legend: bool = True,
        colormap: bool = False):
    """
    See https://stackoverflow.com/a/73388839
    """

    outputs = torch.cat([tmp['outputs'] for tmp in outs])
    labels = torch.cat([tmp['labels'] for tmp in outs])

    fig = confusion_matrix(
        outputs, labels, label_idx_by_name, normalize, legend, colormap)

    buf = io.BytesIO()

    fig.savefig(buf, format='jpeg', bbox_inches='tight')
    plt.close('all')
    buf.seek(0)
    with Image.open(buf) as im:
        return torchvision.transforms.ToTensor()(im)


def generate_confusion_matrix(
        outs,
        label_idx_by_name: dict['str', 'int'],
        normalize: bool = False,
        legend: bool = True,
        colormap: bool = False):
    """
    See https://stackoverflow.com/a/73388839
    """

    outputs = torch.cat([tmp['outputs'] for tmp in outs])
    labels = torch.cat([tmp['labels'] for tmp in outs])

    fig = confusion_matrix(
        outputs, labels, label_idx_by_name, normalize, legend, colormap)
    return fig


def confusion_matrix(outputs: torch.Tensor,
                     labels: torch.Tensor,
                     label_idx_by_name: dict[str, int],
                     normalize: bool = False,
                     legend: bool = True,
                     colormap: bool = False
                     ) -> Figure:
    """
    Implemented in majority voting.
    """
    n_classes = len(label_idx_by_name)
    normalize = 'true' if normalize else None
    task = "binary" if n_classes == 2 else "multiclass"
    confusion = torchmetrics.ConfusionMatrix(
        num_classes=n_classes, task=task, normalize=normalize
    ).to(outputs.get_device())
    outputs = torch.max(outputs, dim=1)[1]
    confusion(outputs, labels)
    computed_confusion = confusion.compute().detach().cpu().numpy()

    # confusion matrix
    df_cm = pd.DataFrame(
        computed_confusion,
        index=label_idx_by_name.values(),
        columns=label_idx_by_name.values(),
    )

    if not legend:
        df_cm.index = label_idx_by_name.keys()
        df_cm.columns = label_idx_by_name.keys()

    if colormap:
        cmap = LinearSegmentedColormap.from_list(
            'mycmap', ['#b0cffb', '#22418e'])
    else:
        cmap = 'crest'

    if legend:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.subplots_adjust(left=0.05, right=.65)
    else:
        fig, ax = plt.subplots()
    sns.set(font_scale=1.2)
    if normalize:
        sns.heatmap(
            df_cm, annot=True, annot_kws={"size": 16},
            fmt='.2f', ax=ax, cmap=cmap, vmin=0, vmax=1)
    else:
        sns.heatmap(df_cm, annot=True, annot_kws={
                    "size": 16}, fmt='d', ax=ax, cmap=cmap)

    plt.yticks(rotation=0)

    if legend:
        ax.legend(
            label_idx_by_name.values(),
            label_idx_by_name.keys(),
            handler_map={int: IntHandler()},
            loc='upper left',
            bbox_to_anchor=(1.2, 1)
        )

    return fig
