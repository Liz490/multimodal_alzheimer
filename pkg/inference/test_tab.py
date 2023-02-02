from pkg.models.tabular_models.tabular_pl_wrapper import Tabular_Model
from pkg.utils.dataloader import MultiModalDataset
from pkg.utils.test import test
from pkg.utils.load_path_config import load_path_config


def tab_testset(paths: dict, binary_classification: bool = True):
    testset = MultiModalDataset(
        path=paths['test_set_csv'],
        modalities=['tabular'],
        binary_classification=binary_classification)
    return testset


def tab_model(paths: dict, binary_classification: bool = True):
    if binary_classification:
        model = Tabular_Model.load_from_checkpoint(
            paths['tab_2_class']
        )
    else:
        model = Tabular_Model.load_from_checkpoint(
            paths['tab_3_class']
        )
    return model


if __name__ == '__main__':
    paths = load_path_config()

    # Two class
    testset = tab_testset(paths, binary_classification=True)
    model = tab_model(paths, binary_classification=True)
    experiment_name = 'test_set_tab_2_class'
    test(testset, model, experiment_name)

    # Three class
    testset = tab_testset(paths, binary_classification=False)
    model = tab_model(paths, binary_classification=False)
    experiment_name = 'test_set_tab_3_class'
    test(testset, model, experiment_name)
