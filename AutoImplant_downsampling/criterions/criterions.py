from criterions.diceloss import DiceLoss
from criterions.focalloss import FocalLoss


def criterion_selector(criterion_name):
    avail_criterion={
        "DiceLoss": DiceLoss(),
        "FocalLoss":FocalLoss()
    }

    try:
        criterion_class = avail_criterion[criterion_name]
    except KeyError:
        raise KeyError("Criterion {} does not exist in available dataset".format(criterion_name))

    return criterion_class