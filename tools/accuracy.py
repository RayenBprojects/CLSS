import torch


def _valid_points(
    predicted_Y: torch.Tensor,
    ground_truth_Y: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> torch.Tensor:
    if predicted_Y.shape != ground_truth_Y.shape:
        raise ValueError("Predictions and ground truth must have the same shape")
    if valid_mask is None:
        return torch.ones_like(ground_truth_Y, dtype=torch.bool)
    if valid_mask.shape != ground_truth_Y.shape:
        raise ValueError("The validity mask must match the label shape")
    return valid_mask.to(device=ground_truth_Y.device, dtype=torch.bool)


def _class_counts(
    predicted_Y: torch.Tensor,
    ground_truth_Y: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted_flat = predicted_Y[valid_mask].reshape(-1)
    ground_truth_flat = ground_truth_Y[valid_mask].reshape(-1)
    predicted_by_class = predicted_flat[:, None] == labels
    ground_truth_by_class = ground_truth_flat[:, None] == labels

    intersection = (predicted_by_class & ground_truth_by_class).sum(dim=0)
    ground_truth_count = ground_truth_by_class.sum(dim=0)
    union = (predicted_by_class | ground_truth_by_class).sum(dim=0)
    return intersection, ground_truth_count, union


def _mean_present_classes(
    intersection: torch.Tensor | None, denominator: torch.Tensor | None
) -> float:
    if intersection is None or denominator is None:
        raise ValueError("No metric counts have been accumulated")
    present = denominator > 0
    if not present.any():
        raise ValueError("No valid classes are available to calculate the metric")
    return float((intersection[present] / denominator[present]).mean())


class OAcc:
    def __init__(self):
        self._sum_correct = None
        self._sum_total = None

    def calculcate_oAcc(
        self,
        predicted_Y: torch.Tensor,
        ground_truth_Y: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        mask = _valid_points(predicted_Y, ground_truth_Y, valid_mask)
        predicted_flat = predicted_Y[mask]
        ground_truth_flat = ground_truth_Y[mask]
        correct = (predicted_flat == ground_truth_flat).sum()
        total = predicted_flat.numel()

        self._sum_correct = (
            self._sum_correct + correct
            if self._sum_correct is not None
            else correct
        )
        self._sum_total = (
            self._sum_total + total if self._sum_total is not None else total
        )

    def get_oAcc(self) -> float:
        if not self._sum_total:
            raise ValueError("No valid points are available to calculate accuracy")
        return float(self._sum_correct / self._sum_total)


class MAcc:
    def __init__(self):
        self._intersection_sum = None
        self._ground_truth_sum = None
        self._labels = None

    def calculate_mAcc(
        self,
        predicted_Y: torch.Tensor,
        ground_truth_Y: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        if self._labels is None:
            self._labels = labels

        mask = _valid_points(predicted_Y, ground_truth_Y, valid_mask)
        intersection, ground_truth_count, _ = _class_counts(
            predicted_Y, ground_truth_Y, self._labels, mask
        )
        self._intersection_sum = (
            self._intersection_sum + intersection
            if self._intersection_sum is not None
            else intersection
        )
        self._ground_truth_sum = (
            self._ground_truth_sum + ground_truth_count
            if self._ground_truth_sum is not None
            else ground_truth_count
        )

    def get_mAcc(self) -> float:
        return _mean_present_classes(self._intersection_sum, self._ground_truth_sum)


class MIoU:
    def __init__(self):
        self._intersection_sum = None
        self._union_sum = None
        self._labels = None

    def calculate_mIoU(
        self,
        predicted_Y: torch.Tensor,
        ground_truth_Y: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        if self._labels is None:
            self._labels = labels

        mask = _valid_points(predicted_Y, ground_truth_Y, valid_mask)
        intersection, _, union = _class_counts(
            predicted_Y, ground_truth_Y, self._labels, mask
        )
        self._intersection_sum = (
            self._intersection_sum + intersection
            if self._intersection_sum is not None
            else intersection
        )
        self._union_sum = (
            self._union_sum + union if self._union_sum is not None else union
        )

    def get_mIoU(self) -> float:
        return _mean_present_classes(self._intersection_sum, self._union_sum)


def get_oAcc(
    predicted_Y: torch.Tensor,
    ground_truth_Y: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> float:
    metric = OAcc()
    metric.calculcate_oAcc(predicted_Y, ground_truth_Y, valid_mask)
    return metric.get_oAcc()


def get_mAcc(
    predicted_Y: torch.Tensor,
    ground_truth_Y: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> float:
    metric = MAcc()
    metric.calculate_mAcc(predicted_Y, ground_truth_Y, labels, valid_mask)
    return metric.get_mAcc()


def get_mIoU(
    predicted_Y: torch.Tensor,
    ground_truth_Y: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> float:
    metric = MIoU()
    metric.calculate_mIoU(predicted_Y, ground_truth_Y, labels, valid_mask)
    return metric.get_mIoU()
