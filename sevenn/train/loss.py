from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

import sevenn._keys as KEY


class LossDefinition:
    """
    Base class for loss definition
    weights are defined in outside of the class
    """

    def __init__(
        self,
        name: str,
        unit: Optional[str] = None,
        criterion: Optional[Callable] = None,
        ref_key: Optional[str] = None,
        pred_key: Optional[str] = None,
        use_weight: bool = False,
        ignore_unlabeled: bool = True,
    ) -> None:
        self.name = name
        self.unit = unit
        self.criterion = criterion
        self.ref_key = ref_key
        self.pred_key = pred_key
        self.use_weight = use_weight
        self.ignore_unlabeled = ignore_unlabeled

    def __repr__(self):
        return self.name

    def assign_criteria(self, criterion: Callable) -> None:
        if self.criterion is not None:
            raise ValueError('Loss uses its own criterion.')
        self.criterion = criterion

    def _preprocess(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.pred_key is None or self.ref_key is None:
            raise NotImplementedError('LossDefinition is not implemented.')
        pred = torch.reshape(batch_data[self.pred_key], (-1,))
        ref = torch.reshape(batch_data[self.ref_key], (-1,))
        return pred, ref, None

    def _ignore_unlabeled(
        self,
        pred: torch.Tensor,
        ref: torch.Tensor,
        data_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        unlabeled = torch.isnan(ref)
        pred = pred[~unlabeled]
        ref = ref[~unlabeled]
        if data_weights is not None:
            data_weights = data_weights[~unlabeled]
        return pred, ref, data_weights

    def get_loss(self, batch_data: Dict[str, Any], model: Optional[Callable] = None):
        """
        Function that return scalar
        """
        if self.criterion is None:
            raise NotImplementedError('LossDefinition has no criterion.')
        pred, ref, w_tensor = self._preprocess(batch_data, model)

        if self.ignore_unlabeled:
            pred, ref, w_tensor = self._ignore_unlabeled(pred, ref, w_tensor)

        if len(pred) == 0:
            assert self.ref_key is not None
            return torch.zeros(1, device=batch_data[self.ref_key].device)

        loss = self.criterion(pred, ref)
        if self.use_weight:
            loss = torch.mean(loss * w_tensor)
        return loss


class PerAtomEnergyLoss(LossDefinition):
    """
    Loss for per atom energy
    """

    def __init__(
        self,
        name: str = 'Energy',
        unit: str = 'eV/atom',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.ENERGY,
        pred_key: str = KEY.PRED_TOTAL_ENERGY,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key,
            **kwargs,
        )

    def _preprocess(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        num_atoms = batch_data[KEY.NUM_ATOMS]
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)
        pred = batch_data[self.pred_key] / num_atoms
        ref = batch_data[self.ref_key] / num_atoms
        w_tensor = None

        if self.use_weight:
            loss_type = self.name.lower()
            weight = batch_data[KEY.DATA_WEIGHT][loss_type]
            w_tensor = torch.repeat_interleave(weight, 1)

        return pred, ref, w_tensor


class ForceLoss(LossDefinition):
    """
    Loss for force
    """

    def __init__(
        self,
        name: str = 'Force',
        unit: str = 'eV/A',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.FORCE,
        pred_key: str = KEY.PRED_FORCE,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key,
            **kwargs,
        )

    def _preprocess(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)
        pred = torch.reshape(batch_data[self.pred_key], (-1,))
        ref = torch.reshape(batch_data[self.ref_key], (-1,))
        w_tensor = None

        if self.use_weight:
            loss_type = self.name.lower()
            weight = batch_data[KEY.DATA_WEIGHT][loss_type]
            w_tensor = weight[batch_data[KEY.BATCH]]
            w_tensor = torch.repeat_interleave(w_tensor, 3)

        return pred, ref, w_tensor


class StressLoss(LossDefinition):
    """
    Loss for stress this is kbar
    """

    def __init__(
        self,
        name: str = 'Stress',
        unit: str = 'kbar',
        criterion: Optional[Callable] = None,
        ref_key: str = KEY.STRESS,
        pred_key: str = KEY.PRED_STRESS,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            unit=unit,
            criterion=criterion,
            ref_key=ref_key,
            pred_key=pred_key,
            **kwargs,
        )
        self.TO_KB = 1602.1766208  # eV/A^3 to kbar

    def _preprocess(
        self, batch_data: Dict[str, Any], model: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        assert isinstance(self.pred_key, str) and isinstance(self.ref_key, str)

        pred = torch.reshape(batch_data[self.pred_key] * self.TO_KB, (-1,))
        ref = torch.reshape(batch_data[self.ref_key] * self.TO_KB, (-1,))
        w_tensor = None

        if self.use_weight:
            loss_type = self.name.lower()
            weight = batch_data[KEY.DATA_WEIGHT][loss_type]
            w_tensor = torch.repeat_interleave(weight, 6)

        return pred, ref, w_tensor


class EWCLoss(LossDefinition):
    """
    Elastic Weight Consolidation Loss.
    Penalizes changes to important weights based on Fisher information matrix.
    """

    def __init__(
        self,
        fisher_dict: Dict[str, torch.Tensor],
        opt_params_dict: Dict[str, torch.Tensor],
        name: str = 'EWC',
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name=name,
            criterion=None,
            ref_key=None,
            pred_key=None,
            use_weight=False,
            ignore_unlabeled=False,
        )
        self.fisher_dict = fisher_dict
        self.opt_params_dict = opt_params_dict
        self.device = device
        if device is not None:
            self.to(device)

    def to(self, device) -> None:
        for k in self.fisher_dict:
            self.fisher_dict[k] = self.fisher_dict[k].to(device)
        for k in self.opt_params_dict:
            self.opt_params_dict[k] = self.opt_params_dict[k].to(device)
        self.device = device

    def get_loss(
        self,
        batch_data: Dict[str, Any],
        model: Optional[Callable] = None
    ) -> torch.Tensor:
        _ = batch_data
        if model is None:
            raise ValueError('EWC requires model to compute loss')
        ewc_loss = torch.tensor([0.0], device=self.device)
        for name, _param in model.named_parameters():
            if name not in self.fisher_dict or name not in self.opt_params_dict:
                continue
            fisher = self.fisher_dict[name]
            opt_param = self.opt_params_dict[name]
            ewc_loss += torch.sum(fisher * (_param - opt_param) ** 2)
        return ewc_loss


def get_loss_functions_from_config(
    config: Dict[str, Any],
) -> List[Tuple[LossDefinition, float]]:
    from sevenn.train.optim import loss_dict

    loss_functions = []  # list of tuples (loss_definition, weight)

    loss = loss_dict[config[KEY.LOSS].lower()]
    loss_param = config.get(KEY.LOSS_PARAM, {})

    use_weight = config.get(KEY.USE_WEIGHT, False)
    if use_weight:
        loss_param['reduction'] = 'none'
    criterion = loss(**loss_param)

    commons = {'use_weight': use_weight}

    loss_functions.append((PerAtomEnergyLoss(**commons), 1.0))
    loss_functions.append((ForceLoss(**commons), config[KEY.FORCE_WEIGHT]))
    if config[KEY.IS_TRAIN_STRESS]:
        loss_functions.append((StressLoss(**commons), config[KEY.STRESS_WEIGHT]))

    for loss_function, _ in loss_functions:
        if loss_function.criterion is None:
            loss_function.assign_criteria(criterion)

    # Add EWC loss if fisher information and optimal params are provided
    fisher_information_path = config.get(KEY.CONTINUE, {}).get(KEY.FISHER, False)
    optimal_params_path = config.get(KEY.CONTINUE, {}).get(KEY.OPT_PARAMS, False)
    
    if fisher_information_path and optimal_params_path:
        fisher_dict = torch.load(fisher_information_path, weights_only=True)
        opt_params_dict = torch.load(optimal_params_path, weights_only=True)
        ewc_lambda = float(config.get(KEY.CONTINUE, {}).get(KEY.EWC_LAMBDA, 0))
        device = config.get(KEY.DEVICE, 'cpu')
        if ewc_lambda > 0:
            loss_functions.append(
                (EWCLoss(fisher_dict, opt_params_dict, device=device), ewc_lambda / 2.0)
            )

    return loss_functions

