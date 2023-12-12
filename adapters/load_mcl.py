import logging

from transformers.models.bert.modeling_bert import BertSelfOutput
from torch import nn

from .adapters import LoRALinear


logger = logging.getLogger(__name__)

ATTENTION_LINEARS_T5 = ["k", "v", "q", "o"]
ATTENTION_LINEARS = ATTENTION_LINEARS_T5


def replace_layers(model, adapter_class, ac_kwargs):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_layers(module, adapter_class, ac_kwargs)

        if isinstance(module, nn.Linear):
            if name in ATTENTION_LINEARS:
                new_linear = adapter_class(module.weight, module.bias, **ac_kwargs)
                setattr(model, name, new_linear)

        if isinstance(model, BertSelfOutput):
            if name == "dense":
                new_linear = adapter_class(module.weight, module.bias, **ac_kwargs)
                setattr(model, name, new_linear)


def update_weights(model, adapter_class):
    for module in model.children():
        if len(list(module.children())) > 0:
            update_weights(module, adapter_class)
        if isinstance(module, adapter_class):
            module.update()


class ModularMixin(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        freeze: bool,
        ac_kwargs: dict,
    ):
        super().__init__()
        self.model = model
        self.adapter_class = LoRALinear
        self.ac_kwargs = ac_kwargs

        if freeze:
            for n, p in self.model.named_parameters():
                if "classifier" in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        replace_layers(
            model=self.model,
            adapter_class=LoRALinear,
            ac_kwargs=ac_kwargs,
        )

    def update_weights(self):
        update_weights(self.model, self.adapter_class)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def forward(self, *args, **kwargs):
        outputs = self.model.forward(*args, **kwargs)
        return outputs
