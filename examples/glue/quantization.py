import copy
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

class CustomElectraIntermediate(nn.Module):
    def __init__(self, electra_intermediate):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.electra_intermediate = electra_intermediate
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.quant(hidden_states)
        x = self.electra_intermediate(x)
        x = self.dequant(x)
        return x
    

class CustomElectraClassificationHead(nn.Module):
    def __init__(self, electra_classification_head):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.electra_classification_head = electra_classification_head

    def forward(self, features, **kwargs):
        x = self.quant(features)
        x = self.electra_classification_head(x)
        x = self.dequant(x)
        return x


def make_PTSQ_model(args, backend, model, train_dataloader):
    """Post-training Static Quantization for efficient inferencing
    model: this input is trained
    """

    #backend = 'qnnpack'#"fbgemm"
    ptsq_model = copy.deepcopy(model)
    ptsq_model.eval()

    # prepare (inserting observer modules to record the data for quantizing activations)
    torch.backends.quantized.engine = backend
    ptsq_model.qconfig = torch.quantization.get_default_qconfig(backend)

    # Insert/replace Custom layers
    # ptsq_model.electra.embeddings.LayerNorm = CustomLayerNorm(ptsq_model.electra.embeddings.LayerNorm)
    for i in range(12):
        ptsq_model.electra.encoder.layer[i].intermediate = CustomElectraIntermediate(ptsq_model.electra.encoder.layer[i].intermediate)
    ptsq_model.classifier = CustomElectraClassificationHead(ptsq_model.classifier)
    
    # Specify where not to quantize
    attention_names = [f"electra.encoder.layer.{x}.attention" for x in range(12)]
    output_names = [f"electra.encoder.layer.{x}.output" for x in range(12)]
    for name, mod in ptsq_model.named_modules():
        if name == "electra.embeddings": #isinstance(mod, torch.nn.Embedding):
            #mod.qconfig = torch.ao.quantization.float_qparams_weight_only_qconfig
            mod.qconfig = None
        if name in attention_names:
            mod.qconfig = None
        if name in output_names:
            mod.qconfig = None
        if name == "electra.embeddings_project":
            mod.qconfig = None

    torch.quantization.prepare(ptsq_model, inplace=True)

    # calibrate
    with torch.inference_mode():
        for batch in train_dataloader:
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            inputs["token_type_ids"] = (
                batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
            )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = ptsq_model(**inputs)
            loss = outputs[0]

    # convert
    torch.quantization.convert(ptsq_model, inplace=True)

    # check
    print(ptsq_model)
    return ptsq_model


def prepare_qat_model(backend, model):
    """Post-training Static Quantization for efficient inferencing
    model: this input is trained
    """

    #backend = 'qnnpack'#"fbgemm"
    qat_model = copy.deepcopy(model)
    qat_model.train()

    # prepare (inserting observer modules to record the data for quantizing activations)
    torch.backends.quantized.engine = backend
    qat_model.qconfig = torch.quantization.get_default_qconfig(backend)

    # Insert/replace Custom layers
    # ptsq_model.electra.embeddings.LayerNorm = CustomLayerNorm(ptsq_model.electra.embeddings.LayerNorm)
    for i in range(12):
        qat_model.electra.encoder.layer[i].intermediate = CustomElectraIntermediate(qat_model.electra.encoder.layer[i].intermediate)
    qat_model.classifier = CustomElectraClassificationHead(qat_model.classifier)
    
    # Specify where not to quantize
    attention_names = [f"electra.encoder.layer.{x}.attention" for x in range(12)]
    output_names = [f"electra.encoder.layer.{x}.output" for x in range(12)]
    for name, mod in qat_model.named_modules():
        if name == "electra.embeddings": #isinstance(mod, torch.nn.Embedding):
            #mod.qconfig = torch.ao.quantization.float_qparams_weight_only_qconfig
            mod.qconfig = None
        if name in attention_names:
            mod.qconfig = None
        if name in output_names:
            mod.qconfig = None
        if name == "electra.embeddings_project":
            mod.qconfig = None

    torch.quantization.prepare_qat(qat_model, inplace=True)

    return qat_model
