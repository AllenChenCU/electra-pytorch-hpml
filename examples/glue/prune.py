from functools import partial
from torch.nn.utils import prune


def get_params_to_prune(model):
    embeddings_params_to_prune = [
        (model.electra.embeddings_project, 'weight'), 
        (model.electra.embeddings_project, 'bias'), 
    ]
    layers_params_to_prune = []
    for i in range(12):
        layers_params_to_prune.append([
            (model.electra.encoder.layer[i].attention.self.query, 'weight'),
            (model.electra.encoder.layer[i].attention.self.query, 'bias'), 
            (model.electra.encoder.layer[i].attention.self.key, 'weight'), 
            (model.electra.encoder.layer[i].attention.self.key, 'bias'), 
            (model.electra.encoder.layer[i].attention.self.value, 'weight'), 
            (model.electra.encoder.layer[i].attention.self.value, 'bias'), 
            (model.electra.encoder.layer[i].attention.output.dense, 'weight'), 
            (model.electra.encoder.layer[i].attention.output.dense, 'bias'), 
            (model.electra.encoder.layer[i].intermediate.dense, 'weight'), 
            (model.electra.encoder.layer[i].intermediate.dense, 'bias'), 
            (model.electra.encoder.layer[i].output.dense, 'weight'), 
            (model.electra.encoder.layer[i].output.dense, 'bias'), 
        ])
    layers_params_to_prune = [y for x in layers_params_to_prune for y in x]
    classifier_params_to_prune = [
        (model.classifier.dense, 'weight'), 
        (model.classifier.dense, 'bias'), 
        (model.classifier.out_proj, 'weight'), 
        (model.classifier.out_proj, 'bias'), 
    ]
    params_to_prune = embeddings_params_to_prune + layers_params_to_prune + classifier_params_to_prune
    return params_to_prune


def unstructured_prune_global(params_to_prune, pruning_method, amount=0.25):
    """Prune model globally inplace"""
    prune.global_unstructured(
        params_to_prune,
        pruning_method=pruning_method,
        amount=amount,
    )


def unstructured_prune_layer(params_to_prune, amount=0.25):
    """Prune layers in params_to_prune based on L1 norm criterion element-wise"""
    for mod, name in params_to_prune:
        prune.l1_unstructured(mod, name=name, amount=amount)


def structured_prune_layer(params_to_prune, amount=0.25, n=1, dim=0):
    """Prune layers in params_to_prune based on L-n norm criterion and vector along dim"""
    for mod, name in params_to_prune:
        if name == "weight":
            prune.ln_structured(mod, name=name, amount=amount, n=n, dim=dim)
        else:
            prune.l1_unstructured(mod, name=name, amount=amount)

#params_to_prune = get_params_to_prune(model)

#unstructured_prune_global(params_to_prune, prune.RandomUnstructured, amount=0.25)
#unstructured_prune_global(params_to_prune, prune.L1Unstructured, amount=0.25)
#unstructured_prune_layer(params_to_prune, amount=0.25)

#structured_prune_layer(params_to_prune, amount=0.25, n=1, dim=0)
#structured_prune_layer(params_to_prune, amount=0.25, n=2, dim=0)
#structured_prune_layer(params_to_prune, amount=0.25, n=1, dim=1)
#structured_prune_layer(params_to_prune, amount=0.25, n=2, dim=1)
