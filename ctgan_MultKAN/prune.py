import sys
import os
import torch
import pandas as pd
import yaml



current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from kan.MultKAN import MultKAN  


poda = "node"  # node/edge
arquitectura = "24-12"
estructura = "discriminator"
modelo_path = "/home/gtav-tft/Desktop/paula/eval_ctgan/COMP_TAMAÑOSRED/"+arquitectura+"/final_"+estructura+"_state_modelo.pt"
modelo_cache = "/home/gtav-tft/Desktop/paula/eval_ctgan/COMP_TAMAÑOSRED/"+arquitectura+"/final_"+estructura+"_cache_data_modelo.pt"
device= 'cuda'
state_dict = torch.load(modelo_path, map_location="cpu")

#ver parametros por terminal
for key, value in state_dict.items():
    print(f"{key}: {value.shape}")


def reconstruccion_checkpoint(modelo_path, device='cpu'):
    dic = torch.load(modelo_path, device)
    width = []
    i = 0
    while f'act_fun.{i}.coef' in dic:
        in_dim, out_dim, n_splines = dic[f'act_fun.{i}.coef'].shape
        if i == 0:
            width.append(in_dim)
        width.append(out_dim)
        i += 1


    print(f"Reconstruyendo MultKAN con dimensiones: {width}")
    multkan = MultKAN(width=width, device=device)
    multkan.load_state_dict(dic, strict=True)
    return multkan


from synthesizers.ctgan import Discriminator 
from synthesizers.ctgan import Generator

compress_dims = [24, 12] 
embedding_dim = 12

if estructura == "discriminator":
    modelo = Discriminator(input_dim=embedding_dim, discriminator_dim=compress_dims, pac=10).to(device)
elif estructura == "generator":
    modelo = Generator(embedding_dim=embedding_dim, generator_dim=compress_dims, data_dim=16).to(device)
else:
    raise ValueError("estructura debe ser 'generator' o 'discriminator'")

multkan = reconstruccion_checkpoint(modelo_path, device=device)
multkan.cache_data = torch.load(modelo_cache, map_location=device)
multkan.eval()


print("Establecemos activaciones antes de prune_node...")
acts = []
acts_premult = []
spline_preacts = []
spline_postacts = []
spline_postsplines = []
acts_scale = []
acts_scale_spline = []
subnode_actscale = []
edge_actscale = []


if estructura == "generator":
    modelo.generator_multkan = multkan
    net = modelo.generator_multkan
elif estructura == "discriminator":
    modelo.discriminator_multkan = multkan
    net = modelo.discriminator_multkan
else:
    raise ValueError("estructura no válida")

x = net.cache_data.to(device)

acts.append(x)




for l in range(net.depth):
    x_numerical, preacts, postacts_numerical, postspline = net.act_fun[l](x)

    if net.symbolic_enabled:
        x_symbolic, _ = net.symbolic_fun[l](x, singularity_avoiding=False, y_th=1000)
    else:
        x_symbolic = 0.

    x = x_numerical + x_symbolic
    subnode_actscale.append(torch.std(x, dim=0).detach())
    x = net.subnode_scale[l][None, :] * x + net.subnode_bias[l][None, :]

    input_range = torch.std(preacts, dim=0) + 0.1
    output_range = torch.std(postacts_numerical, dim=0)
    edge_actscale.append(output_range)
    acts_scale.append((output_range / input_range).detach())
    acts_scale_spline.append(output_range / input_range)
    spline_preacts.append(preacts.detach())
    spline_postacts.append(postacts_numerical.detach())
    spline_postsplines.append(postspline.detach())

    acts_premult.append(x.detach())
    acts.append(x.detach())


# Asignamos las activaciones restauradas
net.acts = acts
net.acts_premult = acts_premult
net.spline_preacts = spline_preacts
net.spline_postacts = spline_postacts
net.spline_postsplines = spline_postsplines
net.acts_scale = acts_scale
net.acts_scale_spline = acts_scale_spline
net.subnode_actscale = subnode_actscale
net.edge_actscale = edge_actscale

net.attribute()


if poda == "node":

    print(f"\nEjecutando poda en nodos del {estructura.upper()}...")
    print(f"Dimensiones antes de poda: {multkan.width}")
    multkan = multkan.prune_node(threshold=6e-2)
    print(f"Dimensiones después de poda: {multkan.width}")

elif poda == "edge":
    print(f"\nEjecutando poda en conexiones del {estructura.upper()}...")
    multkan.attribute()
    multkan = multkan.prune_edge()
else:
    raise ValueError("Valor de poda debe ser 'node' o 'edge'")


if estructura == "generator":
    modelo.generator_multkan =multkan
else:
    modelo.discriminator_multkan = multkan



print("Restaurando activaciones tras prune_node...")
acts = []
acts_premult = []
spline_preacts = []
spline_postacts = []
spline_postsplines = []
acts_scale = []
acts_scale_spline = []
subnode_actscale = []
edge_actscale = []


if estructura == "generator":
    net = modelo.generator_multkan
elif estructura == "discriminator":
    net = modelo.discriminator_multkan
else:
    raise ValueError("Estructura debe ser 'generator' o 'discriminator'")

x = net.cache_data.to('cuda')

acts.append(x)

for l in range(net.depth):
    x_numerical, preacts, postacts_numerical, postspline = net.act_fun[l](x)

    if net.symbolic_enabled:
        x_symbolic, _ = net.symbolic_fun[l](x, singularity_avoiding=False, y_th=1000)
    else:
        x_symbolic = 0.

    x = x_numerical + x_symbolic
    subnode_actscale.append(torch.std(x, dim=0).detach())
    x = net.subnode_scale[l][None, :] * x + net.subnode_bias[l][None, :]

    input_range = torch.std(preacts, dim=0) + 0.1
    output_range = torch.std(postacts_numerical, dim=0)
    edge_actscale.append(output_range)
    acts_scale.append((output_range / input_range).detach())
    acts_scale_spline.append(output_range / input_range)
    spline_preacts.append(preacts.detach())
    spline_postacts.append(postacts_numerical.detach())
    spline_postsplines.append(postspline.detach())

    acts_premult.append(x.detach())
    acts.append(x.detach())


# Asignamos las activaciones restauradas
net.acts = acts
net.acts_premult = acts_premult
net.spline_preacts = spline_preacts
net.spline_postacts = spline_postacts
net.spline_postsplines = spline_postsplines
net.acts_scale = acts_scale
net.acts_scale_spline = acts_scale_spline
net.subnode_actscale = subnode_actscale
net.edge_actscale = edge_actscale
net.attribute()

print(f"Activaciones restauradas correctamente en {estructura.upper()}.")


def save_multkan_model(model, path):
    """Guarda el modelo MultKAN en tres archivos: _state, _config.yml, _cache_data."""
    
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    config = dict(
        width = model.width,
        grid = model.grid,
        k = model.k,
        mult_arity = model.mult_arity,
        base_fun_name = model.base_fun_name,
        symbolic_enabled = model.symbolic_enabled,
        affine_trainable = model.affine_trainable,
        grid_eps = model.grid_eps,
        grid_range = model.grid_range,
        sp_trainable = model.sp_trainable,
        sb_trainable = model.sb_trainable,
        state_id = model.state_id,
        auto_save = model.auto_save,
        ckpt_path = model.ckpt_path,
        round = model.round,
        device = str(model.device)
    )

    for i in range(model.depth):
        config[f'symbolic.funs_name.{i}'] = model.symbolic_fun[i].funs_name

    with open(f'{path}_config.yml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    torch.save(model.state_dict(), f'{path}_state')

    if model.cache_data is not None:
        torch.save(model.cache_data, f'{path}_cache_data')

    print(f"Modelo guardado correctamente en: {path}_state, {path}_config.yml, {path}_cache_data")


print(f"\nGuardamos modelo tras poda en {poda} sobre {estructura.upper()}...")
#fichero = "./prune_result/"+estructura+"_pruned"
fichero = "/home/gtav-tft/Desktop/paula/eval_ctgan/resultado_poda_node_ctgan/24-12-0.10/"+estructura+"_pruned"
save_multkan_model(net, fichero)
