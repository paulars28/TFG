import sys
import os
import torch
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

from kan.MultKAN import MultKAN  

arquitectura = "10-5"
estructura = "decoder" #encoder/decoder
modelo_path = "/home/gtav-tft/Desktop/paula/eval/COMP_TAMAÑOSRED/"+arquitectura+"/final_"+estructura+"_stateHeartDisease"
modelo_cache = "/home/gtav-tft/Desktop/paula/eval/COMP_TAMAÑOSRED/"+arquitectura+"/final_"+estructura+"_cache_dataHeartDisease"
device= 'cuda'
device= 'cuda'

folder = "./symbolic_result"

data_dim = 9
compress_dims = [10, 5] 
embedding_dim = 5


from synthesizers.tvae import Decoder 
from synthesizers.tvae import Encoder
if estructura == "decoder":
    modelo = Decoder(embedding_dim, compress_dims, data_dim).to('cuda') 
    multkan = modelo.multkan_decoder
elif estructura == "encoder":
    modelo = Encoder(embedding_dim, compress_dims, data_dim).to('cuda')   
    multkan = modelo.multkan_encoder
else:
    raise ValueError("No se ha indicado correctamente la estructura a analizar: ENCODER/DECODER")
multkan.load_state_dict(torch.load(modelo_path, map_location=device), strict=True)
multkan.cache_data = torch.load(modelo_cache, map_location=device)
modelo.eval()


print("EStablecemos activaciones antes de prune_node...")
acts = []
acts_premult = []
spline_preacts = []
spline_postacts = []
spline_postsplines = []
acts_scale = []
acts_scale_spline = []
subnode_actscale = []
edge_actscale = []


if estructura == "encoder":
    net = modelo.multkan_encoder
elif estructura == "decoder":
    net = modelo.multkan_decoder
else:
    raise ValueError("Estructura debe ser 'encoder' o 'decoder'")

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



# Extraer y guardar fórmula simbólica
try:
    if estructura == "decoder":
        modelo.multkan_decoder.auto_symbolic()
        symbolic_formula = modelo.multkan_decoder.symbolic_formula()

    elif estructura == "encoder":
        modelo.multkan_encoder.auto_symbolic()
        symbolic_formula = modelo.multkan_encoder.symbolic_formula()

    with open("symbolic_formula_" + estructura + ".txt", "w") as f:
        f.write(estructura + " SYMBOLIC FORMULA:\n")
        f.write("\n".join([str(eq) for eq in symbolic_formula]))
    print("Fórmula simbólica final guardada en symbolic_formula.txt")



except Exception as e:
    print(f"Fallo al extraer la función simbólica: {e}")








