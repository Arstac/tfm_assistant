from .load_config import load_config
 
config = load_config()

print(config)
path_extraccion_datos = config["prompts"]["path_prompt_extraccion_datos"]
path_extraccion_materiales = config["prompts"]["path_prompt_extraccion_materiales"]
path_extraccion_precios = config["prompts"]["path_prompt_extraccion_precios"]

with open(path_extraccion_datos, "r") as file:
    prompt_extraccion_datos = file.read()
    
with open(path_extraccion_materiales, "r") as file:
    prompt_extraccion_materiales = file.read()
    
with open(path_extraccion_precios, "r") as file:
    prompt_extraccion_precios = file.read()