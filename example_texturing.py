import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import trimesh
from PIL import Image
from trellis2.pipelines import Trellis2TexturingPipeline

pipeline = Trellis2TexturingPipeline.from_pretrained("microsoft/TRELLIS.2-4B", config_file="texturing_pipeline.json")
pipeline.cuda()

mesh = trimesh.load("assets/example_texturing/the_forgotten_knight.ply")
image = Image.open("assets/example_texturing/image.webp")
output = pipeline.run(mesh, image)

output.export("textured.glb", extension_webp=True)