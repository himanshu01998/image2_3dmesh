import torch
import PIL.Image
import trimesh  # You might need to install this: pip install trimesh
import zipfile
import os
from diffusers import DiffusionPipeline
from diffusers.utils import load_image

def generate_3d_from_image(image_path: str, output_obj_path: str, output_zip_path: str):
    """
    Generates a 3D mesh from an image, saves it as an OBJ file, and then zips it.
    """
    # 1. Set up the device and load the pipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo = "openai/shap-e-img2img"
    
    print("Loading the pipeline...")
    # Use float16 for faster inference and less memory usage on CUDA devices
    pipe = DiffusionPipeline.from_pretrained(repo, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    print("Pipeline loaded.")

    # 2. Load the input image
    print(f"Loading input image from: {image_path}")
    image = load_image(image_path).convert("RGB")

    # 3. Run inference to generate the 3D mesh
    # The pipeline can directly output a mesh object.
    print("Generating 3D mesh... (This may take a while)")
    output = pipe(
        image,
        guidance_scale=3.0,
        num_inference_steps=64,
        output_type="mesh", # This is key to get a 3D mesh object
    )
    
    # The output is a list of meshes
    mesh_result = output.images[0]
    print("Mesh generated successfully.")

    # 4. Save the mesh to a file
    # The output from the Shap-E renderer is a trimesh.Trimesh object,
    # which can be exported to various formats.
    print(f"Saving mesh to {output_obj_path}...")
    try:
        # The mesh object has an export method
        mesh_result.export(output_obj_path)
        print("Mesh saved.")
    except Exception as e:
        print(f"Could not save the mesh directly. Error: {e}")
        print("You may need to install additional dependencies for trimesh like 'pyglet'.")
        return

    # 5. Compress the file into a zip archive
    print(f"Compressing {output_obj_path} into {output_zip_path}...")
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_obj_path, os.path.basename(output_obj_path))
    print("Zipping complete.")
    
    # Optional: Clean up the intermediate .obj file
    os.remove(output_obj_path)
    print(f"Cleaned up {output_obj_path}.")


if __name__ == "__main__":
    # Make sure to have an image file named 'corgi.png' in the same directory
    # or provide a different path.
    input_image = input("Enter image path:") # You can download this from: https://hf.co/datasets/diffusers/docs-images/resolve/main/shap-e/corgi.png
    if not os.path.exists(input_image):
        print(f"Error: Input image '{input_image}' not found.")
        print("Please download it or use a different image.")
    else:
        output_path = input_image.rsplit('.', 1)[0]  # Remove the file extension for output names
        generate_3d_from_image(
            image_path=input_image,
            output_obj_path="corgi_3d.obj",
            output_zip_path=f"results/{output_path}.zip"
        )
