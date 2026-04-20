import os
import torch
from smplx import MANO

mano_model_path = "./mano/models"
output_obj_path = "random_mano_hand.obj"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MANO(
    model_path=mano_model_path,
    is_rhand=True,
    use_pca=False,
    flat_hand_mean=False,
    batch_size=1
).to(device)

global_orient = torch.randn(1, 3, device=device) * 0.3
hand_pose = torch.randn(1, 45, device=device) * 0.5
betas = torch.randn(1, 10, device=device) * 0.03
transl = torch.zeros(1, 3, device=device)

output = model(
    global_orient=global_orient,
    hand_pose=hand_pose,
    betas=betas,
    transl=transl,
    return_verts=True
)

verts = output.vertices[0].detach().cpu().numpy()
faces = model.faces

with open(output_obj_path, "w") as f:
    for v in verts:
        f.write(f"v {v[0]} {v[1]} {v[2]}\n")
    for face in faces:
        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

print(f"Saved OBJ to {output_obj_path}")
