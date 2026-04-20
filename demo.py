import os
import pickle
import torch
import numpy as np
from smplx import MANO

mano_model_path = "./mano/models"
num_samples = 10
hand_side = "right"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MANO(
    model_path=mano_model_path,
    is_rhand=(hand_side == "right"),
    use_pca=False,
    flat_hand_mean=False,
    batch_size=num_samples
).to(device)

global_orient = torch.randn(num_samples, 3, device=device) * 0.3
hand_pose = torch.randn(num_samples, 45, device=device) * 0.5
betas = torch.randn(num_samples, 10, device=device) * 0.03
transl = torch.randn(num_samples, 3, device=device) * 0.02

output = model(
    global_orient=global_orient,
    hand_pose=hand_pose,
    betas=betas,
    transl=transl,
    return_verts=True,
    return_full_pose=True
)

joints = output.joints.detach().cpu().numpy()
verts = output.vertices.detach().cpu().numpy()

print("Joints shape:", joints.shape)
print("Verts shape:", verts.shape)

save_data = {
    "joints": joints,
    "verts": verts,
    "global_orient": global_orient.detach().cpu().numpy(),
    "hand_pose": hand_pose.detach().cpu().numpy(),
    "betas": betas.detach().cpu().numpy(),
    "transl": transl.detach().cpu().numpy()
}

with open("random_mano_samples.pkl", "wb") as f:
    pickle.dump(save_data, f)

np.save("random_mano_joints.npy", joints)
print("Saved to random_mano_samples.pkl and random_mano_joints.npy")
