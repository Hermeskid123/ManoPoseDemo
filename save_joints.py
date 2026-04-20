import json
import torch
from smplx import MANO

mano_model_path = "./mano/models"
hand_side = "right"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MANO(
    model_path=mano_model_path,
    is_rhand=(hand_side == "right"),
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

joints = output.joints[0].detach().cpu().numpy().tolist()

data = {
    "joints": joints
}

with open("mano_joints.json", "w") as f:
    json.dump(data, f, indent=2)

print("Saved mano_joints.json")
