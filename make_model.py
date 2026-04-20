import bpy
import json
import mathutils
from math import acos

json_path = "mano_joints.json"
with open(json_path, "r") as f:
    data = json.load(f)

joints = data["joints"]
print("Number of joints:", len(joints))

if len(joints) == 21:
    mano_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20)
    ]
elif len(joints) == 16:
    mano_connections = [
        (0, 1), (1, 2), (2, 3),
        (0, 4), (4, 5), (5, 6),
        (0, 7), (7, 8), (8, 9),
        (0, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15)
    ]
else:
    raise ValueError(f"Unsupported joint count: {len(joints)}")

for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

def create_joint(location, name, radius=0.005):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location)
    obj = bpy.context.object
    obj.name = name
    return obj

def create_bone_mesh(start, end, name, radius=0.0025):
    start = mathutils.Vector(start)
    end = mathutils.Vector(end)
    diff = end - start
    length = diff.length
    mid = (start + end) / 2

    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=length, location=mid)
    obj = bpy.context.object
    obj.name = name

    up = mathutils.Vector((0, 0, 1))
    direction = diff.normalized()

    if (direction - up).length < 1e-6:
        quat = mathutils.Quaternion()
    elif (direction + up).length < 1e-6:
        quat = mathutils.Quaternion((1, 0, 0), 3.141592653589793)
    else:
        axis = up.cross(direction).normalized()
        angle = acos(max(-1.0, min(1.0, up.dot(direction))))
        quat = mathutils.Quaternion(axis, angle)

    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = quat
    return obj

for i, joint in enumerate(joints):
    create_joint(joint, f"joint_{i}")

for a, b in mano_connections:
    if a < len(joints) and b < len(joints):
        create_bone_mesh(joints[a], joints[b], f"bone_{a}_{b}")
    else:
        print(f"Skipping invalid connection ({a}, {b})")

print("Hand model created successfully.")
