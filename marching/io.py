import numpy as np
from numba import njit


def read_polygon_ply(path):
    with open(path, "rb") as f:
        while (line := f.readline().decode()) != "end_header\n":
            if line.startswith("element vertex"):
                num_vertices = int(line[15:-1])
            elif line.startswith("property") and line.endswith(" x\n"):
                vertex_dtype = np.float32 if line[9:-3] == "float" else np.float64
            elif line.startswith("element face"):
                num_faces = int(line[13:-1])
        vertices = np.fromfile(f, dtype=vertex_dtype, count=num_vertices * 3).reshape(num_vertices, 3)
        faces, face_sizes = _read_polygon_ply_faces(np.fromfile(f, dtype=np.uint8), num_faces)
        return vertices, faces, face_sizes


def write_polygon_ply(path, vertices, faces, face_sizes):
    if (vertices.dtype != np.float64 or
            faces.dtype not in (np.uint32, np.int32) or
            face_sizes.dtype not in (np.uint8, np.int8)):
        raise ValueError("Some arrays have the wrong dtype.")
    with open(path, "wb") as f:
        f.write(f"""ply
format binary_little_endian 1.0
element vertex {len(vertices)}
property double x
property double y
property double z
element face {len(face_sizes)}
property list uchar uint vertex_indices
end_header
""".encode())
        f.write(vertices)
        buf = np.empty(len(face_sizes) + int(face_sizes.sum()) * 4, dtype=np.uint8)
        _write_polygon_ply_faces(buf, faces, face_sizes)
        f.write(buf)


@njit
def _read_polygon_ply_faces(buf, num_faces):
    # Note: This method assumes that the vertex_index property list is of type "uchar uint".
    faces = np.empty((len(buf) - num_faces) // 4, dtype=np.uint32)
    face_sizes = np.empty(num_faces, dtype=np.uint32)
    b = 0
    f = 0
    for p in range(num_faces):
        face_size = buf[b]
        b += 1
        face_sizes[p] = face_size
        faces[f:f + face_size] = buf[b:b + face_size * 4].view(np.uint32)
        b += face_size * 4
        f += face_size
    return faces, face_sizes


@njit
def _write_polygon_ply_faces(buf, faces, face_sizes):
    f = 0
    b = 0
    for face_size in face_sizes:
        buf[b] = face_size
        b += 1
        buf[b:b + face_size * 4].view(np.uint32)[:] = faces[f:f + face_size]
        f += face_size
        b += face_size * 4


