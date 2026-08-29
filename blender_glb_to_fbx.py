import sys
import os
import bpy


def _log(msg):
    print(f"[BlenderFBX] {msg}", flush=True)


def _get_blender_version():
    """Return (major, minor, patch) tuple, e.g. (4, 1, 0)."""
    v = bpy.app.version
    return (v[0], v[1], v[2])


def _dump_mesh_diagnostics(label, objs):
    """Print detailed mesh/material/texture diagnostics."""
    mesh_objs = [o for o in objs if o.type == 'MESH']
    total_verts = sum(len(o.data.vertices) for o in mesh_objs)
    total_faces = sum(len(o.data.polygons) for o in mesh_objs)
    total_mats = 0
    total_uvs = 0
    for o in mesh_objs:
        total_mats += len(o.data.materials)
        total_uvs += len(o.data.uv_layers)
    _log(f"  [{label}] meshes={len(mesh_objs)} verts={total_verts} faces={total_faces} "
         f"materials={total_mats} uv_layers={total_uvs}")


def _ensure_materials(mesh_obj):
    """Walk through material slots and log what we see."""
    for i, slot in enumerate(mesh_obj.material_slots):
        mat = slot.material
        if mat is None:
            _log(f"    slot[{i}] = NONE (empty)")
            continue
        _log(f"    slot[{i}] name='{mat.name}' use_nodes={mat.use_nodes}")
        if mat.use_nodes and mat.node_tree:
            for node in mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    _log(f"      texture node '{node.name}' -> image '{node.image.name}' "
                         f"size={node.image.size[0]}x{node.image.size[1]}")


def _force_smooth_shading(objs):
    """Set all mesh objects to smooth shading (preserves vertex normals from GLB)."""
    for o in objs:
        if o.type != 'MESH':
            continue
        for poly in o.data.polygons:
            poly.use_smooth = True
        _log(f"  Set smooth shading on '{o.name}'")


def _ensure_split_normals(objs):
    """
    Preserve custom split normals from the GLB import.
    Blender's glTF importer stores them; we make sure they are not overridden.

    Compatibility:
      - Blender < 4.1: use_mesh.use_auto_smooth + auto_smooth_angle
      - Blender >= 4.1: "Smooth by Angle" modifier (auto_smooth removed)
    """
    major, minor, _ = _get_blender_version()
    use_modifier = (major, minor) >= (4, 1)

    for o in objs:
        if o.type != 'MESH':
            continue
        if hasattr(o.data, 'has_custom_normals') and o.data.has_custom_normals:
            _log(f"  '{o.name}' has custom split normals — preserved")
        elif use_modifier:
            # Blender 4.1+: add "Smooth by Angle" modifier
            mod = o.modifiers.new(name="Smooth by Angle", type='NODES')
            if mod and mod.node_group:
                # Set angle to 180° so everything is smooth
                for inp in mod.node_group.inputs:
                    if inp.name == 'Angle':
                        inp.default_value = 3.14159
            _log(f"  '{o.name}' using Smooth by Angle modifier (Blender 4.1+)")
        else:
            # Blender < 4.1: use legacy auto_smooth
            try:
                o.data.use_auto_smooth = True
                o.data.auto_smooth_angle = 3.14159
                _log(f"  '{o.name}' using auto_smooth for normals")
            except AttributeError:
                _log(f"  '{o.name}' could not set auto_smooth — skipping")


def _build_fbx_export_kwargs():
    """
    Build FBX export kwargs compatible with the running Blender version.
    Removes parameters that were deprecated/removed across versions.
    """
    major, minor, _ = _get_blender_version()
    _log(f"Blender version: {major}.{minor}.{_}")

    kwargs = dict(
        filepath=None,  # caller fills this in
        use_selection=True,
        global_scale=1.0,
        apply_unit_scale=True,
        apply_scale_options='FBX_SCALE_ALL',
        axis_forward='-Z',
        axis_up='Y',
        object_types={'MESH', 'ARMATURE', 'EMPTY', 'CAMERA', 'LIGHT'},
        use_mesh_modifiers=True,
        mesh_smooth_type='OFF',
        path_mode='COPY',
        batch_mode='OFF',
        use_metadata=True,
        primary_bone_axis='Y',
        secondary_bone_axis='X',
        use_armature_deform_only=True,
    )

    # use_tspace: deprecated in 4.0, removed in 4.2+
    if (major, minor) < (4, 2):
        kwargs['use_tspace'] = True

    # add_leaf_bones: removed in 4.0
    if (major, minor) < (4, 0):
        kwargs['add_leaf_bones'] = True

    # embed_textures: always on (single-file output)
    kwargs['embed_textures'] = True

    _log(f"FBX export kwargs (version-aware): use_tspace={kwargs.get('use_tspace', 'omitted')}, "
         f"add_leaf_bones={kwargs.get('add_leaf_bones', 'omitted')}")
    return kwargs


def glb_to_fbx(glb_path, fbx_path):
    bpy.ops.wm.read_homefile(use_empty=True)

    # Snapshot scene objects BEFORE import to reliably detect new ones
    objects_before = set(bpy.context.scene.objects)

    _log(f"Importing GLB: {glb_path}")
    try:
        import_result = bpy.ops.import_scene.gltf(filepath=glb_path)
    except Exception as exc:
        _log(f"ERROR: GLB import raised exception: {exc}")
        return False

    if 'FINISHED' not in import_result:
        _log(f"ERROR: GLB import operator failed: {import_result}")
        return False

    # Detect imported objects by set difference
    objects_after = set(bpy.context.scene.objects)
    imported_objects = list(objects_after - objects_before)

    if not imported_objects:
        imported_objects = [
            obj for obj in bpy.context.scene.objects
            if obj.name not in ('Camera', 'Light')
        ]

    _log(f"Imported {len(imported_objects)} objects")

    if not imported_objects:
        _log("ERROR: No objects imported from GLB")
        return False

    # ── Diagnostic dump ──────────────────────────────────────────────
    _dump_mesh_diagnostics("AFTER IMPORT", imported_objects)
    mesh_objs = [o for o in imported_objects if o.type == 'MESH']
    for o in mesh_objs:
        _log(f"  '{o.name}' materials:")
        _ensure_materials(o)

    # ── Select all imported objects ──────────────────────────────────
    bpy.ops.object.select_all(action='DESELECT')
    for obj in imported_objects:
        obj.select_set(True)

    bpy.context.view_layer.objects.active = mesh_objs[0] if mesh_objs else imported_objects[0]

    # ── Fix shading & normals BEFORE export ──────────────────────────
    _log("Fixing shading & normals for FBX export...")
    _force_smooth_shading(imported_objects)
    _ensure_split_normals(imported_objects)

    # ── Ensure output directory exists ───────────────────────────────
    out_dir = os.path.dirname(fbx_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # ── FBX Export ───────────────────────────────────────────────────
    _log(f"Exporting to FBX: {fbx_path}")
    kwargs = _build_fbx_export_kwargs()
    kwargs['filepath'] = fbx_path

    try:
        export_result = bpy.ops.export_scene.fbx(**kwargs)
    except Exception as exc:
        _log(f"ERROR: FBX export raised exception: {exc}")
        return False

    if 'FINISHED' not in export_result:
        _log(f"ERROR: FBX export operator failed: {export_result}")
        return False

    # ── Post-export verification ─────────────────────────────────────
    if not os.path.exists(fbx_path):
        _log(f"ERROR: FBX file not found after export: {fbx_path}")
        return False
    fbx_size = os.path.getsize(fbx_path)
    if fbx_size == 0:
        _log(f"ERROR: FBX file is empty (0 bytes): {fbx_path}")
        return False

    _log(f"Export completed: {fbx_size} bytes")
    return True


if __name__ == "__main__":
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
    else:
        args = sys.argv[1:]

    if len(args) < 2:
        print("Usage: blender -b -P blender_glb_to_fbx.py -- glb_path fbx_path")
        sys.exit(1)

    glb_path = args[0]
    fbx_path = args[1]

    if not os.path.exists(glb_path):
        print(f"ERROR: GLB file not found: {glb_path}")
        sys.exit(1)

    fbx_dir = os.path.dirname(fbx_path)
    if fbx_dir:
        os.makedirs(fbx_dir, exist_ok=True)

    success = glb_to_fbx(glb_path, fbx_path)

    if success:
        print(f"SUCCESS: {glb_path} -> {fbx_path}")
        sys.exit(0)
    else:
        print(f"FAILED: {glb_path} -> {fbx_path}")
        sys.exit(1)
