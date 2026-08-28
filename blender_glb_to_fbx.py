import sys
import os
import bpy

def glb_to_fbx(glb_path, fbx_path):
    bpy.ops.wm.read_homefile(use_empty=True)
    
    print(f"[Blender] Importing GLB: {glb_path}")
    bpy.ops.import_scene.gltf(filepath=glb_path)
    
    imported_objects = [obj for obj in bpy.context.scene.objects if obj.select_get()]
    print(f"[Blender] Imported {len(imported_objects)} objects")
    
    if not imported_objects:
        print("[Blender] ERROR: No objects imported from GLB")
        return False
    
    bpy.ops.object.select_all(action='DESELECT')
    for obj in imported_objects:
        obj.select_set(True)
    
    bpy.context.view_layer.objects.active = imported_objects[0]
    
    print(f"[Blender] Exporting to FBX: {fbx_path}")
    bpy.ops.export_scene.fbx(
        filepath=fbx_path,
        use_selection=True,
        global_scale=1.0,
        apply_unit_scale=True,
        apply_scale_options='FBX_SCALE_ALL',
        axis_forward='-Z',
        axis_up='Y',
        object_types={'MESH', 'ARMATURE', 'EMPTY', 'CAMERA', 'LIGHT'},
        use_mesh_modifiers=True,
        mesh_smooth_type='FACE',
        use_tspace=True,
        embed_textures=True,
        path_mode='COPY',
        batch_mode='OFF',
        use_metadata=True,
        primary_bone_axis='Y',
        secondary_bone_axis='X',
        use_armature_deform_only=True,
        add_leaf_bones=True,
    )
    
    print(f"[Blender] Export completed successfully")
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
    
    os.makedirs(os.path.dirname(fbx_path), exist_ok=True)
    
    success = glb_to_fbx(glb_path, fbx_path)
    
    if success:
        print(f"SUCCESS: {glb_path} -> {fbx_path}")
        sys.exit(0)
    else:
        print(f"FAILED: {glb_path} -> {fbx_path}")
        sys.exit(1)
