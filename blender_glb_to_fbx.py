import sys
import os
import struct
import json
from io import BytesIO
import bpy


def _log(msg):
    print(f"[BlenderFBX] {msg}", flush=True)


try:
    import numpy
    _log(f"numpy available: {numpy.__version__}")
except ImportError:
    _log("FATAL: numpy is NOT installed. FBX export requires numpy for glTF import.")
    _log("Install with: blender --background --python-expr \"import subprocess,sys; subprocess.check_call([sys.executable,'-m','pip','install','numpy'])\"")
    sys.exit(1)

try:
    from PIL import Image
    _log(f"Pillow available: {Image.__version__}")
except ImportError:
    _log("WARNING: Pillow not available — WebP textures cannot be converted")
    Image = None


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


def _verify_material_connections(mesh_obj):
    """Verify that textures are properly connected to Principled BSDF inputs."""
    issues = 0
    for slot in mesh_obj.material_slots:
        mat = slot.material
        if mat is None or not mat.use_nodes or not mat.node_tree:
            continue

        principled = None
        for node in mat.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                principled = node
                break

        if principled is None:
            _log(f"  WARNING: Material '{mat.name}' has no Principled BSDF node")
            issues += 1
            continue

        connected_images = set()
        for link in mat.node_tree.links:
            if link.from_node.type == 'TEX_IMAGE' and link.from_node.image:
                connected_images.add(link.from_node.name)

        for node in mat.node_tree.nodes:
            if node.type == 'TEX_IMAGE' and node.name not in connected_images:
                _log(f"  WARNING: '{mat.name}' orphaned texture '{node.name}' (not connected to Principled BSDF)")
                issues += 1

        base_color_linked = principled.inputs['Base Color'].is_linked
        if not base_color_linked:
            _log(f"  WARNING: '{mat.name}' Base Color not connected")
            issues += 1

    return issues


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


def _has_webp_magic(data):
    """Check if binary data starts with RIFF....WEBP magic bytes."""
    if len(data) < 12:
        return False
    riff = struct.unpack('<I', data[0:4])[0]
    webp = struct.unpack('<I', data[8:12])[0]
    return riff == 0x52494646 and webp == 0x57454250


def _find_webp_buffer_views(gltf, bin_data):
    """Find all bufferViews that contain WebP data by scanning for magic bytes."""
    webp_views = []
    for i, bv in enumerate(gltf.get('bufferViews', [])):
        offset = bv.get('byteOffset', 0)
        size = bv.get('byteLength', 0)
        if size >= 12:
            sample = bytes(bin_data[offset:offset + min(16, size)])
            if _has_webp_magic(sample):
                webp_views.append(i)
    return webp_views


def _preprocess_glb_webp_to_png(glb_path):
    """Convert WebP textures in GLB to PNG for Blender 3.0.1 compatibility."""
    _log(f"Preprocessing GLB for WebP->PNG: {glb_path}")

    try:
        with open(glb_path, 'rb') as f:
            magic, version, length = struct.unpack('<III', f.read(12))
            if magic != 0x46546C67:
                _log(f"Not a GLB file (magic=0x{magic:08X}) — skipping preprocessing")
                return glb_path

            bin_data = bytearray()
            json_bytes = b''

            while True:
                chunk_header = f.read(8)
                if len(chunk_header) < 8:
                    break
                chunk_length, chunk_type = struct.unpack('<II', chunk_header)
                chunk_data = f.read(chunk_length)
                if len(chunk_data) < chunk_length:
                    _log(f"  Truncated chunk type=0x{chunk_type:08X}")
                    break
                if chunk_type == 0x4E4F534A:
                    json_bytes = chunk_data
                elif chunk_type == 0x004E4942:
                    bin_data.extend(chunk_data)
                else:
                    _log(f"  Unknown chunk type=0x{chunk_type:08X}, length={chunk_length}")
    except Exception as exc:
        _log(f"ERROR: Failed to parse GLB: {exc}")
        return glb_path

    if not json_bytes:
        _log("No JSON chunk found in GLB — skipping preprocessing")
        return glb_path

    gltf = json.loads(json_bytes)
    ext_used = gltf.get('extensionsUsed', [])
    ext_req = gltf.get('extensionsRequired', [])

    _log(f"  GLB extensionsUsed: {ext_used}")
    _log(f"  GLB extensionsRequired: {ext_req}")

    has_webp_ext = ('EXT_texture_webp' in ext_used or 'EXT_texture_webp' in ext_req or
                     'KHR_texture_webp' in ext_used or 'KHR_texture_webp' in ext_req)

    webp_views = _find_webp_buffer_views(gltf, bin_data) if bin_data else []
    _log(f"  WebP magic bytes found in {len(webp_views)} bufferViews: {webp_views}")

    has_webp_mime = False
    for img in gltf.get('images', []):
        mime = img.get('mimeType', '')
        if 'webp' in mime.lower():
            has_webp_mime = True
            break

    json_str = json_bytes.decode('utf-8', errors='replace').lower()
    has_webp_in_json = 'webp' in json_str

    needs_conversion = has_webp_ext or len(webp_views) > 0 or has_webp_mime or has_webp_in_json
    _log(f"  Detection: ext_declared={has_webp_ext} binary_magic={len(webp_views)>0} "
         f"mime_type={has_webp_mime} json_ref={has_webp_in_json} -> needs_conversion={needs_conversion}")

    if not needs_conversion:
        _log("No WebP textures detected — skipping preprocessing")
        return glb_path

    if Image is None:
        _log("ERROR: WebP textures detected but Pillow not available — cannot convert")
        _log("Attempting import without conversion (may fail)...")
        return glb_path

    _log("Converting WebP textures to PNG for Blender compatibility...")
    converted = 0

    images = gltf.get('images', [])
    for i, image in enumerate(images):
        mime = image.get('mimeType', '')
        ext_data = image.get('extensions', {}).get('EXT_texture_webp', {})
        if not ext_data:
            ext_data = image.get('extensions', {}).get('KHR_texture_webp', {})
        bv_index = ext_data.get('bufferView')

        if bv_index is None:
            bv_index = image.get('bufferView')

        source_index = ext_data.get('source')

        if bv_index is None and source_index is not None and source_index < len(images):
            source_img = images[source_index]
            bv_index = source_img.get('bufferView')
            if bv_index is None:
                src_ext = source_img.get('extensions', {}).get('EXT_texture_webp', {})
                bv_index = src_ext.get('bufferView')

        if bv_index is None and i in webp_views:
            bv_index = i

        if bv_index is None:
            _log(f"  image[{i}]: no bufferView found — skipping")
            continue

        if bv_index >= len(gltf.get('bufferViews', [])):
            _log(f"  image[{i}]: bufferView index {bv_index} out of range — skipping")
            continue

        bv = gltf['bufferViews'][bv_index]
        offset = bv.get('byteOffset', 0)
        size = bv['byteLength']

        if offset + size > len(bin_data):
            _log(f"  image[{i}]: bufferView data out of range — skipping")
            continue

        raw_data = bytes(bin_data[offset:offset + size])

        if mime != 'image/webp' and not _has_webp_magic(raw_data):
            _log(f"  image[{i}]: not WebP data (mime={mime}) — skipping")
            continue

        try:
            img = Image.open(BytesIO(raw_data))
            png_buf = BytesIO()
            img.save(png_buf, format='PNG')
            png_data = png_buf.getvalue()
        except Exception as exc:
            _log(f"  image[{i}] WebP->PNG convert failed: {exc}")
            continue

        new_offset = len(bin_data)
        bin_data.extend(png_data)

        bv['byteOffset'] = new_offset
        bv['byteLength'] = len(png_data)
        bv.pop('byteStride', None)

        image['mimeType'] = 'image/png'

        if 'bufferView' in image:
            pass
        else:
            image['bufferView'] = bv_index

        image.get('extensions', {}).pop('EXT_texture_webp', None)
        if not image.get('extensions'):
            image.pop('extensions', None)

        converted += 1
        _log(f"  image[{i}] WebP -> PNG ({size} -> {len(png_data)} bytes)")

    for i in webp_views:
        found_in_images = False
        for img in images:
            if img.get('bufferView') == i:
                found_in_images = True
                break
            ext_data = img.get('extensions', {}).get('EXT_texture_webp', {})
            if ext_data.get('bufferView') == i:
                found_in_images = True
                break
            ext_data = img.get('extensions', {}).get('KHR_texture_webp', {})
            if ext_data.get('bufferView') == i:
                found_in_images = True
                break
        if not found_in_images:
            bv = gltf['bufferViews'][i]
            offset = bv.get('byteOffset', 0)
            size = bv.get('byteLength', 0)
            raw_data = bytes(bin_data[offset:offset + size])
            if _has_webp_magic(raw_data) and Image is not None:
                try:
                    img = Image.open(BytesIO(raw_data))
                    png_buf = BytesIO()
                    img.save(png_buf, format='PNG')
                    png_data = png_buf.getvalue()
                    new_offset = len(bin_data)
                    bin_data.extend(png_data)
                    bv['byteOffset'] = new_offset
                    bv['byteLength'] = len(png_data)
                    bv.pop('byteStride', None)
                    converted += 1
                    _log(f"  orphan bufferView[{i}] WebP -> PNG ({size} -> {len(png_data)} bytes)")
                except Exception as exc:
                    _log(f"  orphan bufferView[{i}] convert failed: {exc}")

    all_webp_exts = ['EXT_texture_webp', 'KHR_texture_webp']
    for ext_name in all_webp_exts:
        while ext_name in ext_used:
            ext_used.remove(ext_name)
        while ext_name in ext_req:
            ext_req.remove(ext_name)

    for img in images:
        img.get('extensions', {}).pop('EXT_texture_webp', None)
        img.get('extensions', {}).pop('KHR_texture_webp', None)
        if not img.get('extensions'):
            img.pop('extensions', None)

    if not ext_used and 'extensionsUsed' in gltf:
        gltf.pop('extensionsUsed', None)
    if not ext_req and 'extensionsRequired' in gltf:
        gltf.pop('extensionsRequired', None)

    _log(f"Converted {converted} texture(s) from WebP to PNG")

    if converted == 0:
        _log("WARNING: Detected WebP but could not convert any textures")
        _log("Will try to import anyway (Blender may fail)...")

    new_glb = glb_path.replace('.glb', '_converted.glb')
    new_json = json.dumps(gltf).encode('utf-8')
    while len(new_json) % 4 != 0:
        new_json += b'\x20'
    while len(bin_data) % 4 != 0:
        bin_data.append(0)

    total = 12 + 8 + len(new_json) + 8 + len(bin_data)
    with open(new_glb, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, version, total))
        f.write(struct.pack('<II', len(new_json), 0x4E4F534A))
        f.write(new_json)
        f.write(struct.pack('<II', len(bin_data), 0x004E4942))
        f.write(bin_data)

    _log(f"Preprocessed GLB written: {new_glb} ({total} bytes)")
    return new_glb


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
        mesh_smooth_type='FACE',
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


def _patch_blender_webp_support():
    """Try to make Blender's glTF importer accept EXT_texture_webp by monkey-patching."""
    try:
        from io_scene_gltf2.io.imp.gltf2_io_gltf import glTFImporter as _Imp
        if hasattr(_Imp, 'supported_extensions'):
            if 'EXT_texture_webp' not in _Imp.supported_extensions:
                _Imp.supported_extensions = set(_Imp.supported_extensions) | {'EXT_texture_webp'}
                _log("  Patched glTF importer: added EXT_texture_webp to supported extensions")
                return True
    except Exception:
        pass

    _log("  Could not monkey-patch glTF importer — relying on preprocessed GLB only")
    return False


def _try_import_glb(glb_path, objects_before):
    """Try to import GLB and return imported objects list, or None on failure."""
    try:
        import_result = bpy.ops.import_scene.gltf(filepath=glb_path)
    except Exception as exc:
        _log(f"  Import exception: {exc}")
        return None

    if 'FINISHED' not in import_result:
        _log(f"  Import operator returned: {import_result}")
        return None

    objects_after = set(bpy.context.scene.objects)
    imported = list(objects_after - objects_before)

    if not imported:
        imported = [
            obj for obj in bpy.context.scene.objects
            if obj.name not in ('Camera', 'Light')
        ]

    return imported if imported else None


def _preprocess_strip_all_extensions(glb_path):
    """Nuclear option: strip ALL extensions and convert WebP->PNG data from GLB."""
    _log("Stripping ALL extensions from GLB (nuclear option)...")
    try:
        with open(glb_path, 'rb') as f:
            magic, version, length = struct.unpack('<III', f.read(12))
            if magic != 0x46546C67:
                return glb_path

            bin_data = bytearray()
            json_bytes = b''

            while True:
                chunk_header = f.read(8)
                if len(chunk_header) < 8:
                    break
                chunk_length, chunk_type = struct.unpack('<II', chunk_header)
                chunk_data = f.read(chunk_length)
                if len(chunk_data) < chunk_length:
                    break
                if chunk_type == 0x4E4F534A:
                    json_bytes = chunk_data
                elif chunk_type == 0x004E4942:
                    bin_data.extend(chunk_data)

        if not json_bytes:
            return glb_path

        gltf = json.loads(json_bytes)
        gltf.pop('extensionsUsed', None)
        gltf.pop('extensionsRequired', None)

        images = gltf.get('images', [])
        converted_count = 0

        for i, img in enumerate(images):
            img.pop('extensions', None)
            bv_index = img.get('bufferView')

            if bv_index is not None and bv_index < len(gltf.get('bufferViews', [])):
                bv = gltf['bufferViews'][bv_index]
                offset = bv.get('byteOffset', 0)
                size = bv.get('byteLength', 0)

                if offset + size <= len(bin_data):
                    raw_data = bytes(bin_data[offset:offset + size])

                    is_webp = (img.get('mimeType', '') == 'image/webp') or _has_webp_magic(raw_data)

                    if is_webp and Image is not None:
                        try:
                            pil_img = Image.open(BytesIO(raw_data))
                            png_buf = BytesIO()
                            pil_img.save(png_buf, format='PNG')
                            png_data = png_buf.getvalue()

                            new_offset = len(bin_data)
                            bin_data.extend(png_data)
                            bv['byteOffset'] = new_offset
                            bv['byteLength'] = len(png_data)
                            bv.pop('byteStride', None)

                            img['mimeType'] = 'image/png'
                            converted_count += 1
                            _log(f"  nuclear: image[{i}] WebP -> PNG ({size} -> {len(png_data)} bytes)")
                            continue
                        except Exception as exc:
                            _log(f"  nuclear: image[{i}] WebP convert failed: {exc}")

                    if is_webp and Image is None:
                        _log(f"  nuclear: image[{i}] is WebP but Pillow unavailable — stripping mimeType")
                        img.pop('mimeType', None)
                        continue

            img.pop('mimeType', None)

        for tex in gltf.get('textures', []):
            tex.pop('extensions', None)
        for mat in gltf.get('materials', []):
            mat.pop('extensions', None)
        for mesh in gltf.get('meshes', []):
            for prim in mesh.get('primitives', []):
                prim.pop('extensions', None)

        _log(f"  nuclear: converted {converted_count} WebP textures, stripped all extensions")

        new_glb = glb_path.replace('.glb', '_stripped.glb')
        new_json = json.dumps(gltf).encode('utf-8')
        while len(new_json) % 4 != 0:
            new_json += b'\x20'
        while len(bin_data) % 4 != 0:
            bin_data.append(0)

        total = 12 + 8 + len(new_json) + 8 + len(bin_data)
        with open(new_glb, 'wb') as f:
            f.write(struct.pack('<III', 0x46546C67, version, total))
            f.write(struct.pack('<II', len(new_json), 0x4E4F534A))
            f.write(new_json)
            f.write(struct.pack('<II', len(bin_data), 0x004E4942))
            f.write(bin_data)

        _log(f"Stripped GLB written: {new_glb} ({total} bytes)")
        return new_glb
    except Exception as exc:
        _log(f"ERROR stripping extensions: {exc}")
        return glb_path


def glb_to_fbx(glb_path, fbx_path):
    bpy.ops.wm.read_homefile(use_empty=True)

    original_glb = glb_path
    temp_files_to_cleanup = []

    glb_path = _preprocess_glb_webp_to_png(glb_path)
    if glb_path != original_glb:
        temp_files_to_cleanup.append(glb_path)

    objects_before = set(bpy.context.scene.objects)

    _log(f"Importing GLB: {glb_path}")
    imported_objects = _try_import_glb(glb_path, objects_before)

    if imported_objects is None:
        _log("First import attempt failed — trying Blender addon patch...")
        _patch_blender_webp_support()

        bpy.ops.wm.read_homefile(use_empty=True)
        objects_before = set(bpy.context.scene.objects)
        imported_objects = _try_import_glb(glb_path, objects_before)

    if imported_objects is None:
        _log("Second import attempt failed — trying to strip ALL extensions from GLB...")
        stripped_glb = _preprocess_strip_all_extensions(glb_path)
        if stripped_glb != glb_path:
            temp_files_to_cleanup.append(stripped_glb)
        glb_path = stripped_glb

        bpy.ops.wm.read_homefile(use_empty=True)
        objects_before = set(bpy.context.scene.objects)
        imported_objects = _try_import_glb(glb_path, objects_before)

    for tf in temp_files_to_cleanup:
        try:
            if os.path.exists(tf):
                os.unlink(tf)
                _log(f"  Cleaned up temp file: {tf}")
        except Exception:
            pass

    if imported_objects is None:
        _log("ERROR: All import attempts failed")
        return False

    _log(f"Imported {len(imported_objects)} objects")

    if not imported_objects:
        _log("ERROR: No objects imported from GLB")
        return False

    # ── Diagnostic dump ──────────────────────────────────────────────
    _dump_mesh_diagnostics("AFTER IMPORT", imported_objects)
    mesh_objs = [o for o in imported_objects if o.type == 'MESH']
    total_connection_issues = 0
    for o in mesh_objs:
        _log(f"  '{o.name}' materials:")
        _ensure_materials(o)
        total_connection_issues += _verify_material_connections(o)

    if total_connection_issues > 0:
        _log(f"  WARNING: {total_connection_issues} material connection issue(s) detected")

    # ── Select all imported objects ──────────────────────────────────
    bpy.ops.object.select_all(action='DESELECT')
    for obj in imported_objects:
        obj.select_set(True)

    bpy.context.view_layer.objects.active = mesh_objs[0] if mesh_objs else imported_objects[0]

    # ── Fix shading & normals BEFORE export ──────────────────────────
    _log("Fixing shading & normals for FBX export...")
    _force_smooth_shading(imported_objects)
    _ensure_split_normals(imported_objects)

    # ── Verify textures loaded & pack them into Blender ─────────────
    _log("Verifying textures and packing for FBX export...")
    total_tex = 0
    pack_failures = 0
    for mat in bpy.data.materials:
        if not mat.use_nodes or not mat.node_tree:
            continue
        tex_count = 0
        for node in mat.node_tree.nodes:
            if node.type == 'TEX_IMAGE' and node.image:
                tex_count += 1
                if not node.image.packed_file:
                    try:
                        node.image.pack()
                    except Exception as exc:
                        _log(f"    FAILED to pack texture '{node.image.name}': {exc}")
                        pack_failures += 1
                        continue

                if node.image.packed_file:
                    packed_size = len(node.image.packed_file.data)
                    _log(f"    Packed: '{node.image.name}' ({node.image.size[0]}x{node.image.size[1]}, {packed_size} bytes)")
                else:
                    _log(f"    CRITICAL: '{node.image.name}' pack() succeeded but packed_file is None")
                    pack_failures += 1
        total_tex += tex_count
        if tex_count > 0:
            _log(f"  Material '{mat.name}': {tex_count} texture(s)")
        else:
            _log(f"  WARNING: Material '{mat.name}' has NO textures — FBX will be untextured")

    if pack_failures > 0:
        _log(f"  WARNING: {pack_failures} texture(s) failed to pack — FBX may be missing textures")
    if total_tex == 0:
        _log("WARNING: No textures found across all materials — FBX will have flat shading only")
    else:
        _log(f"  Total textures: {total_tex} (packed: {total_tex - pack_failures})")

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
