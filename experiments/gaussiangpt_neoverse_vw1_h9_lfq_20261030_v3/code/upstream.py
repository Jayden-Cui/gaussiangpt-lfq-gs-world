"""Load pinned VideoWorld QFormer/FSQ without importing ldm or mmengine.

``qformer, FSQ = load_classes()`` returns the unmodified QFormer module
namespace (including QFormerMFSep) and the official FSQ class. The three
vendored files are verified on every call, including the reference config.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
import types


UPSTREAM_COMMIT = "3dc3e563e201cebab0e947177e79117c885ec3ba"
FSQ_LEVELS = (8, 8, 8, 5, 5, 5)
FSQ_DIM = 512
HORIZON = 9
FRAME_COUNT = 10
EXPECTED_SHA256 = {
    "qformer.py": "220f24100c6449e5bd30e340b9ab0aeb0ba059c487eaf54f262d0aae19e678da",
    "finite_scalar_quantization.py": "a5da889b1607c33688146b627975c86cc278a4cbf4e301e989bbfb09b1fc6e21",
    "calvin_ldm.py": "300dd9168bb14dffec9b054b2536e216b2b32c792c3ede767a7055b996b2e355",
}


def _vendor_path(vendor_dir=None):
    return (Path(__file__).resolve().parent / "vendor" if vendor_dir is None
            else Path(vendor_dir).resolve())


def verify_vendor(vendor_dir=None):
    """Reject unpinned source or metadata before executing any vendor code."""
    folder = _vendor_path(vendor_dir)
    provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
    if (provenance.get("schema") != "videoworld-gsldm-upstream-v1"
            or provenance.get("commit") != UPSTREAM_COMMIT
            or set(provenance.get("files", {})) != set(EXPECTED_SHA256)):
        raise ValueError("VideoWorld provenance does not identify the pinned three-file source")
    for name, expected in EXPECTED_SHA256.items():
        content = (folder / name).read_bytes()
        entry = provenance["files"][name]
        if (hashlib.sha256(content).hexdigest() != expected
                or entry.get("sha256") != expected
                or entry.get("bytes") != len(content)):
            raise ValueError("Pinned VideoWorld source hash mismatch: " + name)
    return provenance


def _fsq_without_registry(content, filename):
    """Remove exactly the registry import and FSQ registration decorator."""
    tree = ast.parse(content, filename=filename)
    imports_removed = decorators_removed = 0
    statements = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "ldm.registry":
            if (node.level != 0 or len(node.names) != 1
                    or node.names[0].name != "MODELS" or node.names[0].asname is not None):
                raise ValueError("Unexpected upstream registry import")
            imports_removed += 1
            continue
        if isinstance(node, ast.ClassDef) and node.name == "FSQ":
            retained = []
            for decorator in node.decorator_list:
                if (isinstance(decorator, ast.Call)
                        and isinstance(decorator.func, ast.Attribute)
                        and isinstance(decorator.func.value, ast.Name)
                        and decorator.func.value.id == "MODELS"
                        and decorator.func.attr == "register_module"
                        and not decorator.args and not decorator.keywords):
                    decorators_removed += 1
                else:
                    retained.append(decorator)
            node.decorator_list = retained
        statements.append(node)
    if (imports_removed, decorators_removed) != (1, 1):
        raise ValueError("Pinned FSQ registry boundary changed")
    tree.body = statements
    if any(isinstance(node, ast.Name) and node.id == "MODELS" for node in ast.walk(tree)):
        raise ValueError("FSQ still refers to the removed registry")
    return ast.fix_missing_locations(tree)


def _module(name, path, source, transform=None):
    # Register stable module names so class identity and ordinary pickling remain
    # usable. Neither module is imported through the ldm package hierarchy.
    if name in sys.modules:
        module = sys.modules[name]
        if getattr(module, "__source_sha256__", None) != hashlib.sha256(source).hexdigest():
            raise ValueError("An incompatible upstream module is already loaded: " + name)
        return module
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__source_sha256__ = hashlib.sha256(source).hexdigest()
    program = source if transform is None else transform(source, str(path))
    sys.modules[name] = module
    try:
        exec(compile(program, str(path), "exec"), module.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def load_classes(vendor_dir=None):
    """Return ``(qformer_module, FSQ_class)`` after strict source verification.

    Example: ``qformer.QFormerMFSep(2, 512, 4, 512, qformer_num=9,
    time_padding=3)`` and ``FSQ(levels=list(FSQ_LEVELS), dim=FSQ_DIM)``.
    QFormer is executed unchanged. Only FSQ's framework registration is removed;
    projections, quantization, straight-through gradients and indexing are the
    original implementation. PyTorch, NumPy and einops remain real dependencies.
    """
    verify_vendor(vendor_dir)
    folder = _vendor_path(vendor_dir)
    qpath, fpath = folder / "qformer.py", folder / "finite_scalar_quantization.py"
    qformer = _module("_gsldm_videoworld_3dc3e563_qformer", qpath, qpath.read_bytes())
    quantization = _module("_gsldm_videoworld_3dc3e563_fsq", fpath, fpath.read_bytes(),
                           _fsq_without_registry)
    return qformer, quantization.FSQ
