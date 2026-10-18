"""Build deepseek_v4.synthetic_token_loop_decode_flash from its preserved official metadata."""

import ast
from types import ModuleType, SimpleNamespace
from typing import Callable

import common
import torch
from golden import ScalarSpec, TensorSpec


def _functions(path, identity, names, bindings, *, metadata_import=False):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in functions} != set(names):
        raise ValueError(f"missing official helpers in {path}")
    if metadata_import:
        for function in functions:
            function.body = [
                node for node in function.body
                if not (isinstance(node, ast.ImportFrom) and node.module == "decode_metadata")
            ]
    module = ModuleType(identity)
    module.__file__ = str(path)
    module.__dict__.update({name: value for name, value in bindings.items() if not name.startswith("__")})
    module.__dict__.update(torch=torch, Callable=Callable, SimpleNamespace=SimpleNamespace)
    selected = ast.Module(body=functions, type_ignores=[])
    exec(compile(selected, str(path), "exec"), module.__dict__)
    return module


def _source_contract(path, config, entry):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == entry
    )
    bindings = dict(vars(config))
    bindings["EMBED_VOCAB_DYN"] = config.CASE_CONFIG.vocab_size
    dtypes = {
        "BF16": torch.bfloat16, "FP32": torch.float32,
        "INT8": torch.int8, "INT32": torch.int32, "INT64": torch.int64,
    }
    tensors = []
    scalars = []
    for parameter in function.args.args:
        annotation = parameter.annotation
        direction = "in"
        if annotation.value.attr in ("Out", "InOut"):
            direction = annotation.value.attr.lower()
            annotation = annotation.slice
        if annotation.value.attr == "Scalar":
            scalars.append((parameter.arg, dtypes[annotation.slice.attr]))
            continue
        shape_node, dtype_node = annotation.slice.elts
        shape = eval(compile(ast.Expression(shape_node), str(path), "eval"), bindings)
        tensors.append((parameter.arg, list(shape), dtypes[dtype_node.attr], direction))
    assignments = {
        target.id: node.value
        for node in tree.body if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name)
    }
    constants = {}

    def constant(name):
        if name not in constants:
            expression = assignments[name]
            for dependency in ast.walk(expression):
                if isinstance(dependency, ast.Name) and dependency.id in assignments:
                    constant(dependency.id)
            constants[name] = eval(
                compile(ast.Expression(expression), str(path), "eval"), dict(constants)
            )
        return constants[name]

    resident = set(constant("RESIDENT_WEIGHT_NAMES"))
    resident.update(constant("RESIDENT_CACHE_OUTPUT_NAMES"))
    return tensors, scalars, resident


def _metadata(case_dir, sealed, case, config, tensors, resident):
    identity = case["case_id"].replace(".", "_")
    official = sealed / "official"
    bindings = dict(vars(config))
    bindings.update(M=config.CASE_CONFIG, MODEL_CONFIG=config.CASE_CONFIG)
    metadata = _functions(
        official / "decode_metadata.py", identity + "_official_metadata",
        (
            "resolve_start_positions", "position_ids_from_starts",
            "kv_seq_lens_from_starts", "block_table", "ori_slot_mapping",
            "paged_slot_mapping", "swa_indices_and_lens", "compressed_slot_mapping",
            "mask_uncommitted_compressed_boundaries", "state_slot_mapping",
            "_validate_starts",
        ),
        bindings,
    )
    session_bindings = dict(bindings, PROMPT_TOKENS=config.PREFILL_TOKENS)
    if case["entry"] == "l3_prefill_fwd":
        session_names = (
            "_initialize_tid2eid", "_session_table", "_build_session_tables",
            "_ranked_prefill", "_initialize_prefill_io",
        )
    else:
        session_names = (
            "_initialize_tid2eid", "_session_table", "_build_session_tables",
            "_decode_base_shapes", "_refresh_decode_io",
        )
    session = _functions(
        official / "original.py", identity + "_official_session",
        session_names, session_bindings,
    )
    decode_bindings = dict(bindings, B=config.DECODE_BATCH, T=config.DECODE_TOKENS)
    decode = SimpleNamespace(**decode_bindings)
    tables = session._build_session_tables(decode)
    output_names = set(case["correctness"]["outputs"])
    io_tensors = {
        name: torch.zeros(shape, dtype=dtype)
        for name, shape, dtype, direction in tensors
        if name not in resident and name not in output_names
    }
    scalars = {}
    if case["entry"] == "l3_prefill_fwd":
        session._initialize_prefill_io(config, metadata, io_tensors, scalars, tables)
    else:
        forward_bindings = dict(decode_bindings)
        forward_bindings.update(vars(metadata))
        forward = _functions(
            official / "decode_fwd.py", identity + "_official_decode_metadata",
            ("make_forward_metadata_tensors",), forward_bindings, metadata_import=True,
        )
        decode.make_forward_metadata_tensors = forward.make_forward_metadata_tensors
        io_tensors.update(logits=torch.empty(0), sampled_ids=torch.empty(0, dtype=torch.int32))
        session._refresh_decode_io(
            decode, io_tensors, scalars, tables, config.START_POS,
            torch.full((config.N_RANKS,), config.INPUT_TOKEN, dtype=torch.int64),
        )
        io_tensors.pop("logits")
        io_tensors.pop("sampled_ids")
    return io_tensors, session, metadata, tables


def build(case_dir, sealed, case, seed):
    """Return lazy specs; only metadata is constructed before materialization."""
    identity = case["case_id"].replace(".", "_")
    config = common.load_module(case_dir / case["config"], identity + "_fixture_config")
    if case["case_id"] != config.CASE_ID or case["entry"] != config.ENTRY:
        raise ValueError("Case identity disagrees with the fixed public contract")
    torch.manual_seed(seed)
    official = sealed / "official"
    tensors, scalars, resident = _source_contract(
        official / config.SOURCE_ENTRY_FILE, config, case["entry"],
    )
    outputs = {name for name, shape, dtype, direction in tensors if direction in ("out", "inout")}
    mutable = {name for name, shape, dtype, direction in tensors if direction == "inout"}
    if outputs != set(case["correctness"]["outputs"]) or mutable != set(case["state"]["initial_state"]):
        raise ValueError("official annotations disagree with output/state coverage")
    metadata_values, session, metadata, tables = _metadata(
        case_dir, sealed, case, config, tensors, resident,
    )

    def initializer(name, shape, dtype):
        if name in metadata_values:
            return metadata_values[name].clone().contiguous()
        if name == "logits":
            return torch.full(shape, float("nan"), dtype=dtype)
        if name == "sampled_ids":
            return torch.full(shape, -1, dtype=dtype)
        value = torch.zeros(shape, dtype=dtype)
        if name == "tid2eid":
            session._initialize_tid2eid(
                value, vocab_size=config.VOCAB, topk=config.TOPK,
                num_experts=config.N_EXPERTS_GLOBAL,
            )
        if name == "idx_kv_scale" and case["entry"] == "l3_decode_fwd":
            positions = torch.arange(config.PREFILL_TOKENS, dtype=torch.int32).reshape(1, -1)
            slots = metadata.compressed_slot_mapping(
                positions, tables["idx_block_table"][0:1],
                compress_ratio=config.CSA_COMPRESS_RATIO, block_size=config.BLOCK_SIZE,
            ).reshape(-1)
            scale = torch.reciprocal(
                torch.tensor(config.INT8_SCALE_MAX, dtype=torch.float32)
                / torch.tensor(config.INT8_AMAX_EPS, dtype=torch.float32)
            )
            history = value.view(config.N_RANKS, config.CSA_NUM_LAYERS, -1)
            history[:, :, slots[slots >= 0]] = scale
        return value

    specs = [
        TensorSpec(
            name, shape, dtype,
            init_value=lambda name=name, shape=shape, dtype=dtype: initializer(name, shape, dtype),
            resident="stacked" if name in resident else None,
        )
        for name, shape, dtype, direction in tensors
    ]
    for name, dtype in scalars:
        if name not in case["scalars"]:
            raise ValueError(f"undeclared official scalar {name}")
        specs.append(ScalarSpec(name, dtype, config.NUM_TOKENS))
    return specs
