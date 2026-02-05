# encoding: utf-8
# ONNX APIs: https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md

import argparse
import io
import logging
import os
import sys
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.onnx import OperatorExportTypes

try:
    import onnx
    import onnxoptimizer
    import onnxruntime
    import onnxsim
except Exception as e:
    print("COUND NOT IMPORT ONNX:", e)
import time

try:
    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
except Exception as e:
    print("COULD NOT IMPORT TENSORRT", e)

# from onnxconverter_common.auto_mixed_precision import auto_convert_mixed_precision
# from onnxconverter_common.float16 import convert_float_to_float16

logging.basicConfig()
logger = logging.getLogger(__name__)


def _remove_initializer_from_input(model):
    if model.ir_version < 4:
        raise AssertionError(
            "Model with ir_version below 4 requires to include initilizer in graph input"
        )

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    removes = []
    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])
            removes.append(initializer.name)
    return model, removes


def _create_feed_dict(input_names, inputs):
    assert len(input_names) == len(inputs)
    return {inp_name: inp for inp_name, inp in zip(input_names, inputs)}


def as_torch_tensors(arrs, device="cuda"):
    """To torch Tensors"""
    if not ((isinstance(arrs, list) or isinstance(arrs, tuple))):
        arrs = (arrs,)
    arrs = tuple(torch.Tensor(inp).to(device) for inp in arrs)
    return arrs


def as_numpy_arrays(arrs):
    """To numpy arrays"""
    if not (isinstance(arrs, list) or isinstance(arrs, tuple)):
        arrs = (arrs,)
    ret_inputs = []
    for inp in arrs:
        if isinstance(inp, Tensor):
            ret_inp = inp.cpu().numpy()
        else:
            ret_inp = np.array(inp)
        ret_inputs.append(ret_inp)
    return ret_inputs


def validate_all_close(arrs1, arrs2, rtol=1e-3, atol=1e-5):
    for r1, r2 in zip(arrs1, arrs2):
        if not np.testing.assert_allclose(r1, r2, rtol=rtol, atol=atol):
            return False
    return True


def torch2onnx(
    torch_model: Union[torch.nn.Module, torch.jit.ScriptModule],
    sample_inputs: Union[Sequence[Tensor], Tensor],
    input_names: List[str],
    output_names: List[str],
    save_path: str,
    precision: str = "fp16",
    device="cuda",
    export_type: int = OperatorExportTypes.ONNX_ATEN_FALLBACK,
    opset_version: Optional[int] = None,
    dynamic_batching: bool = True,
    batch_axis: int = 0,
    validate_fn=None,
    rtol=1e-3,
    atol=1e-5,
    verbose: bool = True,
):
    """
    Trace and export a model to onnx format.

    Args:
        torch_model:
            Torch model to convert.
        sample_inputs:
            If list or tuple, the model will be called by `model(*sample_inputs)`.
            Else if single Tensor, `model(sample_inputs)`.
        save_path:
            Path to save onnx model.
        device:
            cpu or cuda
        export_type:
            export type, more https://pytorch.org/docs/stable/onnx.html#functions.
            Default, try to export each ATen op (in the TorchScript namespace “aten”) as a regular ONNX op.
            If we are unable to do so (e.g. because support has not been added to convert a particular torch op to ONNX),
            fall back to exporting an ATen op.
        opset_version:
            ONNX opset version used to convert. If None (default), use lastest opset version available `onnx.defs.onnx_opset_version()`.
        dynamic_batching:
            Whether or not using dynamic batching.
        batch_axis:
            batch dimension axis index
        num_ouputs:
            Number of model outputs.
        verbose:
            Show verbose information or not.

    Returns:
        Optimized ONNX model
    """
    assert precision in ["fp32", "fp16", "amp", "int8"]
    # Internally, torch.onnx.export() requires a torch.jit.ScriptModule rather than a torch.nn.Module.
    # If the passed-in model is not already a ScriptModule, export() will use tracing to convert it to one
    assert isinstance(torch_model, torch.nn.Module) or isinstance(
        torch_model, torch.jit.ScriptModule
    )
    torch_model.to(device)
    torch_model.eval()

    num_outputs = len(output_names)

    # test model inference in Torch first
    sample_torch_inputs = as_torch_tensors(sample_inputs, device=device)
    sample_batch_size = sample_torch_inputs[0].size(batch_axis)
    with torch.inference_mode():
        sample_torch_outputs = torch_model(*sample_torch_inputs)
        if num_outputs == 1:
            sample_torch_outputs = (sample_torch_outputs,)
        assert len(sample_torch_outputs) == num_outputs
        for output in sample_torch_outputs:
            assert (
                isinstance(output, Tensor)
                and output.size()[batch_axis] == sample_batch_size
            ), f"{type(output)}, {output.keys()}"

    # make sure all modules are in eval mode, onnx may change the training state
    # of the module if the states are not consistent
    def _check_eval(module):
        assert not module.training

    torch_model.apply(_check_eval)
    # if opset_version is None:
    #     opset_version = onnx.defs.onnx_opset_version()

    # dynamic batching
    dynamic_axes = {}
    if dynamic_batching:
        for input_name in input_names:
            dynamic_axes[input_name] = {batch_axis: "batch_size"}
        for output_name in output_names:
            dynamic_axes[output_name] = {batch_axis: "batch_size"}

    logging.info("Input names: %s", input_names)
    logging.info("Ouput names: %s", output_names)
    logging.info(
        "Dynamic batching = %s with config:\n%s", dynamic_batching, dynamic_axes
    )

    logger.info("Beginning ONNX file converting")
    # Export the model to ONNX
    with torch.inference_mode():
        with io.BytesIO() as f:
            torch.onnx.export(
                torch_model,
                sample_torch_inputs,
                f,
                export_params=True,
                verbose=verbose,
                input_names=input_names,
                output_names=output_names,
                operator_export_type=export_type,
                opset_version=opset_version,
                do_constant_folding=True,
                dynamic_axes=dynamic_axes,
                keep_initializers_as_inputs=False,
                custom_opsets={},
                export_modules_as_functions=False,
            )
            onnx_model = onnx.load_from_string(f.getvalue())
    logger.info("Completed convert of ONNX model")

    # Apply ONNX's Optimization with onnxsimplifier
    logger.info("Beginning ONNX optimization with onnxsimplifier")
    onnx_model, check = onnxsim.simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx_model, _removes = _remove_initializer_from_input(onnx_model)
    if _removes:
        logging.info("Removed initializers: %s", _removes)

    ##### QUANTIZATION #####
    if precision == "fp32":
        pass
    elif precision == "fp16":
        onnx_model = convert_float_to_float16(
            onnx_model,
            min_positive_val=1e-7,
            max_finite_val=1e4,
            keep_io_types=False,
            disable_shape_infer=False,
            op_block_list=None,
            node_block_list=None,
        )
    elif precision == "amp":
        logging.info("Performing automatic mixed precision. This may take a while..")
        _t0 = time.time()
        sample_np_inputs = as_numpy_arrays(sample_inputs)
        sample_feed_dict = _create_feed_dict(input_names, sample_inputs)
        onnx_model = auto_convert_mixed_precision(
            onnx_model,
            sample_feed_dict,
            validate_fn,
            rtol=rtol,
            atol=atol,
            keep_io_types=True,
        )
        _t1 = time.time()
        _take = round((_t1 - _t0), 4)
        logging.info("Done AMP conversion. Take %f seconds", _take)
    elif precision == "int8":
        raise NotImplementedError()

    ##### SAVING #####
    onnx.checker.check_model(onnx_model)
    save_dir = os.path.dirname(save_path)
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)
    if os.path.isfile(save_path):
        logging.warn("Save path %s is existed. Overwriting..", save_path)
    onnx.save_model(onnx_model, save_path)

    # Recheck
    print("Starting Torch-ONNX output comparation..")
    # onnx infer
    onnx_session = onnxruntime.InferenceSession(save_path)

    sample_onnx_outputs = onnx_session.run(
        None,
        _create_feed_dict(
            input_names, [inp.cpu().numpy() for inp in sample_torch_inputs]
        ),
    )

    sample_torch_outputs = as_numpy_arrays(sample_torch_outputs)
    # compare
    for torch_output, onnx_output in zip(sample_torch_outputs, sample_onnx_outputs):
        try:
            np.testing.assert_allclose(torch_output, onnx_output, rtol=rtol, atol=atol)
        except Exception as e:
            print(e)
    print("Done Torch-Onnx output comparation.")

    # AMP
    return onnx_model


def onnx2trt(
    onnx_file_path,
    engine_file_path,
    input_shape,
    precision="fp16",  # Options: "fp32", "fp16", "int8", "mixed"
    min_batch=1,
    opt_batch=1,
    max_batch=1,
    enable_dynamic_batching=True,
    workspace_size=1 << 30,  # 1GB default workspace
    calibrator=None,  # For INT8 quantization
):
    """
    Convert ONNX model to TensorRT engine.

    Args:
        onnx_file_path: Path to input ONNX file
        engine_file_path: Path to output TensorRT engine file
        input_shape: Input tensor shape without batch dimension, e.g (C, H, W)
        precision: Precision mode ("fp32", "fp16", "int8", "mixed")
        min_batch, opt_batch, max_batch: Batch size range for optimization
        enable_dynamic_batching: Whether to enable dynamic batch sizes
        workspace_size: Workspace memory size in bytes
        calibrator: INT8 calibrator object (required for INT8 precision)

    Returns:
        TensorRT engine object or None if failed
    """
    assert os.path.exists(onnx_file_path), f"ONNX file not found: {onnx_file_path}"

    # Validate precision and calibrator
    precision = precision.lower()
    if precision == "int8" and calibrator is None:
        print(
            "WARNING: INT8 precision requires a calibrator for accuracy. Consider using FP16 instead."
        )

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:

        # Create builder configuration
        config = builder.create_builder_config()

        # Set workspace size
        if workspace_size is not None:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
            print(f"Workspace size set to: {workspace_size / (1024**3):.2f} GB")

        # Set precision flags
        if precision == "fp16":
            if builder.platform_has_fast_fp16:
                print("CONVERTING WITH PRECISION FP16")
                config.set_flag(trt.BuilderFlag.FP16)
            else:
                print("WARNING: Platform does not support fast FP16. Using FP32.")
                precision = "fp32"

        elif precision == "int8":
            if builder.platform_has_fast_int8:
                print("CONVERTING WITH PRECISION INT8")
                config.set_flag(trt.BuilderFlag.INT8)
                if calibrator is not None:
                    config.int8_calibrator = calibrator
                    print("INT8 calibrator provided")
                else:
                    print(
                        "WARNING: No calibrator provided for INT8. Results may be inaccurate."
                    )
            else:
                print("WARNING: Platform does not support fast INT8. Using FP32.")
                precision = "fp32"

        elif precision == "mixed":
            has_fp16 = builder.platform_has_fast_fp16
            has_int8 = builder.platform_has_fast_int8

            if has_fp16 and has_int8:
                print("CONVERTING WITH PRECISION MIXED (FP16 + INT8)")
                config.set_flag(trt.BuilderFlag.FP16)
                config.set_flag(trt.BuilderFlag.INT8)
                if calibrator is not None:
                    config.int8_calibrator = calibrator
            elif has_fp16:
                print("CONVERTING WITH PRECISION FP16 (INT8 not supported)")
                config.set_flag(trt.BuilderFlag.FP16)
            else:
                print("CONVERTING WITH PRECISION FP32 (FP16/INT8 not supported)")
        else:
            print("CONVERTING WITH PRECISION FP32 (DEFAULT)")

        # Parse ONNX model
        with open(onnx_file_path, "rb") as model:
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for i in range(parser.num_errors):
                    print(f"Parser error {i}: {parser.get_error(i)}")
                return None

        print(
            f"Successfully parsed ONNX model with {network.num_inputs} inputs and {network.num_outputs} outputs"
        )

        # Handle input shapes and optimization profiles
        if network.num_inputs > 1:
            print(
                f"WARNING: Model has {network.num_inputs} inputs. Only setting profile for first input."
            )

        input_tensor = network.get_input(0)
        input_name = input_tensor.name
        print(f"Input tensor name: {input_name}")
        print(f"Input tensor shape from ONNX: {input_tensor.shape}")

        if enable_dynamic_batching:
            # Create optimization profile for dynamic batching
            profile = builder.create_optimization_profile()

            # Validate batch size parameters
            if not (1 <= min_batch <= opt_batch <= max_batch):
                raise ValueError(
                    f"Invalid batch sizes: min_batch({min_batch}) <= opt_batch({opt_batch}) <= max_batch({max_batch}) must hold"
                )

            min_shape = (min_batch, *input_shape)
            opt_shape = (opt_batch, *input_shape)
            max_shape = (max_batch, *input_shape)

            print(f"Setting optimization profile:")
            print(f"  Min shape: {min_shape}")
            print(f"  Opt shape: {opt_shape}")
            print(f"  Max shape: {max_shape}")

            profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)
            config.add_optimization_profile(profile)
        else:
            # For static batching, ensure the network input shape is correct
            expected_shape = (opt_batch, *input_shape)
            if tuple(input_tensor.shape) != expected_shape:
                print(
                    f"WARNING: ONNX input shape {input_tensor.shape} doesn't match expected {expected_shape}"
                )

        # Additional optimization flags (optional but recommended)
        # config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)  # Uncomment for strict precision adherence

        # Build serialized engine
        print("Building TensorRT engine... This may take several minutes.")
        try:
            serialized_engine = builder.build_serialized_network(network, config)
        except Exception as e:
            print(f"ERROR: Exception during engine building: {e}")
            return None

        if serialized_engine is None:
            print("ERROR: Failed to build serialized engine.")
            return None

        print("Engine built successfully!")

        # Deserialize engine for validation and return
        with trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(serialized_engine)
            if engine is None:
                print("ERROR: Failed to deserialize engine")
                return None

        # Save engine to file
        try:
            with open(engine_file_path, "wb") as f:
                f.write(serialized_engine)
            print(f"Saved TensorRT engine to {engine_file_path}")

            # Print engine info
            print(f"Engine info:")
            print(
                f"  Number of bindings: {engine.num_bindings if hasattr(engine, 'num_bindings') else 'N/A'}"
            )
            print(f"  Number of IO tensors: {engine.num_io_tensors}")

        except Exception as e:
            print(f"ERROR: Failed to save engine file: {e}")
            return None

        return engine


# Example usage with proper error handling
def _convert_with_error_handling():
    """Example of how to use the conversion function with proper error handling."""

    onnx_path = "model.onnx"
    engine_path = "model.engine"
    input_shape = (3, 224, 224)  # C, H, W without batch dimension

    try:
        engine = onnx2trt(
            onnx_file_path=onnx_path,
            engine_file_path=engine_path,
            input_shape=input_shape,
            precision="fp16",
            min_batch=1,
            opt_batch=4,
            max_batch=8,
            enable_dynamic_batching=True,
            workspace_size=2 << 30,  # 2GB workspace
        )

        if engine is not None:
            print("Conversion successful!")
            return engine
        else:
            print("Conversion failed!")
            return None

    except Exception as e:
        print(f"Conversion error: {e}")
        return None
