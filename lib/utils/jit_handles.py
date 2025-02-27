from collections import Counter, OrderedDict
from numpy import prod

import torch

# A list that contains ignored operations.
_IGNORED_OPS = [
    'aten::Int',
    'aten::__and__',
    'aten::arange',
    'aten::cat',
    'aten::clamp',
    'aten::clamp_',
    'aten::contiguous',
    'aten::copy_',
    'aten::detach',
    'aten::empty',
    'aten::eq',
    'aten::floor_divide',
    'aten::ScalarImplicit',
    'aten::expand',
    'aten::flatten',
    'aten::floor',
    'aten::ge',
    'aten::chunk',
    'aten::split',
    'aten::stack',
    'aten::full',
    'aten::gt',
    'aten::index',
    'aten::index_put_',
    'aten::max',
    'aten::nonzero',
    'aten::permute',
    'aten::remainder',
    'aten::reshape',
    'aten::select',
    'aten::size',
    'aten::slice',
    'aten::split_with_sizes',
    'aten::squeeze',
    'aten::t',
    'aten::to',
    'aten::transpose',
    'aten::unsqueeze',
    'aten::unsqueeze_',
    'aten::constant_pad_nd',
    'aten::view',
    'aten::zeros',
    'aten::zeros_like',
    'prim::Constant',
    'prim::Int',
    'prim::ListConstruct',
    'prim::ListUnpack',
    'prim::NumToTensor',
    'prim::TupleConstruct',
    'prim::ImplicitTensorToNum',
]


def get_jit_model_analysis(model, inputs, ops_handles,):
    """
    Given a model, the inputs and the handles for each operation, return the
    results for the model analysis.

    Args:
        model (nn.Module): The model for torch script to trace.
        inputs (tuple): Inputs that are passed to `model` to trace. Inputs need
            to be in a tuple.
        ops_handles (dict[str, Callable]): A dictionary of handles
            for model analysis.

    Returns:
        tuple[Counter[str], Counter[str]]: A counter that
            contains the results of per operation analysis of the model and a
            Counter of ignored operations.
    """
    # Torch script does not support parallel torch models.
    if isinstance(model, (torch.nn.parallel.distributed.DistributedDataParallel, torch.nn.DataParallel)):
        model = model.module  # pyre-ignore

    # Compatibility with torch.jit.
    if hasattr(torch.jit, 'get_trace_graph'):
        trace, _ = torch.jit.get_trace_graph(model, inputs)
        trace_nodes = trace.graph().nodes()
    else:
        trace, _ = torch.jit._get_trace_graph(model, inputs)
        trace_nodes = trace.nodes()

    skipped_ops = Counter()
    total_count = Counter()

    for node in trace_nodes:
        kind = node.kind()
        if kind not in ops_handles.keys():
            # If the operation is not in _IGNORED_OPS, count skipped operations.
            if kind not in _IGNORED_OPS:
                skipped_ops[kind] += 1
            continue

        handle_count = ops_handles.get(kind, None)
        if handle_count is None:
            continue
        # pyre-ignore
        inputs, outputs = list(node.inputs()), list(node.outputs())
        op_count = handle_count(inputs, outputs)
        total_count += op_count

    return total_count, skipped_ops


def generic_activation_jit(op_name,):
    """
    This method return a handle that counts the number of activation from the
    output shape for the specified operation.

    Args:
        op_name (str): The name of the operation.

    Returns:
        Callable: An activation handle for the given operation.
    """

    def _generic_activation_jit(outputs):
        """
        This is a generic jit handle that counts the number of activations for any
        operation given the output shape.

        Args:
            outputs (list(torch._C.Value)): The output shape in the form of a list
                of jit object.

        Returns:
            int: Total number of activations for each operation.
        """
        out_shape = get_shape(outputs[0])
        ac_count = prod(out_shape)
        return ac_count

    return lambda inputs, outputs: Counter({op_name: _generic_activation_jit(outputs)})


def get_shape(val):
    """
    Get the shapes from a jit value object.

    Args:
        val (torch._C.Value): jit value object.

    Returns:
        list(int): return a list of ints.
    """
    if val.isCompleteTensor():  # pyre-ignore
        return val.type().sizes()  # pyre-ignore
    else:
        raise ValueError()


def addmm_flop_jit(inputs, outputs):
    """
    This method counts the flops for fully connected layers with torch script.

    Args:
        inputs (list(torch._C.Value)): The input shape in the form of a list of
            jit object.
        outputs (list(torch._C.Value)): The output shape in the form of a list
            of jit object.

    Returns:
        Counter: A Counter dictionary that records the number of flops for each
            operation.
    """
    # Count flop for nn.Linear
    # inputs is a list of length 3.
    input_shapes = [get_shape(v) for v in inputs[1:3]]
    # input_shapes[0]: [batch size, input feature dimension]
    # input_shapes[1]: [batch size, output feature dimension]
    assert len(input_shapes[0]) == 2
    assert len(input_shapes[1]) == 2
    batch_size, input_dim = input_shapes[0]
    output_dim = input_shapes[1][1]
    flop = batch_size * input_dim * output_dim
    flop_counter = Counter({'addmm': flop})
    return flop_counter


def conv_flop_count(x_shape, w_shape, out_shape):
    """
    This method counts the flops for convolution. Note only multiplication is
    counted. Computation for addition and bias is ignored.

    Args:
        x_shape (list(int)): The input shape before convolution.
        w_shape (list(int)): The filter shape.
        out_shape (list(int)): The output shape after convolution.
    Returns:
        Counter: A Counter dictionary that records the number of flops for each
            operation.
    """
    batch_size, Cin_dim, Cout_dim = x_shape[0], w_shape[1], out_shape[1]
    out_size = prod(out_shape[2:])
    kernel_size = prod(w_shape[2:])
    flop = batch_size * out_size * Cout_dim * Cin_dim * kernel_size
    flop_counter = Counter({'conv': flop})
    return flop_counter


def conv_flop_jit(inputs, outputs):
    """
    This method counts the flops for convolution using torch script.

    Args:
        inputs (list(torch._C.Value)): The input shape in the form of a list of
            jit object before convolution.
        outputs (list(torch._C.Value)): The output shape in the form of a list
            of jit object after convolution.

    Returns:
        Counter: A Counter dictionary that records the number of flops for each
            operation.
    """
    # Inputs of Convolution should be a list of length 12. They represent:
    # 0) input tensor, 1) convolution filter, 2) bias, 3) stride, 4) padding,
    # 5) dilation, 6) transposed, 7) out_pad, 8) groups, 9) benchmark_cudnn,
    # 10) deterministic_cudnn and 11) user_enabled_cudnn.
    #assert len(inputs) == 12
    x, w = inputs[:2]
    x_shape, w_shape, out_shape = (get_shape(x), get_shape(w), get_shape(outputs[0]))
    return conv_flop_count(x_shape, w_shape, out_shape)


def einsum_flop_jit(inputs, outputs):
    """
    This method counts the flops for the einsum operation. We currently support
    two einsum operations: 'nct,ncp->ntp' and 'ntg,ncg->nct'.

    Args:
        inputs (list(torch._C.Value)): The input shape in the form of a list of
            jit object before einsum.
        outputs (list(torch._C.Value)): The output shape in the form of a list
            of jit object after einsum.

    Returns:
        Counter: A Counter dictionary that records the number of flops for each
            operation.
    """
    # Inputs of einsum should be a list of length 2.
    # Inputs[0] stores the equation used for einsum.
    # Inputs[1] stores the list of input shapes.
    assert len(inputs) == 2
    equation = inputs[0].toIValue()  # pyre-ignore
    # Get rid of white space in the equation string.
    equation = equation.replace(' ', '')
    # Re-map equation so that same equation with different alphabet
    # representations will look the same.
    letter_order = OrderedDict((k, 0) for k in equation if k.isalpha()).keys()
    mapping = {ord(x): 97 + i for i, x in enumerate(letter_order)}
    equation = equation.translate(mapping)
    input_shapes_jit = inputs[1].node().inputs()  # pyre-ignore
    input_shapes = [get_shape(v) for v in input_shapes_jit]

    if equation == 'abc,abd->acd':
        n, c, t = input_shapes[0]
        p = input_shapes[-1][-1]
        flop = n * c * t * p
        flop_counter = Counter({'einsum': flop})
        return flop_counter
    elif equation == 'abc,adc->adb':
        n, t, g = input_shapes[0]
        c = input_shapes[-1][1]
        flop = n * t * g * c
        flop_counter = Counter({'einsum': flop})
        return flop_counter
    else:
        raise NotImplementedError('Unsupported einsum operation.')


def matmul_flop_jit(inputs, outputs):
    """
    This method counts the flops for matmul.

    Args:
        inputs (list(torch._C.Value)): The input shape in the form of a list of
            jit object before matmul.
        outputs (list(torch._C.Value)): The output shape in the form of a list
            of jit object after matmul.

    Returns:
        Counter: A Counter dictionary that records the number of flops for each
            operation.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two matrices.
    input_shapes = [get_shape(v) for v in inputs]
    assert len(input_shapes) == 2
    assert len(input_shapes[1]) == 2
    assert input_shapes[0][-1] == input_shapes[1][0]
    batch_dim = input_shapes[0][0]
    m1_dim, m2_dim = input_shapes[1]
    flop = m1_dim * m2_dim * batch_dim
    flop_counter = Counter({'matmul': flop})
    return flop_counter


def batchnorm_flop_jit(inputs, outputs):
    """
    This method counts the flops for batch norm.

    Args:
        inputs (list(torch._C.Value)): The input shape in the form of a list of
            jit object before batch norm.
        outputs (list(torch._C.Value)): The output shape in the form of a list
            of jit object after batch norm.

    Returns:
        Counter: A Counter dictionary that records the number of flops for each
            operation.
    """
    # Inputs[0] contains the shape of the input.
    input_shape = get_shape(inputs[0])
    assert 2 <= len(input_shape) <= 5
    flop = prod(input_shape) * 4
    flop_counter = Counter({'batchnorm': flop})
    return flop_counter
