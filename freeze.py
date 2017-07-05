from tensorflow.python.tools import freeze_graph

freeze_graph.freeze_graph(
    input_graph='models/graph.pb',
    input_saver=None,
    input_binary=True,
    input_checkpoint='models/checkpoint.ckpt',
    output_node_names='prediction',
    restore_op_name='save/restore_all',
    filename_tensor_name='save/Const:0',
    output_graph='models/freeze.pb',
    clear_devices=True,
    initializer_nodes="",
    variable_names_blacklist=""
)
