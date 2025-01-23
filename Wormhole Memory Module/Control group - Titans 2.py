class NeuralMemoryGatingWrapper(Module):
    def __init__(
        self,
        dim,
        attn: SegmentedAttention,
        neural_mem: NeuralMemory | None = None,
        gate_attn_output = True
    ):
        super().__init__()
        self.attn = attn
        self.neural_mem = neural_mem
        self.gate_attn_output = gate_attn_output

    def forward(
        self,
        seq,
        *args,
        **kwargs
    ):
        batch, seq_len = seq.shape[:2]
        mem = self.neural_mem

        if not exists(mem):
            return self.attn(seq, *args, **kwargs), 0.

        # initial retrieve, still should store first, it doesn't make sense not to, unless if all layers share the same neural memory

        retrieved, kv_aux_loss = mem(seq, return_aux_kv_loss = True)

        if not self.gate_attn_output:
            seq = seq + retrieved

        # attention

        attn_out, values = self.attn(seq, *args, **kwargs)

        if self.gate_attn_output:
            attn_out = attn_out * retrieved.sigmoid()

        return (attn_out, values), kv_aux_loss
