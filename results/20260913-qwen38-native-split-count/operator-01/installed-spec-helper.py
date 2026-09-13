def _spec_decode_varlen_fwd(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    block_table,
    max_seqlen_k,
    k_descale,
    v_descale,
    softmax_scale,
    s_aux,
    window_size,
    softcap,
):
    """Run the split-K decode kernel once for a speculative-decode batch.

    Assumes a uniform query length: every sequence contributes the same
    ``q_len`` consecutive query tokens, laid out as ``q[i * q_len + j]`` for
    sequence ``i`` and position ``j``. Each (sequence, position) pair is
    treated as an independent single-token decode "sequence", so the whole
    batch runs in a single ``varlen_fwd`` launch. Returns the output tensor
    (same row order as ``q``).
    """
    batch = cu_seqlens_q.numel() - 1
    q_len = q.shape[0] // batch
    total = q.shape[0]  # batch * q_len

    # One decode query per (sequence, position) pair; the row order already
    # matches q, so the output needs no reordering.
    cu_seqlens_q_spec = torch.arange(total + 1,
                                     dtype=torch.int32,
                                     device=q.device)
    dummy_cu_seqlens_k = torch.zeros_like(cu_seqlens_q_spec)
    # Causal per-position KV length: query row i*q_len + j (position j within
    # the q_len window of sequence i) attends to seqused_k[i] - (q_len-1-j)
    # tokens.
    pos = torch.arange(q_len, device=q.device, dtype=seqused_k.dtype)
    seqused_k_spec = (seqused_k.view(batch, 1) -
                      (q_len - 1 - pos).view(1, q_len)).clamp_(min=1).reshape(
                          total).to(torch.int32)

    # Expand the block table so each pseudo-sequence points at its parent
    # sequence's blocks.
    block_table_spec = block_table.repeat_interleave(q_len, dim=0)

    out, _ = torch.ops._vllm_fa2_C.varlen_fwd(
        q,
        k,
        v,
        out,
        cu_seqlens_q_spec,
        dummy_cu_seqlens_k,
        seqused_k_spec,
        None,
        block_table_spec,
        None,  # alibi_slopes
        1,  # max_seqlen_q
        max_seqlen_k,
        0.0,  # dropout_p
        k_descale,
        v_descale,
        softmax_scale,
        s_aux,
        False,  # zero_tensors
        False,  # causal: handled by per-position seqused_k
        window_size[0],
        window_size[1],
        softcap,
        False,  # return_softmax
        None,  # gen
        None,  # num_splits (let the kernel pick via get_num_splits)
        False,  # is_mix_batch
        None,  # splits_per_seq
        None,  # work_list
    )
    return out
