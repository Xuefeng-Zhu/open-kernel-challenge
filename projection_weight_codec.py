"""CPU-only feasibility codec for lossless static BF16 weight storage.

No activations, inference outputs, GPU execution, or performance claims. A future
native decoder must reconstruct the identical BF16 bit pattern before arithmetic.

Serialization v1 (row-major contiguous tensors; NB=ceil(K/256)):
  sign_mantissa:   uint8 [N,NB,256], original sign in bit7, fraction in bits0..6
  exponent_delta:  uint8 [N,NB,128], even-K delta low nibble, odd-K high nibble
  exponent_base:   uint8 [N,NB], minimum exponent over valid elements
  escape_index:    int32 [N,NB], -1 normal, otherwise row of escape_exponents
  escape_exponents:uint8 [E,256], complete exponents for range>15 blocks

Original shape and format/block size are ordinary scalar metadata. Last blocks
are padded; the original K determines valid output columns. Escaped blocks still
occupy their fixed delta slots (zero-filled), and all bytes count in the model.
"""
import torch

FORMAT = "bf16-signmantissa-expdelta4-k256-v1"
BLOCK_K = 256
STREAM_DTYPES = {
    "sign_mantissa": torch.uint8,
    "exponent_delta": torch.uint8,
    "exponent_base": torch.uint8,
    "escape_index": torch.int32,
    "escape_exponents": torch.uint8,
}


def encode_weights(weights):
    """Encode only CPU BF16 matrix bits; input is never modified."""
    if weights.device.type != "cpu":
        raise ValueError("This feasibility codec is CPU-only; no GPU path is validated")
    if weights.dtype != torch.bfloat16 or weights.ndim != 2:
        raise ValueError("Expected a two-dimensional BF16 weight matrix")
    n, k = weights.shape
    if n <= 0 or k <= 0:
        raise ValueError("Weight dimensions must be positive")
    nb = (k + BLOCK_K - 1)//BLOCK_K
    # Read bit patterns, including NaN payloads, rather than converting values.
    bits = weights.contiguous().view(torch.int16).to(torch.int32) & 0xffff
    padded = torch.zeros((n, nb*BLOCK_K), dtype=torch.int32)
    padded[:, :k] = bits
    padded = padded.view(n, nb, BLOCK_K)
    exp = (padded >> 7) & 0xff
    valid = (torch.arange(nb*BLOCK_K).view(1, nb, BLOCK_K) < k).expand(n, -1, -1)
    base = exp.masked_fill(~valid, 255).amin(-1)
    high = exp.masked_fill(~valid, 0).amax(-1)
    escaped = high-base > 15
    # Padding is ignored by decode, but use zero deltas there for an unambiguous format.
    delta = torch.where(valid & ~escaped[..., None], exp-base[..., None], 0)
    packed_delta = (delta[..., 0::2] | (delta[..., 1::2] << 4)).to(torch.uint8)
    sign_mantissa = (((padded >> 8) & 0x80) | (padded & 0x7f)).to(torch.uint8)
    escape_exponents = exp.reshape(-1, BLOCK_K)[escaped.reshape(-1)].to(torch.uint8).contiguous()
    indices = torch.full((n, nb), -1, dtype=torch.int32)
    indices[escaped] = torch.arange(escape_exponents.shape[0], dtype=torch.int32)
    return dict(format=FORMAT, original_shape=(n, k), block_k=BLOCK_K,
                sign_mantissa=sign_mantissa.contiguous(), exponent_delta=packed_delta.contiguous(),
                exponent_base=base.to(torch.uint8).contiguous(), escape_index=indices,
                escape_exponents=escape_exponents)


def _validate(encoded):
    if encoded.get("format") != FORMAT or encoded.get("block_k") != BLOCK_K:
        raise ValueError("Unsupported codec format/block size")
    shape = encoded.get("original_shape")
    if not isinstance(shape, (tuple, list)) or len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError("Invalid original shape")
    n, k = shape
    nb = (k+BLOCK_K-1)//BLOCK_K
    for name, dtype in STREAM_DTYPES.items():
        value = encoded[name]
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != dtype or not value.is_contiguous():
            raise ValueError("Invalid CPU contiguous stream: " + name)
    expected = dict(sign_mantissa=(n,nb,BLOCK_K), exponent_delta=(n,nb,BLOCK_K//2),
                    exponent_base=(n,nb), escape_index=(n,nb))
    for name, wanted in expected.items():
        if tuple(encoded[name].shape) != wanted:
            raise ValueError("Invalid stream shape: " + name)
    esc = encoded["escape_exponents"]
    if esc.ndim != 2 or esc.shape[1] != BLOCK_K:
        raise ValueError("Invalid escape stream shape")
    idx = encoded["escape_index"]
    if torch.any(idx < -1) or torch.any(idx >= esc.shape[0]):
        raise ValueError("Escape index out of range")
    return n, k, nb


def decode_weights(encoded):
    """Restore original BF16 bits, including signed zero/subnormal/Inf/NaN."""
    n, k, nb = _validate(encoded)
    packed = encoded["exponent_delta"].to(torch.int32)
    delta = torch.empty((n,nb,BLOCK_K), dtype=torch.int32)
    delta[..., 0::2] = packed & 15
    delta[..., 1::2] = packed >> 4
    exponent = encoded["exponent_base"].to(torch.int32)[..., None] + delta
    idx = encoded["escape_index"]
    if encoded["escape_exponents"].shape[0]:
        full = encoded["escape_exponents"][idx.clamp_min(0).long()].to(torch.int32)
        exponent = torch.where((idx >= 0)[..., None], full, exponent)
    sign_man = encoded["sign_mantissa"].to(torch.int32)
    bits = (sign_man & 0x7f) | ((sign_man & 0x80) << 8) | (exponent << 7)
    return bits.reshape(n,nb*BLOCK_K)[:, :k].contiguous().to(torch.int16).view(torch.bfloat16)


def storage_report(encoded):
    """Exact allocated stream bytes; scalar metadata is separately enumerated."""
    n, k, nb = _validate(encoded)
    streams = {name: dict(shape=list(encoded[name].shape), dtype=str(dtype),
                         bytes=encoded[name].numel()*encoded[name].element_size())
               for name,dtype in STREAM_DTYPES.items()}
    total = sum(v["bytes"] for v in streams.values())
    # Fixed portable scalar header: uint32 version/block_k + uint64 N/K/E = 32B.
    header = 32
    original = n*k*2
    return dict(format=FORMAT, original_shape=[n,k], blocks=n*nb,
                escape_blocks=encoded["escape_exponents"].shape[0],
                escape_fraction=encoded["escape_exponents"].shape[0]/(n*nb),
                streams=streams, tensor_bytes=total, scalar_header_bytes=header,
                encoded_bytes=total+header, original_bytes=original,
                byte_reduction=1-(total+header)/original,
                qualification="Exact stream storage plus specified 32-byte portable scalar header; Python objects/allocator overhead/original fallback weights excluded. Not measured traffic or speedup.")
