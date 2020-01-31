using Flux
using Flux: gate

"""
    batch_matvec(A::Tensor, X::Matrix)

for A ∈ ℝ^{n × m × d}, X ∈ ℝ^{n × d}. Performs ``n`` matrix-vec multiplications

    A[i,:,:] * X[i,:]

for ``i \\in 1,\\ldots,n``. This is performed via expansion of `X` and can be
efficiently evaluated using BLAS operations, which (I have assumed!) will often
be faster than a loop, especially in relation to AD and GPUs.
"""
batch_matvec(A::AbstractArray{T,3}, X::AbstractArray{T,2}) where T =
    dropdims(sum(A .* unsqueeze(X, 1), dims=2), dims=2)

Flux.gate(x::AbstractArray{T,3}, h, n) where T = x[gate(h,n),:,:]
Flux.gate(x::AbstractArray{T,4}, h, n) where T = x[gate(h,n),:,:,:]


mutable struct MTGRU_NoU{F}
    N::Int
    G::F
end

mutable struct BatchedGRUCell_NoU{A,W}
    Wh::A
    b::W
    h::W
end

"""
    MTGRU_NoU(N, G)
Multi-task GRU (Gated Recurrent Unit) model. Produces a GRU layer
for each value of `z` ∈ Rᵈ which depends on the map `G` inside
the MTGRU. N = out dimension, G: Rᵈ->R^{nLSTM}. z can be batched
as `z` ∈ R^{d × nbatch}.

In order to simplify the implementation, there is no input depen-
dent evolution (hence no `U`s, and hence no `Wi`). It is
straight-forward to extend to this case, should it be reqd.
"""
function (m::MTGRU_NoU)(z, h=nothing)
    N, nB = m.N, size(z, 2)
    λ = m.G(z)
    Wh = reshape(λ[1:N*N*3,:], N*3, N, nB)
    b = λ[N*N*3+1:(N+1)*N*3, :]
    open_forget_gate = zero(b)   # no backprop, and hence not tracked, even if `b` is.
    open_forget_gate[gate(N, 2), :] .= 1
    b += open_forget_gate
    h = something(h, Flux.param(zero(b)))

    Flux.Recur(BatchedGRUCell_NoU(Wh, b, h))
end

Flux.@treelike MTGRU_NoU
Base.show(io::IO, l::MTGRU_NoU) =
  print(io, "Multitask-GRU(", l.N, ", ", typeof(l.G), ", no inputs)")


"""
    BatchedGRUCell_NoU(Wh, b, h)
Multi-task GRU Cell (which takes no input).
`Wh`, `b`, `h` are the (concatenated) transformation matrices,
offsets, and initial hidden state respectively.

Each time the cell is called (no arguments) it will perform a one-
step evolution of the hidden state and return the current value.
The cell is implemented in batch-mode, and the final dimension of
each quantity is the batch index.
"""
function (m::BatchedGRUCell_NoU)(h, x=nothing)
  b, o = m.b, size(h, 1)
  gh = batch_matvec(m.Wh, h)
  r = σ.(gate(gh, o, 1) .+ gate(b, o, 1))
  z = σ.(gate(gh, o, 2) .+ gate(b, o, 2))
  h̃ = tanh.(r .* gate(gh, o, 3) .+ gate(b, o, 3))
  h′ = (1 .- z).*h̃ .+ z.*h
  return h′, h′
end

Flux.@treelike BatchedGRUCell_NoU
Base.show(io::IO, l::BatchedGRUCell_NoU) =
  print(io, "Batched GRU Cell(dim=", size(l.Wh,2), ", batch=", size(l.Wh,3), ", no inputs)")
Flux.hidden(m::BatchedGRUCell_NoU) = m.h
