module mtmodel

using Flux, ArgCheck, StatsBase
using Flux: gate
using Flux.Tracker: istracked

using ..modelutils
import ..modelutils: load!, randn_repar, mlp
export load!

const D_IM = 32


################################################################################
##                                                                            ##
##              Define a basic MTGRU model (with no inputs)                   ##
##                                                                            ##
################################################################################

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
    h = something(h, istracked(λ) ? Flux.param(zero(b)) : zero(b))

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


################################################################################
##                                                                            ##
##           Define *THE* multi-task model used in our experiments            ##
##                                                                            ##
################################################################################

"""
    MTSeqModel_E3(enc_tform, init_enc, init_post, mt_enc, mt_post, chaos_enc, chaos_post,
        x0decoder, mtbias_deocder, gen_model, d_y, mtbias_only)

A sequence-to-sequence model 𝒴→𝒴 which imbibes `T_enc` steps of the sequence,
and predicts `T_steps` (including the original `T_enc`), but unlike the
`Seq2SeqModel`, the `MTSeqModel_E3` can *modulate* its prediction based on
hierarchical 'multi-task' variables. (The `E3` suffix refers to the number of
encoders: previously I had `E1`, `E2` and `E3` types, but this seems redundant.)

As for the `Seq2SeqModel`, the posterior over the initial state of the generator
is learned variationally via the `init_enc` RNN with the final state passed to
the `init_post` layer which generates the mean and variance of the latent
posterior. Since the posterior is typically lower dimensional than the state of
the generator network, the latent variable is expanded via the `x0decoder`
network into the state of the `generator`. The MT variable is encoded with a
full-length sequence encoding via the `mt_enc` network, and the final state is
passed to the `mt_post` layer to generate the posterior mean and variance. A
similar set of operations is performed for the more local 'chaos variable'
using the `chaos_enc` and `chaos_post` networks. The 'chaos variable' encodes
sequence variation due to inherent unpredictability of a system, perhaps due to
chaos. This allows different long term predictions from the same state. The MT
and chaos variables enter the generator network either:

1. by predicting all parameters of the network via a `MTGRU_NoU` construction
(see docstring).
2. by forming a latent (constant) input at each time step, which functions as a
variable bias of the generator recurrent cell. This may be directly from the
latent variables, or via a decoder (`mtbias_decoder`).

Once these variables are sampled, the `generator` is iterated in an open-loop
fashion for `T_steps`, similar to the `Seq2SeqModel`. The penultimate slot `d`
is a record of the dimension of 𝒴, used currently just for `show`; and the
`mtbias_only` variable dictates whether either approach (1) or (2) is used as
above.

--------------------
This type can be called via
```julia
    m = Seq2SeqModel(...)
    m(y::Vector)
```
where `y` is a vector of slices of the sequence for the encoder. This means each
`y[i]` is an `AbstractMatrix` of size ``d \\times n_{\\text{batch}}`.

Options include `T_enc`, `T_steps` (already discussed), `reset` (which performs
a state reset before execution; true by default), and `stoch`, whether to
randomly sample the latent state, or use the state mean (default true).
"""

struct MTSeq_CNN end
struct MTSeq_Single end
struct MTSeq_Double end

struct MTSeqModel_E3{U,A,B,V,W,S,N,M}
    enc_tform::U
    init_enc::A
    init_post::B
    mt_enc::A
    mt_post::B
    chaos_enc::A
    chaos_post::B
    x0decoder::V
    mtbias_decoder::W
    generator::S
    dec_tform::N
    model_type::M
    d::Int64
    mtbias_only::Bool
end

Flux.@treelike MTSeqModel_E3
unpack(m::MTSeqModel_E3) = Flux.children(m)[1:11] # unpack struct; exclude `d`, `mtbias_only`.

modelutils.load!(m::MTSeqModel_E3, fname::String) = load!(Flux.params(m), fname)

model_type(m::MTSeqModel_E3{U,A,B,V,W,S,N,M}) where {U,A,B,V,W,S,N,M} = M

function Base.show(io::IO, l::MTSeqModel_E3)
    d_x0 = size(l.init_post.Dense1.W, 1)
    d_mt = size(l.mt_post.Dense1.W, 1)
    d_c = size(l.chaos_post.Dense1.W, 1)
    out_type = Dict(MTSeq_CNN=>"CNN", MTSeq_Single=>"Deterministic", MTSeq_Double=>"Probabilistic")[model_type(l)]
    mttype = l.mtbias_only ? "Bias-only" : "Full-MT Recurrent Cell"
    print(io, "MTSeqModel_E3(", mttype, ", d_x0=", d_x0, ", d_mt=", d_mt, ", d_chaos=", d_c, ", ", out_type, ")")
end


function (m::MTSeqModel_E3)(x0::AbstractVecOrMat, z::AbstractVecOrMat, c::AbstractVecOrMat; T_steps=70)

    x0decoder, mtbias_decoder, gen_model, dec_tform = unpack(m)[8:11]

    # Generative model
    # --------------------------------------------------
    if m.mtbias_only
        gen_model.state = x0decoder(x0)
        mtbias = mtbias_decoder(vcat(z, c))   # if linear, this is just the identity ∵ GRUCell.Wi
        return [dec_tform(gen_model(mtbias)) for t in 1:T_steps]
    else
        # Get GRU models from samples from the MT GRU model
        posterior_grus = gen_model(vcat(z, c)) # output: BatchedGRU (def. above)
        # Set state (batch-wise) to x0 sample
        posterior_grus.state = x0decoder(x0)  # 2×linear d_x0 × nbatch → d_x × nbatch
        # Run generative model
        return [dec_tform(posterior_grus()) for t in 1:T_steps]
    end
end


function (m::MTSeqModel_E3)(y::AbstractVector, yfull::AbstractVector; T_steps=70, T_enc=10, stoch=true)
    x0, z, c = posterior_samples(m, y, yfull; T_steps=T_steps, T_enc=T_enc, stoch=stoch)[1]
    return m(x0, z, c; T_steps=T_steps)
end


function posterior_samples(m::MTSeqModel_E3, y::AbstractVector, yfull::AbstractVector;
    T_steps=70, T_enc=10, stoch=true, x0=nothing, z=nothing, c=nothing)

    enc_tform, init_enc, init_post, mt_enc, mt_post, chaos_enc, chaos_post = unpack(m)[1:7]

    # RNN 1: Amortized inference for initial state, h0, of generator
    if x0 === nothing
        enc_yslices = [enc_tform(yy) for yy in y]
        x0, μ_x0, σ_x0 = _posterior_sample(init_enc, init_post, enc_yslices, T_enc, stoch)
    else
        μ_x0, σ_x0 = nothing, nothing
    end

    # RNN 2: Amortized inference for z
    # Technically this should be conditioned on the sample x₀, nevertheless, the posterior
    # of x0 is usually very tight, and the `mt_enc` has access to the same information.
    if z === nothing
        enc_fullseqs = [enc_tform(yy) for yy in yfull]
        z, μ_z, σ_z = _posterior_sample(mt_enc, mt_post, enc_fullseqs, length(yfull), stoch)
    else
        μ_z, σ_z = nothing, nothing
    end

    # RNN 3: Encode the chaos
    # Want a lowdim rep. of departure from expected trajected.
    if c === nothing
        (x0 === nothing) && (enc_yslices = [enc_tform(yy) for yy in y])
        c, μ_c, σ_c = _posterior_sample(chaos_enc, chaos_post, enc_yslices, length(y),
            stoch, vcat(x0, z))  # note conditioning on (x0, z) here.
    else
        μ_c, σ_c = nothing, nothing
    end
    return (x0,z,c), (μ_x0, μ_z, μ_c), (σ_x0, σ_z, σ_c)
end


function create_model(d_x, d_x0, d_y, d_enc_state, d_mt, d_chaos; encoder=:GRU,
    cnn=false, out_heads=1, d_hidden=d_x, mtbias_only=false, d_hidden_mt=32,
    mt_is_linear=true, decoder_fudge_layer=false)

    @assert !(out_heads > 1 && cnn) "cannot have multiple output heads and CNN."
    @argcheck out_heads in 1:2

    # ENCODER TRANSFORM FROM OBS SPACE => ENC SPACE
    if cnn
        N_FILT=32
        d_conv_result = 4*4*N_FILT
        tform_enc = Chain(
            Conv((3, 3), 2=>N_FILT, relu, stride=2, pad=1),        # out: 16 × 16 × 32 × nbatch
            Conv((3, 3), N_FILT=>N_FILT, relu, stride=2, pad=1),   # out: 8 × 8 × 32 × nbatch
            Conv((3, 3), N_FILT=>N_FILT, relu, stride=2, pad=1),   # out: 4 × 4 × 32 × nbatch
            x->reshape(x, d_conv_result, :),
            Dense(d_conv_result, d_hidden, swish)
        )
        d_rnn_in = d_hidden
        model_type = MTSeq_CNN()
    else
        tform_enc = identity
        d_rnn_in = d_y
        model_type = out_heads == 1 ? MTSeq_Single() : MTSeq_Double()
    end

    # 3× ENCODERS for inference of
    # (a) initial state (init_enc)
    # (b) sequence level MT variable (mt_enc)
    # (c) local variability due to sensitivity/chaos (mt_chaos)
    @argcheck encoder in [:LSTM, :GRU, :Bidirectional]
    rnn_constructor = Dict(:LSTM=>LSTM, :GRU=>GRU, :Bidirectional=>BRNNenc)[encoder]
    init_enc = rnn_constructor(d_rnn_in, d_enc_state)             # (a)
    mt_enc = rnn_constructor(d_rnn_in, d_enc_state)               # (b)
    chaos_enc = rnn_constructor(d_rnn_in+d_mt+d_x0, d_enc_state)  # (c)
    (encoder == :Bidirectional) && (d_enc_state *= 2)


    # POSTERIOR OVER LVM CORR. TO (a), (b), (c)
    init_post, mt_post, chaos_post = [MultiDense(Dense(d_enc_state, d, identity), Dense(d_enc_state, d, σ))
        for d in (d_x0, d_mt, d_chaos)]
    init_post.Dense2.b.data .= -2   # initialize posteriors to be low variance
    mt_post.Dense2.b.data .= -2
    chaos_post.Dense2.b.data .= -2

    # decode from LV (a) --> size of generative hidden state
    x0decoder = Dense(d_x0, d_x, identity)

    # -------------- GENERATIVE MODEL -----------------
    ###################################################

    if !mtbias_only
        d_out = d_x*(d_x+1)*3
        par_gen_net = mt_is_linear ? mlp(d_mt+d_chaos, d_out) : mlp(d_mt+d_chaos, d_hidden_mt, d_out; activation=tanh)
        gen_rnn = MTGRU_NoU(d_x, par_gen_net)
        mtbias_decoder = identity
    else
        mtbias_decoder = mt_is_linear ? identity : Dense(d_mt+d_chaos, d_hidden_mt, swish)
        _d_in = mt_is_linear ? d_mt+d_chaos : d_hidden_mt
        gen_rnn = GRU(_d_in, d_x)
    end

    if decoder_fudge_layer
        # accidental additional layer in decoder for full MT-model.
        # (This is really low capacity. 64->64 unit relu; if anything, will probably reduce performance.)
        fudge_layer = Dense(d_x, d_x, relu)
    else
        fudge_layer = identity
    end

    if out_heads == 1 && !cnn
        decoder = Chain(Dense(d_x, d_hidden, relu), Dense(d_hidden, d_y, identity))
    elseif out_heads == 2 && !cnn
        decoder = Chain(Dense(d_x, d_hidden, relu), MultiDense(Dense(d_hidden, d_y, identity), Dense(d_hidden, d_y, identity)))
    else    # cnn
        decoder = Chain(
            fudge_layer,
            Dense(d_x, d_conv_result, identity),
            x->reshape(x, 4, 4, N_FILT, :),                               # out: 4 × 4 × 32 × nbatch
            ConvTranspose((3,3), N_FILT=>N_FILT, relu, stride=2),         # out: 9 × 9 × 16 × nbatch
            ConvTranspose((3,3), N_FILT=>N_FILT, relu, stride=2, pad=1),  # out: 17 × 17 × 8 × nbatch
            ConvTranspose((3,3), N_FILT=>2, identity, stride=2, pad=1),   # out: 33 × 33 × 1 × nbatch
            x->x[1:D_IM, 1:D_IM, :, :]   # out: 32 × 32 × 2 × nbatch
        )
    end

    return MTSeqModel_E3(tform_enc, init_enc, init_post, mt_enc, mt_post, chaos_enc, chaos_post,
        x0decoder, mtbias_decoder, gen_rnn, decoder, model_type, d_y, mtbias_only)
end

function _posterior_sample(enc, dec, input, T_max, stochastic=true, input_cat=nothing)
    Flux.reset!(enc)
    if input_cat === nothing
        for tt = 1:T_max; enc(input[tt,:,:]); end
    else
        for tt = 1:T_max; enc(vcat(input[tt,:,:], input_cat)); end
    end
    μ_, σ_ = dec(enc.state)
    n, d = size(μ_)
    smp = μ_ + randn_repar(σ_, n, d, stochastic)
    return smp, μ_, σ_
end

function _posterior_sample(enc, dec, input::Vector, T_max, stochastic=true, input_cat=nothing)
    Flux.reset!(enc)
    if input_cat === nothing
        for tt = 1:T_max; enc(input[tt]); end
    else
        for tt = 1:T_max; enc(vcat(input[tt], input_cat)); end
    end
    μ_, σ_ = dec(enc.state)
    n, d = size(μ_)
    smp = μ_ + randn_repar(σ_, n, d, stochastic)
    return smp, μ_, σ_
end

################################################################################
##                                                                            ##
##                        Objective(s) / Loss Function(s)                     ##
##                                                                            ##
################################################################################

function online_inference_BCE(m::MTSeqModel_E3, x0::AbstractVecOrMat, z::AbstractVecOrMat,
    c::AbstractVecOrMat, y::AbstractVector; T_steps=length(y))

    x0decoder, mtbias_decoder, gen_model, dec_tform = unpack(m)[8:11]

    # Generative model
    # --------------------------------------------------
    if m.mtbias_only
        gen_model.state = x0decoder(x0)
        mtbias = mtbias_decoder(vcat(z, c))   # if linear, this is just the identity ∵ GRUCell.Wi

        nllh = map(1:T_steps) do t
            _nllh_bernoulli_per_batch(dec_tform(gen_model(mtbias)), y[t])  # y broadcasts over the batch implicitly
        end
    else
        posterior_grus = gen_model(vcat(z, c)) # output: BatchedGRU (def. above)
        posterior_grus.state = x0decoder(x0)  # 2×linear d_x0 × nbatch → d_x × nbatch

        nllh = map(1:T_steps) do t
            _nllh_bernoulli_per_batch(dec_tform(posterior_grus()), y[t])  # y broadcasts over the batch implicitly
        end
    end
    return reduce(vcat, nllh)
end


function online_inference_single_BCE(m::MTSeqModel_E3, h0::AbstractVecOrMat, z::AbstractVecOrMat,
    c::AbstractVecOrMat, y::AbstractArray)

    mtbias_decoder, gen_model, dec_tform = unpack(m)[9:11]

    # Generative model
    # --------------------------------------------------
    if m.mtbias_only
        gen_model.state = h0
        mtbias = mtbias_decoder(vcat(z, c))
        h_new = gen_model(mtbias)
        nllh = _nllh_bernoulli_per_batch(dec_tform(h_new), y)
    else
        posterior_grus = gen_model(vcat(z, c)) # output: BatchedGRU (def. above)
        posterior_grus.state = h0  # 2×linear d_x0 × nbatch → d_x × nbatch
        h_new = posterior_grus()
        nllh = _nllh_bernoulli_per_batch(dec_tform(h_new), y)
    end
    return nllh, h_new
end


function _nllh_bernoulli(m::MTSeqModel_E3, y::AbstractVector, yfull::AbstractVector;
        T_steps=70, T_enc=10, stoch=true)
    yhats = m(y, yfull; T_steps=T_steps, T_enc=T_enc, stoch=stoch)
    _nllh_bernoulli(yhats, y)
end

_nllh_bernoulli(ŷ::AbstractVector, y::AbstractVector) =
    sum([sum(Flux.logitbinarycrossentropy.(ŷŷ, yy)) for (ŷŷ, yy) in zip(ŷ, y)])

_nllh_bernoulli(ŷ::AbstractArray, y::AbstractArray) = sum(Flux.logitbinarycrossentropy.(ŷ, y))

_nllh_bernoulli_per_batch(ŷ::AbstractArray, y::AbstractArray) =
    let n=size(ŷ)[end]; res=Flux.logitbinarycrossentropy.(ŷ, y); sum(reshape(res, :, n), dims=1); end

function _nllh_gaussian(m::MTSeqModel_E3, y::AbstractVector, yfull::AbstractVector;
        T_steps=70, T_enc=10, stoch=true)
    model_out = m(y, yfull; T_steps=T_steps, T_enc=T_enc, stoch=stoch)
    _nllh_gaussian(model_out, y)
end

_nllh_gaussian(ŷ_lσ::AbstractVector, y::AbstractVector) =
    0.5*sum([((ŷŷ, ll) = ŷl; δ=yy-ŷŷ; sum(δ.*δ./(exp.(2*ll)))+sum(ll)) for (ŷl, yy) in zip(ŷ_lσ, y)])

function _nllh_gaussian_constvar(m::MTSeqModel_E3, y::AbstractVector, yfull::AbstractVector;
        T_steps=70, T_enc=10, stoch=true, logstd=-2.5)
    yhats = m(y, yfull; T_steps=T_steps, T_enc=T_enc, stoch=stoch)
    _nllh_gaussian_constvar(yhats, y, logstd)
end

_nllh_gaussian_constvar(ŷ::AbstractVector, y::AbstractVector, logstd::Number) =
    0.5*sum([(δ=yy-ŷŷ; sum(δ.*δ./(exp.(2*logstd)))+sum(logstd);) for (ŷŷ, yy) in zip(ŷ, y)])


nllh(m::MTSeqModel_E3{U,A,B,V,W,S,N,M}, y::AbstractVector, yfull::AbstractVector;
    T_steps=70, T_enc=10, stoch=true, logstd=-2.5) where {U,A,B,V,W,S,N,M <: MTSeq_CNN} =
        _nllh_bernoulli(m, y, yfull, T_steps=T_steps, T_enc=T_enc, stoch=stoch)

nllh(m::MTSeqModel_E3{U,A,B,V,W,S,N,M}, y::AbstractVector, yfull::AbstractVector;
    T_steps=70, T_enc=10, stoch=true, logstd=-2.5) where {U,A,B,V,W,S,N,M <: MTSeq_Single} =
        _nllh_gaussian_constvar(m, y, yfull, T_steps=T_steps, T_enc=T_enc, stoch=stoch, logstd=logstd)

nllh(m::MTSeqModel_E3{U,A,B,V,W,S,N,M}, y::AbstractVector, yfull::AbstractVector;
    T_steps=70, T_enc=10, stoch=true, logstd=-2.5) where {U,A,B,V,W,S,N,M <: MTSeq_Double} =
        _nllh_bernoulli(m, y, yfull, T_steps=T_steps, T_enc=T_enc, stoch=stoch)

nllh(::Type{MTSeq_CNN}, ŷ::AbstractVector, y::AbstractVector; logstd=-2.5) = _nllh_bernoulli(ŷ, y)
nllh(::Type{MTSeq_Single}, ŷ::AbstractVector, y::AbstractVector; logstd=-2.5) = _nllh_gaussian_constvar(ŷ, y, logstd)
nllh(::Type{MTSeq_Double}, ŷ::AbstractVector, y::AbstractVector; logstd=-2.5) = _nllh_gaussian(ŷ, y)


StatsBase.loglikelihood(m::MTSeqModel_E3, y::AbstractVector, yfull::AbstractVector;
    T_steps=70, T_enc=10, stoch=true, logstd=-2.5) = -nllh(m, y, yfull, T_steps=T_steps,
        T_enc=T_enc, stoch=stoch, logstd=logstd)


function _kl_penalty_stoch(β::Vector{T}, μ_x0, μ_z, μ_c, σ_x0, σ_z, σ_c) where T <: Float32
    kl = 0f0
    kl = β[1] * 0.5f0 * sum(1 .+ 2*log.(σ_x0.*σ_x0) - μ_x0.*μ_x0 - σ_x0.*σ_x0)
    kl += β[2] * 0.5f0 * sum(1 .+ 2*log.(σ_z.*σ_z) - μ_z.*μ_z - σ_z.*σ_z)
    kl += β[3] * 0.5f0 * sum(1 .+ 2*log.(σ_c.*σ_c) - μ_c.*μ_c - σ_c.*σ_c)
end

function _kl_penalty_det(β::Vector{T}, μ_x0, μ_z, μ_c) where T <: Float32
    kl = 0f0
    kl = β[1] * 0.5f0 * sum(1 .- μ_x0.*μ_x0)
    kl += β[2] * 0.5f0 * sum(1 .- μ_z.*μ_z)
    kl += β[3] * 0.5f0 * sum(1 .- μ_c.*μ_c)
end

function elbo(m::MTSeqModel_E3{U,A,B,V,W,S,N,M}, y::AbstractVector, yfull::AbstractVector; T_steps=70,
    T_enc=10, stoch=true, kl_coeff=1f0, βkl=ones(Float32, 3)) where {U,A,B,V,W,S,N,M}

    (x0, z, c), (μ_x0, μ_z, μ_c), (σ_x0, σ_z, σ_c) =
        posterior_samples(m, y, yfull; T_steps=T_steps, T_enc=T_enc, stoch=stoch)

    model_out = m(x0, z, c; T_steps=T_steps)
    recon = -nllh(M, model_out, y)

    # ⇒ Initially no KL (Hard EM), and then an annealing sched. cf. Bowman et al. etc?
    if stoch
        kl = _kl_penalty_stoch(β, μ_x0, μ_z, μ_c, σ_x0, σ_z, σ_c)
    else
        kl = _kl_penalty_det(β, μ_x0, μ_z, μ_c)
    end
    kl = kl * kl_coeff

    return - (recon + kl)
end




end
