module seq2seq

using Flux, ArgCheck, StatsBase

using ..modelutils
import ..modelutils: load!, randn_repar
export load!

const D_IM = 32

"""
    Seq2SeqModel(encoder, x0_post_dist, x0decoder, generator)

A sequence-to-sequence model 𝒴→𝒴 which imbibes `T_enc` steps of the sequence,
and predicts `T_steps` (including the original `T_enc`).

This type has 5 slots:
1. `encoder`
2. `x0_post_dist`
3. `x0decoder`
4. `generator`
5. `d`

The posterior over the initial state of the generator is learned variationally
via the `encoder` RNN. The final state of this encoder is passed to the
`x0_post_dist` which is a `MultiDense` objects which spits out two values,
i.e. the mean and variance of the latent posterior. Typically the posterior is
lower dimensional than the state of the generator network, so the latent
variable is expanded via the `x0decoder` network into the state of the
`generator`. Finally the `generator` is iterated in an open-loop fashion for
`T_steps` with no inputs to provide the prediction. The final slot `d` is just
a record of the dimension of 𝒴, used currently just for `show`.

--------------------
This type can be called via
```julia
    m = Seq2SeqModel(...)
    m(y::Vector)
```
where `y` is a vector of slices of the sequence for the encoder. This means each
`y[i]` is an `AbstractMatrix` of size ``d \\times n_{\\text{batch}}``.

Options include `T_enc`, `T_steps` (already discussed), `reset` (which performs
a state reset before execution; true by default), and `stoch`, whether to
randomly sample the latent state, or use the state mean (default true).
"""
struct Seq2SeqModel{S,U,V,W}
    encoder::S
    x0_post_dist::U
    x0decoder::V
    generator::W
    d::Int
end

Flux.@treelike Seq2SeqModel
unpack(m::Seq2SeqModel) = Flux.children(m)[1:4] # unpack struct; exclude `d`.

@eval modelutils load!(m::Seq2SeqModel, fname::String) = load!(Flux.params(m), fname)

function Base.show(io::IO, l::Seq2SeqModel)
    has_cnn = length(l.encoder) != 1
    out_type = has_cnn ? "CNN" : l.generator.layers[end][end] isa MultiDense ? "Probabilistic" : "Deterministic"
    print(io, "Seq2SeqModel(dim=", l.d, ", ", out_type, ")")
end


# ≡ __call__
function (m::Seq2SeqModel)(y::AbstractVector; T_steps=70, T_enc=10, reset=true, stoch=true)
    encoder, x0_post_dist, x0decoder, generator = unpack(m)
    reset && Flux.reset!(encoder)

    [encoder(y[tt]) for tt in 1:T_enc];
    enc_state = encoder.layers[end].state # d × d × nbatch → d_enc_state
    enc_state = enc_state isa Tuple ? enc_state[1] : enc_state  # handle LSTM
    μ_x0, σ_x0 = x0_post_dist(enc_state)  # 2×Dense: d_enc_state × nbatch → d_x0 × nbatch
    d_x0, nbatch = size(μ_x0)
    x0_smp = μ_x0 + randn_repar(σ_x0, d_x0, nbatch, stoch)

    # Set state (no need to `reset'!)
    generator.layers[1].state = x0decoder(x0_smp)  # RHS: d_x0 × nbatch → d_x × nbatch
    gmove = Flux.has_cuarrays() ? cu : identity   # isn't super important, can probably make `gpu`
    yhats = [generator(gmove(zeros(1, nbatch))) for t in 1:T_steps]
end


function nllh(m::Seq2SeqModel, y::AbstractVector; T_steps=70, T_enc=10, reset=true, stoch=true)
    yhats = m(y; T_steps=T_steps, T_enc=T_enc, reset=reset, stoch=stoch)
    sum([sum(Flux.logitbinarycrossentropy.(ŷ, y)) for (ŷ, y) in zip(yhats, y)])
end

StatsBase.loglikelihood(m::Seq2SeqModel, y::AbstractVector; T_steps=70, T_enc=10, reset=true,
    stoch=true) = -nllh(m, y; T_steps=T_steps, T_enc=T_enc, reset=reset, stoch=stoch)

"""
    create_model(d_x, d_x0, d_y, d_enc_state; encoder=:LSTM,
    cnn=false, out_heads=1, d_hidden=d_x)

Create `Seq2SeqModel` with the number of hidden units in the recurrent
generative model as `d_x`, the size of the latent variable for the initial state
as `d_x0` (typically ``d_x0 \\ll d_x``), `d_y` the dimension of the observations
and `d_enc_state` the number of hidden units in the recurrent encoder. This
encoder can be specified as `:LSTM`, `:GRU` or `:Bidirectional`. If the
observations are video data and one wishes to use a CNN to encode and decode,
specify `cnn=true`. By default the `Seq2SeqModel` looks to make a point estimate
of the future series, but specifying `out_heads=2` will result in uncertainty
estimation too. Finally `d_hidden` specifies the number of hidden units in non
CNN decoders.
"""
function create_model(d_x, d_x0, d_y, d_enc_state; encoder=:LSTM,
    cnn=false, out_heads=1, d_hidden=d_x)

    @assert !(out_heads > 1 && cnn) "cannot have multiple output heads and CNN."
    @argcheck out_heads in 1:2

    # for trained models cnn_enc_dec=(32,32,32)
    if cnn
        N_FILT=32
        d_conv_result = 4*4*N_FILT
        tform_enc = Chain(
            Conv((3, 3), 2=>N_FILT, relu, stride=2, pad=1),        # out: 16 × 16 × 32 × nbatch
            Conv((3, 3), N_FILT=>N_FILT, relu, stride=2, pad=1),   # out: 8 × 8 × 32 × nbatch
            Conv((3, 3), N_FILT=>N_FILT, relu, stride=2, pad=1),   # out: 4 × 4 × 32 × nbatch
            x->reshape(x, d_conv_result, :)
        )
        d_rnn_in = d_conv_result
    else
        tform_enc = Chain()
        d_rnn_in = d_y
    end


    if encoder == :LSTM
        init_enc = LSTM(d_rnn_in, d_enc_state)
    elseif encoder == :GRU
        init_enc = GRU(d_rnn_in, d_enc_state)
    elseif encoder == :Bidirectional
        init_enc = BRNNenc(d_rnn_in, d_enc_state)
        d_enc_state *= 2
    else
        error("encoder must be specified as :LSTM, :GRU, :Bidirectional")
    end

    enc_rnn = Chain(tform_enc.layers..., init_enc)

    post_x0 = MultiDense(Dense(d_enc_state, d_x0, identity), Dense(d_enc_state, d_x0, σ))
    post_x0.Dense2.b.data .= -2
    x0decoder = Dense(d_x0, d_x, identity)

    # (tform_enc, init_enc, post_x0, x0decoder)

    gen_rnn = GRU(1, d_x)

    if out_heads == 1 && !cnn
        decoder = Chain(Dense(d_x, d_hidden, relu), Dense(d_hidden, d_y, identity))
    elseif out_heads == 2 && !cnn
        decoder = Chain(Dense(d_x, d_hidden, relu), MultiDense(Dense(d_hidden, d_y, identity), Dense(d_hidden, d_y, identity)))
    else    # cnn
        decoder = Chain(
            Dense(d_x, d_conv_result, identity),
            x->reshape(x, 4, 4, N_FILT, :),                               # out: 4 × 4 × 32 × nbatch
            ConvTranspose((3,3), N_FILT=>N_FILT, relu, stride=2),         # out: 9 × 9 × 16 × nbatch
            ConvTranspose((3,3), N_FILT=>N_FILT, relu, stride=2, pad=1),  # out: 17 × 17 × 8 × nbatch
            ConvTranspose((3,3), N_FILT=>2, identity, stride=2, pad=1),   # out: 33 × 33 × 1 × nbatch
            x->x[1:D_IM, 1:D_IM, :, :]   # out: 32 × 32 × 2 × nbatch
        )
    end
    gen_model = Chain(gen_rnn, decoder)

    return Seq2SeqModel(enc_rnn, post_x0, x0decoder, gen_model, d_y)
end


end
