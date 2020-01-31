using Flux   # must be Flux@0.90
using BSON

if Flux.has_cuarrays()
    using CuArrays
end

e_k(T, n, k) = begin; out = zeros(T, n); out[k] = 1; out; end
e_k(n, k) = e_k(Float32, n, k)
e1(T, n) = e_k(T, n, 1)
e1(n) = e_k(Float32, n, 1)

unsqueeze(xs, dim) = reshape(xs, (size(xs)[1:dim-1]..., 1, size(xs)[dim:end]...));

"""
    MultiDense(Dense(in_1, out_1), Dense(in_2, out_2))

2× Dense layer for convenience for variational (μ/σ) like application. Allows
one to keep both layers together, and can be called like:

    m(x)  # = (m.Dense1(x), m.Dense2(x))

for an instance `m`; the result is a Tuple.
"""
struct MultiDense{F1, F2, S, T}
    Dense1::Dense{F1,S,T}
    Dense2::Dense{F2,S,T}
end
Flux.@treelike MultiDense    # new Flux calls this '@functor'. May need to replace.
(a::MultiDense)(x::AbstractArray) = (a.Dense1(x), a.Dense2(x))

function Base.show(io::IO, l::MultiDense)
  fs = [d.σ == identity ? "" : ", " * string(d.σ) for d in [l.Dense1, l.Dense2]]
  print(io, "MultiDense((", size(l.Dense1.W, 2), ", ", size(l.Dense1.W, 1), fs[1], ")")
  print(io, ", (", size(l.Dense2.W, 2), ", ", size(l.Dense2.W, 1), fs[2], "))")
end


"""
    BRNNenc(d_in, d_state)

Bidirectional RNN using an LSTM in each direction with `d_in` dimensional inputs
and `d_state` dimension for _each_. This is an encoder variant which only uses
the final state (of each LSTM), concatenated. Therefore, its usage is as
follows for `xs`, a Vector where each element corresponds to the input at time
``t``:

    m(xs)::Vector

for an instance `m`, where the output vector is the final state concatenation.
"""
struct BRNNenc{V}
    forward::V
    backward::V
end

BRNNenc(in::Integer, out::Integer) = BRNNenc(LSTM(in, out), LSTM(in, out))

function (m::BRNNenc)(xs::AbstractVector)
    m.forward.(xs)
    m.backward.(reverse(xs))
    return vcat(m.forward.state[1], m.backward.state[1])
end

Flux.@treelike BRNNenc

function Base.show(io::IO, l::BRNNenc)
  fs = [d.σ == identity ? "" : ", " * string(d.σ) for d in [l.Dense1, l.Dense2]]
  print(io, "Bidirectional RNN. (Forward / Backward):")
  show(l.forward); show(l.backward)
end




################################################################################
##  Save and Load utils for parameters of arbitrary models.                   ##
##                                                                            ##
##  These utils can be used for *any* models as they don't save the model def ##
##  itself, but uses the fact that the parameter 'vector' can be extracted    ##
##  easily using Flux's treelike structure and similarly loaded easily.       ##
################################################################################

"""
    save(fname::String, ps::Flux.Params, opt::Flux.ADAM; timestamp::Bool=false, force::Bool=false)
    save(fname::String, ps::Flux.Params; timestamp::Bool=false, force::Bool=false)

Save a parameter vector extracted via `Flux.params(models...)` to `fname`. in
BSON format. Since the state of the optimizer is critical for continuing
optimization (should it be required), the optimizer can optionally be given too.
At the moment this is limited to Adam, but this is easily extended.

Two options:
   (i) `timestamp = true` will insert a timestamp just before the extension.
   (ii) `force=true` will overwrite an existing file of the same name. The
        default behaviour is to throw an error instead.
"""
function save(fname::String, ps::Flux.Params, opt::Flux.ADAM; timestamp::Bool=false, force::Bool=false)
    fname = _get_and_check_fname(fname, timestamp, force)
    BSON.bson(fname, ps=[cpu(Tracker.data(p)) for p in ps], etabeta=(opt.eta, opt.beta),
                opt_state=[mapleaves(cpu, opt.state[p]) for p in ps])
end

function save(fname::String, ps::Flux.Params; timestamp::Bool=false, force::Bool=false)
    fname = _get_and_check_fname(fname, timestamp, force)
    BSON.bson(fname, ps=[cpu(Tracker.data(p)) for p in ps])
end

function _get_and_check_fname(fname::String, timestamp::Bool, force::Bool)
    fn = splitext(fname)
    @argcheck fn[2] in [".bson", ""]
    timestamp && (fname = fn[1] * Dates.format(Dates.now(), "yyyymmddTHH:MM:SS") * fn[2])
    !force && (@assert !isfile(fname) "fname already exists. To overwrite, use `force=true`.")
    return fname
end


"""
    load!(ps::Flux.Params, fname::String)
    load!(ps::Flux.Params, opt::Flux.ADAM, fname::String)

The inverse operation to save; loads a parameter vector from a BSON file created
per the `save` function. This operation will overwrite the parameter values in
the `ps` parameter 'vector' and optionally also the optimizer state (and η, β)
if specified as an argument.

The function takes an _existing_ parameter vector in order to link it to some
pre-specified models; this allows the file format to be agnostic to the original
model.
"""
function load!(ps::Flux.Params, fname::String)
    tf = Tracker.data(first(ps)) isa CuArray ? gpu : identity
    _load_pars!(ps, BSON.load(fname)[:ps], tf)
end

function load!(ps::Flux.Params, opt::Flux.ADAM, fname::String)
    tf = Tracker.data(first(ps)) isa CuArray ? cu : identity
    f=BSON.load(fname)
    _load_pars!(ps, f[:ps], tf)
    opt.eta, opt.beta = f[:etabeta]
    for (p, p_saved) in zip(ps, f[:opt_state])
        opt.state[p] = tf(p_saved)
    end
end


function _load_pars!(ps_to::Flux.Params, ps_from::Vector, tf::Function)
    try
        for (p, p_saved) in zip(ps_to, ps_from)
            p.data .= tf(p_saved)
        end
    catch e
        # Debugging dimension mismatch:
        if e isa DimensionMismatch
            @warn "Specified file has parameters of different dimensions:"; flush(stderr)
            for (p, p_saved) in zip(ps_to, ps_from)
                txt = ("(cur/new): ", size(p.data), " <-- ", size(p_saved) , "\n")
                if size(p.data) == size(p_saved)
                    print(txt...)
                else
                    printstyled(IOContext(stdout, :color => true), txt...; bold=true)
                end
            end
        end
        rethrow(e)
    end
end