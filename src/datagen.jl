module datagen

using Random, Distributions
using OrdinaryDiffEq

export generate_data

clip(x, vmin, vmax) = max(min(x, vmax), vmin)

################################################################################
##                                                                            ##
##          Drawing images of (filled) circles on 32×32 canvases              ##
##                                                                            ##
################################################################################

# Generate quantized circles via Monte Carlo
circs = Matrix{Matrix}(undef, 10, 10)
@info "Pre-calculating circle discretization via Monte Carlo... "
for xx in -0.5:0.1:0.4
    for yy in -0.5:0.1:0.4
        _circmat = zeros(Int, 6, 6)
        _circmat_n = zeros(Int, 6, 6)
        for i in -3:0.01:3-0.01, j in -3:0.01:3-0.01
            _circmat_n[floor(Int, i)+4, floor(Int, j)+4] += 1
            (((i-xx)^2 + (j-yy)^2) <= 1.6^2) &&
                (_circmat[floor(Int, i)+4, floor(Int, j)+4] += 1)
        end
        _circmat = _circmat ./ _circmat_n;
        circs[round(Int, xx/0.1) + 6, round(Int, yy/0.1) + 6] = _circmat
    end
end
println("Done.\n")


"""
    gen_pic_circ!(out::AbstractMatrix, x, y; circ_ims=circs)

Generate a 1-channel image of a circle (via pre-computed `circ_ims`) on a
32×32 canvas supplied as the matrix `out` at the location (`x`, `y`), where
both coordinates are presumed to be in ``0 \\le (x,y) \\le 32``. (If the coods
are outside this range, the circle will be largely clipped, or absent from the
result.) Note that by "circle", I really mean "ball", although this may in
principle be changed in the pre-calculation function.

This is a mutating function which writes directly onto the supplied `out`
matrix. For a non-mutating version, use `gen_pic_circ(x,y)`.
"""
function gen_pic_circ!(out::AbstractMatrix, x, y; circ_ims=circs)
    ox, oy = x, y
    x, y = round(Int, x), round(Int, y)
    rx, ry = ox-x, oy-y
    rx, ry = ceil(Int, rx/0.1 + 5 +eps()), ceil(Int, ry/0.1 + 5 +eps())
    im = circ_ims[rx, ry]
    for xx in -3:2, yy in -3:2
        _x, _y = x+xx, y+yy
        _x = 32 - _x
        !(1 <= _x <= 32) && continue
        !(1 <= _y <= 32) && continue
        out[_x, _y] = max(out[_x, _y], im[xx+4, yy+4])
    end
    return out
end


gen_pic_circ(x, y) = let out = zeros(Float32, 32,32); o=gen_pic_circ!(out, x, y); end

"""
    gen_pic_circ2(x, y, x2, y2)

Draws TWO circles onto the same 32×32 canvas via the function `gen_pic_circ!`
(see docstring of that function). Circle centers are (x,y) and (x2, y2),
presumed to be in range ``0 \\le (x,y) \\le 32``.

Output: 32×32 matrix.
"""
function gen_pic_circ2(x,y,x2,y2)
    out = zeros(Float32, 32,32)
    gen_pic_circ!(out, x, y)
    gen_pic_circ!(out, x2, y2)
end

"""
    gen_pic_circ2(x, y, x2, y2)

Draws one circle onto each of a 2-channel 32×32 canvas via the function
`gen_pic_circ!` (see docstring of that function). Circle centers are (x,y) and
(x2, y2), presumed to be in range ``0 \\le (x,y) \\le 32``.

Output: 32×32×2 tensor.
"""
function gen_pic_circ2_2chan(x,y,x2,y2)
    out = zeros(Float32, 32,32,2)
    @views gen_pic_circ!(out[:,:,1], x, y)
    @views gen_pic_circ!(out[:,:,2], x2, y2)
    return out
end


imcood_tf(x,y) = let xy=7*[x+2, y+2].+2; [xy[1],xy[2]]; end

"""
    gen_pic_circ2_cood_tf(xy1, xy2)

A version of `gen_pic_circ2` (see docstring), but where centers xy1 and xy2 are
specified in the range [-2,2]×[-2,2] and transformed to the [0,32]×[0,32] range.
"""
function gen_pic_circ2_cood_tf(xy1, xy2)
    xy1, xy2 = imcood_tf(xy1...), imcood_tf(xy2...)
    gen_pic_circ2(xy1[2], xy1[1], xy2[2], xy2[1])
end

"""
    gen_pic_circ2_2chan_cood_tf(xy1, xy2)

A version of `gen_pic_circ2_2chan` (see docstring), but where centers xy1 and
xy2 are specified in the range [-2,2]×[-2,2] and transformed to the
[0,32]×[0,32] range.
"""
function gen_pic_circ2_2chan_cood_tf(xy1, xy2)
    xy1, xy2 = imcood_tf(xy1...), imcood_tf(xy2...)
    gen_pic_circ2_2chan(xy1[2], xy1[1], xy2[2], xy2[1])
end


################################################################################
##                                                                            ##
##            Solving and manipulating the Double Pendulum ODE                ##
##                                                                            ##
################################################################################


"""
    double_pendulum(du,u,p,t)

Definition of the double pendulum ODE. Started life from github user `zaman13`.
https://github.com/zaman13/Double-Pendulum-Motion-Animation/blob/master/Code/Double_pendulum-uniformly_spaced_time.ipynb.
See wikipedia article for equations of motion via a Lagrangian approach.
"""
function double_pendulum(du,u,p,t)
    # du = derivatives
    # u = variables
    # p = parameters
    # t = time variable

    m1, m2, L1, L2, g, α₁, α₂ = p;    # α = damping

    c = cos(u[1]-u[3]);  # intermediate variables
    s = sin(u[1]-u[3]);  # intermediate variables

    du[1] = u[2];   # d(theta 1)
    du[2] = ( m2*g*sin(u[3])*c - m2*s*(L1*c*u[2]^2 + L2*u[4]^2) - (m1+m2)*g*sin(u[1]) -α₁*u[2]) /(L1 *(m1+m2*s^2));
    du[3] = u[4];   # d(theta 2)
    du[4] = ((m1+m2)*(L1*u[2]^2*s - g*sin(u[3]) + g*sin(u[1])*c) + m2*L2*u[4]^2*s*c -α₂*u[4]) /(L2 * (m1+m2*s^2));
end


"""
    solve_dblpend(T=20; masses=[1,1], lengths=[1,1], g=9.8, dt=0.05,
    u0 = [2π/3; 0; π/3; 0.0], α₁=0, α₂=0)

Solving the double pendulum ODE defined in `double_pendulum`. The only
positional argument `T` defines the total timespan in terms of seconds, but *not
the length of the output*. The motion is sampled at the interval `dt`, and hence
the total output length is `T/dt`. The ODE parameters are given by the `masses`
and `lengths`, with elements corresponding to the first (inner) and second
(outer) pendulum respectively. The damping factor (which is applied to the
speed) is specified via `α₁` and `α₂` for each pendulum respecitvely. Finally,
`u0` is the initial condition (θ₁, θ̇₁, θ₂, θ̇₂), and `g` is the gravitational
constant.

Output: returns a tuple, the first of which is a matrix of size (T/dt)×4
where each row gives the state (θ₁, θ̇₁, θ₂, θ̇₂) for each point in time; the
second gives the solution in Euclidean coordinates (y₁, x₁, y₂, x₂) for
modelling purposes (and without velocity information).
"""
function solve_dblpend(T=20; masses=[1,1], lengths=[1,1], g=9.8, dt=0.05, u0 = [2π/3; 0; π/3; 0.0],
        α₁=0, α₂=0)
    m1, m2 = masses   # mass of pendulum 1,2 (in kg)
    L1, L2 = lengths  # length of pendulum 1,2 (in meters)
    # u0 == (pend1 θ, θ̇, pend2 θ, θ̇)

    # Solving the system
    pars = [m1, m2, L1, L2, g, α₁, α₂];
    tlims = (0.0, T);
    prob = ODEProblem(double_pendulum, u0, tlims, pars);
    ode_solver = Vern7() # Verner's "Most Efficient" 7/6 Runge-Kutta method
    sol = solve(prob, ode_solver, reltol=1e-6);

    z = reduce(hcat, sol(0:dt:(T-dt)).u)

    xy1 = vcat(L1*sin.(z[1,:]'), -L1*cos.(z[1,:]'))                         # First Pendulum
    xy2 = vcat(xy1[1,:]' + L2*sin.(z[3,:]'), xy1[2,:]' - L2*cos.(z[3,:]'))         # Second Pendulum

    return z, vcat(xy1, xy2)
end


# Various kinetic and potential energy calculations for a pendulum. The
# potential energy is a simple sum of the PE of the bobs, but the KE of the
# combined system requires the extra term, which is reflected in the total
# energy `energy_dblpend`.
ke(θ, θ̇, r=1) = ke(θ̇, r)
ke(θ̇, r=1) = 0.5*(r*θ̇)^2
extra_dbl_ke(θ₁, θ̇₁, θ₂, θ̇₂) = θ̇₁*θ̇₂*cos(θ₁-θ₂)  # extra term from the coupling
pe(θ, θ̇, r=1) = pe(θ)
pe(θ) = 9.8*(1-cos(θ))
energy_dblpend(θ₁, θ̇₁, θ₂, θ̇₂, r₁=1,r₂=1) = sum(2*f(θ₁,θ̇₁,r₁)+f(θ₂,θ̇₂,r₂) for f in [pe,ke]) +θ̇₁*θ̇₂*cos(θ₁-θ₂)


################################################################################
##                                                                            ##
##                            Generating the dataset                          ##
##                                                                            ##
################################################################################

rsgn(; rng=Random.GLOBAL_RNG) = rand(rng, [-1,+1])   # random sign


"""
    sample_init(target)

Sample the initial conditions (θ₁, θ̇₁, θ₂, θ̇₂) of the double pendulum system
with an energy specified by `target`. My initial experiments used the value
`3.5*9.8` which is the default if called without arguments.

It is not entirely uniformly at random over (θ₁, θ̇₁, θ₂, θ̇₂), since some
combinations led to fairly boring behaviour (~simple harmonic motion). I didn't
have time to analyze exactly what initial conditions led to this, but I put a
couple of intuitive (to me) changes in: requiring that the initial height of the
first pendulum (θ₁) is bounded above 0, as is θ₂. Notice also that the ± in the
quadratic formula for θ̇₂ is fixed as the *opposite* sign of θ̇₁. This appears to
avoid some of the synchronization between the pendula that I noticed from
sampling uniformly at random.
"""
function sample_init(target=3.5*9.8; rng=Random.GLOBAL_RNG)
    H = target/9.8
    ul = acos(max(1 - H/2, -1))
    θ₁ = rsgn(rng=rng)*rand(rng, Uniform(ul/2, ul))
    θ₂ = let a=acos(clip(3-H-2*cos(θ₁), -1, 1)); rsgn(rng=rng)*rand(rng, Uniform(a/2, a)); end
    κ = 9.8*((H-3) + 2*cos(θ₁) + cos(θ₂))  # remaining energy
    θ̇₁ = rand(rng, Uniform(0, sqrt(κ)))
    b, c = 2*θ̇₁*cos(θ₁-θ₂), 2*(θ̇₁^2 - κ)
    θ̇₂ = 0.5*(-b -sign(θ̇₁)*sqrt(b^2-4*c))
    return [θ₁, θ̇₁, θ₂, θ̇₂]
end




"""
    generate_data(;N=10, Nvalid=1, Ntest=1, tT=140, _seed=1240,
        max_energy=40, damping_vals=nothing, dt=0.05)

Generate sequences of length `tT` from the cartesian product of the damping_vals
(defaults to
    [0.0,  0.01,  0.02,  0.05,  0.1,  0.20,  0.30,  0.50,  0.75,  1.00]
for each bob). Each element in this cart. prod generates `N` sequences (def 10)
resulting in (10×10)×10 sequences of length `tT`. Also a validation and test
set obtain `Nvalid` and `Ntest` seqeunces from this cartesian product too. The
initial conditions are sampled from `sample_init` with a random energy with
a maximum specified as `max_energy`, with a bias towards higher energy. The
default behaviour is to sample the sequence every `dt=0.05` seconds, and hence
a `tT=140` sequence covers only 7 seconds of ODE rollout.

Output:
(1) `data_xy`: a Dict with keys `[:train]`, `[:valid]`, `[:test]`, each of which
contains a list of all (10×10)×N (default) sequences of length `tT`.

(2) `data_θ`: as above, but now with each row of each sequence containing the
values (θ₁, θ̇₁, θ₂, θ̇₂).

(3) `data_meta`: A Dict of the metadata of each sequence. E.g. the `[:train]`
key holds a list of the [length, %total length, initial energy, damping values,
and initial conditions] of each sequence.
"""
function generate_data(;N=10, Nvalid=1, Ntest=1, tT=140, _seed=1240,
        max_energy=40, damping_vals=nothing, dt=0.05)

    rng = MersenneTwister()
    Random.seed!(rng, _seed)

    damping_vals = something(damping_vals,
        [0.0,  0.01,  0.02,  0.05,  0.1,  0.20,  0.30,  0.50,  0.75,  1.00])

    data_xy = Dict(:train=>[], :valid=>[], :test=>[])
    data_θ = Dict(:train=>[], :valid=>[], :test=>[])
    data_meta = Dict(:train=>[], :valid=>[], :test=>[])

    for α₁ in damping_vals, α₂ in damping_vals
        lens = [repeat([tT], N), repeat([tT], Nvalid), repeat([tT], Ntest)]
        for (ls, tvt) in zip(lens, [:train, :valid, :test])
            for l in ls
                energy = max_energy*rand(rng, Beta(3,1))  # heuristically chosen: bias towards larger energy
                u0 = sample_init(energy);   # original: [2π/3 + 0.00001; 0; π/3 + 0.00001; 0.0]
                lc = ceil(Int, l*dt)
                states, traj = solve_dblpend(lc; u0=u0, α₁=α₁, α₂=α₂, dt=dt)
                data_pend = convert(Array{Float32}, Matrix(traj'))
                thetas = convert(Array{Float32}, Matrix(states'))
                push!(data_xy[tvt], data_pend)
                push!(data_θ[tvt], thetas)
                push!(data_meta[tvt], [lc÷dt, energy, α₁, α₂, u0...])
            end
        end
    end

    data_meta = Dict(k=>Matrix(reduce(hcat, v)) for (k,v) in data_meta)
    data_meta = Dict(k=>(convert(Vector{Int}, v[1,:]), v[1,:]/sum(v[1,:]), v[2,:], v[3:end,:]) for (k,v) in data_meta)

    return data_xy, data_θ, data_meta
end


end
