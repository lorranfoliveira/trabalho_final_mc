module Analise

using LinearAlgebra
using SparseArrays

export No, Material, Elemento, Estrutura

# =======================================================================
"""
Implementa as propriedades de um nó.

id::Int64 -> Identificação do nó.
x::Float64 -> Coordenada x.
y::Float64 -> Coordenada y.
fx::Float64 -> Força na direção x.
fy::Float64 -> Força na direção y.
fxy::Union{Float64, Nothing} -> Momento fletor.
rx::Bool -> Apoio na direção x. true para impedido e 'false' para livre.
ry::Bool -> Apoio na direção y. true para impedido e 'false' para livre.
rxy::Union{Bool, Nothing} -> Apoio na direção xy (impede rotação). 'true' para impedido e 'false' para livre.
"""
struct No
    id::Int64
    x::Float64
    y::Float64
    fx::Float64
    fy::Float64
    fxy::Union{Float64, Nothing}
    rx::Bool
    ry::Bool
    rxy::Union{Bool, Nothing}

    function No(id, x, y; fx=0, fy=0, fxy=nothing, rx=false, ry=false, rxy=nothing)
        if id < 1
            throw(ArgumentError("O número de identificação do nó deve ser maior que 1!"))
        end
        if (fxy isa Nothing && !(rxy isa Nothing)) || (rxy isa Nothing && !(fxy isa Nothing))
            throw(ArgumentError("'fxy' e 'rxy' devem ser simultaneamente 'nothing' para indicar um nó de treliça."))
        end
        new(id, x, y, fx, fy, fxy, rx, ry, rxy)
    end
end

"""
Retorna o número de graus de liberdade do nó em função de seus dados.
Útil para identificar se a estrutura é de pórtico ou de treliça.
"""
function num_gls(no::No)::Int8
    if (no.fxy isa Nothing) || (no.rxy isa Nothing)
        return 2
    else
        return 3
    end
end

"""
Calcula a distância entre dois nós.

no₁::No -> Nó de referência 1.
no₂::No -> Nó de referência 2.
"""
function distancia(no₁::No, no₂::No)::Float64
    return norm([(no₂.x - no₁.x), (no₂.y - no₁.y)])
end

"""
Retorna os graus de liberdade do nó.
"""
function gls_no(no::No)::Vector{Int64}
    ngl = num_gls(no)
    return [ngl * no.id - 2, ngl * no.id - 1, ngl * no.id][4 - ngl:end]
end

"""
Retorna o vetor de forças nodais.
"""
function vetor_forcas(no::No)::Vector{Float64}
    return [no.fx, no.fy, no.fxy][1:num_gls(no)]  
end

"""
Retorna o vetor de apoios nodais.
"""
function vetor_apoios(no::No)::Vector{Bool}
    return [no.rx, no.ry, no.rxy][1:num_gls(no)]
end

# =======================================================================
"""
Implementa as propriedades do material.

v: Coeficiente de Poisson.
E: Módulo de Elasticidade.
"""
struct Material
    id::Int64
    E::Float64

    function Material(id, E)
        if id < 1
            throw(ArgumentError("O número de identificação do material não pode ser menor que 1!"))
        end
        if !(E > 0)
            throw(ArgumentError("O Módulo de Elasticidade deve ser maior que 0!"))
        end
        new(id, E)
    end
end

# =======================================================================
"""
Implementa as propriedades de uma elemento de pórtico.

tipo -> Tipo de elemento a ser utilizado. '1' para elemento de pórtico e '2' para 
        elemento de treliça. 
"""
struct Elemento
    id::Int64
    no₁::No
    no₂::No
    area::Float64
    inercia::Float64
    material::Material
    tipo::Int8

    function Elemento(id, no₁, no₂, area, inercia, material, tipo)
        if id < 1
            throw(ArgumentError("O número de identificação do elemento não pode ser menor que 1!"))
        end
        if !(area > 0)
            throw(ArgumentError("A área da seção deve ser maior que 0!"))
        end
        if !(inercia > 0)
            throw(ArgumentError("O momento de inércia da seção deve ser maior que 0!"))
        end
        if tipo ∉ [1, 2]
            throw(ArgumentError("O tipo de elemento deve ser '1' (elemento de pórtico) ou '2' (elemento de treliça)!"))
        end
        new(id, no₁, no₂, area, inercia, material, tipo)
    end
end

"""
Comprimento do elemento.
"""
function comprimento(elemento::Elemento)::Float64
    return distancia(elemento.no₁, elemento.no₂)
end

"""
Retorna os graus de liberdade do elemento.
"""
function gls_elem(elemento::Elemento)::Vector{Int64}
    gl_no₁ = gls_no(elemento.no₁)
    gl_no₂ = gls_no(elemento.no₂)

    return vcat(gl_no₁, gl_no₂)
end

"""
Retorna o vetor de forças atuantes nos graus de liberdade do elemento.
"""
function vetor_forcas(elemento::Elemento)::Vector{Float64}
    return vcat(vetor_forcas(elemento.no₁), vetor_forcas(elemento.no₂))
end

"""
Ângulo de inclinação do elemento.
"""
function angulo(elemento::Elemento)::Float64
    dx = elemento.no₂.x - elemento.no₁.x
    dy = elemento.no₂.y - elemento.no₁.y

    return atan(dy / dx)
end

"""
Matriz de rotação do elemento.
"""
function mat_rot(elemento::Elemento)::Matrix{Float64}
    θ = angulo(elemento)
    s = sin(θ)
    c = cos(θ)
    
    if elemento.tipo == 1
        return [c s  0 0 0 0
                -s c 0 0 0 0
                0 0 1 0 0 0
                0 0 0 c s 0
                0 0 0 -s c 0
                0 0 0 0 0 1]
    else
        return [c s  0 0 
                -s c 0 0 
                0 0 c s 
                0 0 -s c ]
    end
end

"""
Matriz de rigidez local do elemento.
"""
function ke(elemento::Elemento)::Matrix{Float64}
    L = comprimento(elemento)
    E = elemento.material.E
    I = elemento.inercia
    A = elemento.area

    if elemento.tipo == 1
        return (E*I/L^3) * [A*L^2/I 0 0 -A*L^2/I 0 0
                            0 12 6L 0 -12 6L
                            0 6L 4L^2 0 -6L 2L^2
                            -A*L^2/I 0 0 A*L^2/I 0 0
                            0 -12 -6L 0 12 -6L
                            0 6L 2L^2 0 -6L 4L^2]
    else
        return (E*A/L) * [1 0 -1 0
                          0 0 0 0
                          -1 0 1 0
                          0 0 0 0]
    end
end

"""
Matriz de rigidez do elemento rotacionada para o sistema global.
"""
function ke_rot(elemento::Elemento)::Matrix{Float64}
    k = ke(elemento)
    r = mat_rot(elemento)

    return r' * k * r
end

# =======================================================================
"""
Implementa as propriedades de uma Estrutura.

nos::Array{No} -> Nós que compõem a estrutura.
elementos::Array{Elemento} -> Elementos que compõem a estrutura.
forcas::Dict{Int64, Float64} -> Dicionário contendo os graus de liberdade (chaves) e as forças 
                                neles atuantes (valores).
apoios::Dict{Int64, Int8} -> Dicionário contendo os graus de liberdade (chaves) e o tipo de 
                            vinculação
"""
struct Estrutura
    nos::Vector{No}
    elementos::Vector{Elemento}
    tipo::Int8
end

"""
Retorna o número de graus de liberdade da estrutura.
"""
function num_gls(estrutura::Estrutura)::Int64
    if estrutura.tipo == 1
        return 3 * length(estrutura.nos)
    else
        return 2 * length(estrutura.nos)
    end
end

"""
Retorna os graus de liberdade livres.
"""
function gls_livres(estrutura::Estrutura)::Vector{Int64}
    return setdiff(1:num_gls(estrutura), vetor_apoios(estrutura))
end

"""
Retorna o vetor de forças da estrutura.
"""
function vetor_forcas(estrutura::Estrutura, incluir_gls_impedidos::Bool=true)::Vector{Float64}
    forcas = zeros(num_gls(estrutura))
    for no in estrutura.nos
        gls = gls_no(no)
        forcas[gls] = [no.fx, no.fy, no.fxy][1:length(gls)]
    end

    if incluir_gls_impedidos
        return forcas
    else
        return forcas[gls_livres(estrutura)]
    end
end


"""
Retorna um vetor com os graus de liberdade impedidos.
"""
function vetor_apoios(estrutura::Estrutura)::Vector{Int64}
    apoios::Array{Int64} = []
    for no in estrutura.nos
        vaps = vetor_apoios(no)

        gls = gls_no(no)

        if any(vaps)
            append!(apoios, [gls[i] for i in 1:length(gls) if vaps[i] == true])
        end
    end
    return apoios
end

"""
Retorna a matriz de rigidez esparsa da estrutura.
Procedimento de montagem com 3 vetores separados.
"""
function ke_estrutura(estrutura::Estrutura)::SparseMatrixCSC
    # Número de termos em cada matriz de rigidez de elementos.
    num_gle = gls_elem(estrutura.elementos[1])
    apoios = vetor_apoios(estrutura)

    num_it = length(estrutura.elementos) * num_gle^2

    linhas = ones(Int64, num_it)
    colunas = ones(Int64, num_it)
    termos = zeros(num_it)

    c = 1

    for elem in estrutura.elementos
        gle = gls_elem(elem)
        ke = ke_rot(elem)

        # Posições livres do elemento.
        pos_livres = [i for (i, gl) in enumerate(gle) if gl ∉ apoios]

        if length(pos_livres) > 0
            for i in pos_livres
                lin = gle[i] - count(x -> x < gle[i], apoios)
                for j in pos_livres
                    linhas[c] = lin
                    colunas[c] = gle[j] - count(x -> x < gle[j], apoios)
                    termos[c] = ke[i, j]
                    c += 1
                end
            end
        end
    end
    k = sparse(linhas, colunas, termos)
    dropzeros!(k)

    return k
end

"""
Retorna o vetor dos deslocamentos nodais.
"""
function deslocamentos(estrutura::Estrutura, incluir_gls_impedidos::Bool=true)::Array{Float64}
    u = ke_estrutura(estrutura) \ vetor_forcas(estrutura, false)

    if incluir_gls_impedidos
        for i in vetor_apoios(estrutura)
            insert!(u, i, 0)
        end
    end
    return u
end
end
