module Analise

using LinearAlgebra
using SparseArrays

export No, Material, Elemento, Estrutura

# Número de graus de liberdade de um nó.
num_gls_no = 2

# =======================================================================
"""
Implementa as propriedades de um nó.

id::Int64 -> Identificação do nó.
x::Float64 -> Coordenada x.
y::Float64 -> Coordenada y.
fx::Float64 -> Força na direção x.
fy::Float64 -> Força na direção y.
rx::Bool -> Apoio na direção x. true para impedido e 'false' para livre.
ry::Bool -> Apoio na direção y. true para impedido e 'false' para livre.
"""
struct No
    id::Int64
    x::Float64
    y::Float64
    fx::Float64
    fy::Float64
    rx::Bool
    ry::Bool

    function No(id, x, y; fx=0, fy=0, rx=false, ry=false)
        if id < 1
            throw(ArgumentError("O número de identificação do nó deve ser maior que 1!"))
        end
        new(id, x, y, fx, fy, rx, ry)
    end
end


"""
Retorna a distância entre dois nós.

no₁::No -> Nó de referência 1.
no₂::No -> Nó de referência 2.
"""
distancia(no₁::No, no₂::No)::Float64 = norm([(no₂.x - no₁.x), (no₂.y - no₁.y)])

"""
Retorna um vetor contendo os graus de liberdade do nó.
"""
gls_no(no::No)::Vector{Int64} = [Analise.num_gls_no * no.id - 1, Analise.num_gls_no * no.id]

"""
Retorna um vetor contendo as forças nodais.
"""
vetor_forcas(no::No)::Vector{Float64} = [no.fx, no.fy]

"""
Retorna o vetor de apoios nodais.
"""
vetor_apoios(no::No)::Vector{Bool} = [no.rx, no.ry]

# =======================================================================
"""
Define as propriedades do material que compõe os elementos..

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
Implementa as propriedades de uma elemento de treliça.

id: Número de identificação do elemento.
no₁: Nó inicial.
no₂: Nó final.
area: Área da seção transversal.
material: Material utilizado no elemento.
"""
struct Elemento
    id::Int64
    no₁::No
    no₂::No
    area::Float64
    material::Material

    function Elemento(id, no₁, no₂, area, material)
        if id < 1
            throw(ArgumentError("O número de identificação do elemento não pode ser menor que 1!"))
        end
        if area < 0
            throw(ArgumentError("A área da seção não pode ser menor que 0!"))
        end
        new(id, no₁, no₂, area, material)
    end
end

"""
Retorna o comprimento do elemento.
"""
comprimento(elemento::Elemento)::Float64 = distancia(elemento.no₁, elemento.no₂)

"""
Retorna um vetor contendo os graus de liberdade do elemento.
"""
gls_elemento(elemento::Elemento)::Vector{Int64} = vcat(gls_no(elemento.no₁), gls_no(elemento.no₂))

"""
Retorna o vetor de forças atuantes nos graus de liberdade do elemento.
"""
vetor_forcas(elemento::Elemento)::Vector{Float64} = vcat(vetor_forcas(elemento.no₁), vetor_forcas(elemento.no₂))

"""
Retorna o ângulo de inclinação do elemento.
"""
function angulo(elemento::Elemento)::Float64
    dx = elemento.no₂.x - elemento.no₁.x
    dy = elemento.no₂.y - elemento.no₁.y

    return atan(dy, dx)
end

"""
Retorna a matriz de rotação do elemento.
"""
function matriz_rotacao(elemento::Elemento)::Matrix{Float64}
    θ = angulo(elemento)
    s = sin(θ)
    c = cos(θ)
    
    return [c s  0 0 
           -s c  0 0 
            0 0  c s 
            0 0 -s c]
end

"""
Retorna a matriz de rigidez local do elemento.
"""
function ke(elemento::Elemento)::Matrix{Float64}
    L = comprimento(elemento)
    E = elemento.material.E
    A = elemento.area

    return (E*A/L) * [1 0 -1 0
                      0 0  0 0
                     -1 0  1 0
                      0 0  0 0]
end

"""
Retorna a matriz de rigidez do elemento rotacionada para o sistema global.
"""
function ke_global(elemento::Elemento)::Matrix{Float64}
    k = ke(elemento)
    r = matriz_rotacao(elemento)

    return r' * k * r
end

# =======================================================================
"""
Implementa as propriedades de uma Estrutura.

nos::Array{No} -> Nós que compõem a estrutura.
elementos::Array{Elemento} -> Elementos que compõem a estrutura.
"""
struct Estrutura
    nos::Vector{No}
    elementos::Vector{Elemento}
end

"""
Retorna o número de graus de liberdade da estrutura.
"""
num_gls_estrutura(estrutura::Estrutura)::Int64 = Analise.num_gls_no * length(estrutura.nos)

"""
Retorna os graus de liberdade livres.
"""
gls_livres(estrutura::Estrutura)::Vector{Int64} = setdiff(1:num_gls_estrutura(estrutura), vetor_apoios(estrutura))

"""
Retorna o vetor de forças da estrutura.
"""
function vetor_forcas(estrutura::Estrutura, incluir_gls_impedidos::Bool=true)::Vector{Float64}
    forcas = Array{Float64}(undef, num_gls_estrutura(estrutura))

    for no in estrutura.nos
        forcas[gls_no(no)] = [no.fx, no.fy]
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

        if any(vaps)
            gls = gls_no(no)
            append!(apoios, [gls[i] for (i,j) in enumerate(vaps) if j])
        end
    end
    return apoios
end

"""
Retorna a matriz de rigidez esparsa da estrutura.
Procedimento de montagem feito pela soma das rigidezes de cada elemento
diretamente na matriz esparsa global da estrutura.
"""
function k_estrutura_1(estrutura::Estrutura)::SparseMatrixCSC
    ngl = num_gls_estrutura(estrutura)
    k = spzeros(ngl, ngl)

    for e in estrutura.elementos
        gle = gls_elemento(e)
        k[gle, gle] += ke_global(e)
    end

    # Eliminação dos graus de liberdade impedidos
    gl_liv = gls_livres(estrutura)
    return k[gl_liv, gl_liv]
end

"""
Retorna a matriz de rigidez esparsa da estrutura.
Procedimento de montagem com 3 vetores separados.
"""
function k_estrutura_2(estrutura::Estrutura)::SparseMatrixCSC
    # Número de termos em cada matriz de rigidez de elementos.
    num_gle = (Analise.num_gls_no * 2)
    apoios = vetor_apoios(estrutura)

    num_it = length(estrutura.elementos) * num_gle^2

    linhas = ones(Int64, num_it)
    colunas = ones(Int64, num_it)
    termos = zeros(num_it)

    c = 1

    for elem in estrutura.elementos
        gle = gls_elemento(elem)
        ke = ke_global(elem)

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
function deslocamentos(estrutura::Estrutura, metodo=1)::Array{Float64}
    if metodo == 1
        kf = k_estrutura_1
    else
        kf = k_estrutura_2
    end

    u = kf(estrutura) \ vetor_forcas(estrutura, false)

    uf = zeros(num_gls_estrutura(estrutura))
    uf[gls_livres(estrutura)] = u
    return uf
end

"""
Retorna um vetor contendo os nós em suas novas posições com a estrutura deformada.
"""
function nos_deslocados(estrutura::Estrutura, escala::Int64=1)::Vector{No}
    num_nos = length(estrutura.nos)
    u_pontos = reshape(deslocamentos(estrutura, 2), (2, num_nos))'
    nos_desloc = Array{No}(undef, length(estrutura.nos))

    for i in 1:num_nos
        nos_desloc[i] = No(i, estrutura.nos[i].x + escala * u_pontos[i, 1], 
                           estrutura.nos[i].y + escala * u_pontos[i, 2])
    end
    return nos_desloc
end

"""
Retorna os valores dos esforços normais atuantes em cada elemento.
"""
function forcas_internas(estrutura::Estrutura)::Vector{Float64}
    forcas = Array{Float64}(undef, length(estrutura.elementos))
    nos_def = nos_deslocados(estrutura)

    for (i, elem) in enumerate(estrutura.elementos)
        l₀ = comprimento(elem)
        l₁ = distancia(nos_def[elem.no₁.id], nos_def[elem.no₂.id])
        deform = (l₁ - l₀) / l₀

        forcas[i] = elem.area * elem.material.E * deform
    end

    return forcas
end
    
end
