include("analise.jl")

using Test
using .Analise
using SparseArrays
using LinearAlgebra

@testset verbose = true "Analise" begin
    # =======================================================================
    # Nós
    nos = [No(1, 1, -1; rx=true, ry=true),
           No(2, 3, -1),
           No(3, 5, -1),
           No(4, 7, -1; rx=true, ry=true),
           No(5, 3, 1; fx=10, fy=-5),
           No(6, 5, 1)]
    
    # Material
    material = Material(1, 100e6)
    
    # Elementos
    area = 0.0014
    elementos = [Elemento(1, nos[1], nos[2], area, material),
                 Elemento(2, nos[1], nos[5], area, material),
                 Elemento(3, nos[2], nos[3], area, material),
                 Elemento(4, nos[2], nos[6], area, material),
                 Elemento(5, nos[4], nos[3], area, material),
                 Elemento(6, nos[5], nos[2], area, material),
                 Elemento(7, nos[6], nos[3], area, material),
                 Elemento(8, nos[6], nos[4], area, material),
                 Elemento(9, nos[6], nos[5], area, material)]
    
    # Estrutura
    estrutura = Estrutura(nos, elementos)

    # =======================================================================
    @testset "No" begin
        @test nos[2].id == 2
        @test nos[2].x == 3
        @test nos[2].y == -1
        @test nos[2].fx == 0
        @test nos[2].fy == 0
        @test nos[2].rx == false
        @test nos[2].ry == false

        @test_throws ArgumentError No(0, 3, -1)

        # Métodos
        @test Analise.distancia(nos[1], nos[5]) ≈ 2.8284271247461903
        
        @test Analise.gls_no(nos[3]) == [5, 6]
        
        @test Analise.vetor_forcas(nos[1]) == [0, 0]
        @test Analise.vetor_forcas(nos[5]) == [10, -5]

        @test Analise.vetor_apoios(nos[1]) == [true, true]
        @test Analise.vetor_apoios(nos[2]) == [false, false]
    end

    # =======================================================================
    @testset "Material" begin
        @test material.id == 1
        @test material.E == 100e6

        @test_throws ArgumentError Material(-1, 100e6)
        @test_throws ArgumentError Material(1, -100e6)
    end

    # =======================================================================
    @testset "Elementos" begin
        @test elementos[2].id == 2
        @test elementos[2].no₁ == nos[1]
        @test elementos[2].no₂ == nos[5]
        @test elementos[2].area ≈ area
        @test elementos[2].material == material

        # Métodos
        @test Analise.comprimento(elementos[2]) ≈ 2.8284271247461903
        
        @test Analise.gls_elemento(elementos[2]) == [1, 2, 9, 10]

        @test Analise.vetor_forcas(elementos[2]) == [0, 0, 10, -5]
        @test Analise.vetor_forcas(elementos[9]) == [0, 0, 10, -5]
        @test Analise.vetor_forcas(elementos[3]) == [0, 0, 0, 0]
        
        @test Analise.angulo(elementos[4]) ≈ pi / 4
        @test Analise.angulo(elementos[9]) ≈ 0
        @test Analise.angulo(elementos[8]) ≈ -pi / 4

        @test Analise.matriz_rotacao(elementos[3]) ≈ [1 0 0 0 
                                               0 1 0 0
                                               0 0 1 0
                                               0 0 0 1]
        @test Analise.matriz_rotacao(elementos[4]) ≈ [√2/2 √2/2  0       0
                                              -√2/2 √2/2  0       0
                                               0       0  √2/2 √2/2
                                               0       0 -√2/2 √2/2]

        @test Analise.ke(elementos[4]) ≈ [49497.474683058324  0 -49497.474683058324   0
                                                           0  0                   0   0
                                         -49497.474683058324  0  49497.474683058324   0
                                                           0  0                   0   0]

        @test Analise.ke_global(elementos[4]) ≈ [24748.737341529166 24748.737341529162 -24748.737341529166 -24748.737341529162
                                                 24748.73734152916 24748.737341529155 -24748.73734152916 -24748.737341529155
                                                -24748.737341529166 -24748.737341529162 24748.737341529166 24748.737341529162 
                                                -24748.73734152916 -24748.737341529155 24748.73734152916 24748.737341529155]
    end

    # =======================================================================
    @testset "Estrutura" begin
        @test estrutura.nos == nos
        @test estrutura.elementos == elementos

        # Métodos
        @test Analise.num_gls_estrutura(estrutura) == 12

        @test Analise.gls_livres(estrutura) == [3, 4, 5, 6, 9, 10, 11, 12]

        @test Analise.vetor_forcas(estrutura, true) == [0, 0, 0, 0, 0, 0, 0, 0, 10, -5, 0, 0]
        @test Analise.vetor_forcas(estrutura, false) == [0, 0, 0, 0, 10, -5, 0, 0]

        @test Analise.vetor_apoios(estrutura) == [1, 2, 7, 8]

        k = sparse([164748.73734152917 24748.73734152916 -70000.0 0.0 -2.624579619658251e-28 4.2862637970157364e-12 -24748.737341529166 -24748.737341529162
                    24748.737341529155 94748.73734152915 0.0 0.0 4.2862637970157364e-12 -70000.0 -24748.73734152916 -24748.737341529155
                    -70000.0 0.0 140000.0 -4.2862637970157364e-12 0.0 0.0 -2.624579619658251e-28 4.2862637970157364e-12
                    0.0 0.0 -4.2862637970157364e-12 70000.0 0.0 0.0 4.2862637970157364e-12 -70000.0
                    -2.624579619658251e-28 4.2862637970157364e-12 0.0 0.0 94748.73734152917 24748.73734152916 -70000.0 0.0
                    4.2862637970157364e-12 -70000.0 0.0 0.0 24748.737341529155 94748.73734152915 0.0 0.0
                    -24748.737341529166 -24748.737341529162 -2.624579619658251e-28 4.2862637970157364e-12 -70000.0 0.0 119497.47468305833 -3.637978807091713e-12
                    -24748.73734152916 -24748.737341529155 4.2862637970157364e-12 -70000.0 0.0 0.0 -3.637978807091713e-12 119497.4746830583])
        @test Analise.k_estrutura_1(estrutura) ≈ Analise.k_estrutura_2(estrutura) ≈ k

        @test Analise.deslocamentos(estrutura, 1, true) ≈ Analise.deslocamentos(estrutura, 2, true) ≈ [0.0, 0.0, 4.761904761904763e-5, -0.00019817906943235845, 2.3809523809523814e-5, -7.528001090665546e-5, 0.0, 0.0, 0.0002696076408609297, -0.0002696076408609298, 0.00012675049800378677, -7.528001090665545e-5]
        @test Analise.deslocamentos(estrutura, 1, false) ≈ Analise.deslocamentos(estrutura, 2, false) ≈ [4.761904761904763e-5, -0.00019817906943235845, 2.3809523809523814e-5, -7.528001090665546e-5, 0.0002696076408609297, -0.0002696076408609298, 0.00012675049800378677, -7.528001090665545e-5]

        @test Analise.forca_interna(estrutura) ≈ [3.3340206284604434, 0.0012720448909134228, -1.6664023403833994, 7.0710761922049326, -1.6665674915761208, -4.999137587831104, 0.00018544477509152557, -7.07105622101196, -9.999339096309345]
    end

end