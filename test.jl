include("analise.jl")

using Test
using .Analise

# =======================================================================
@testset "No" begin
    @testset "No_trelica" begin
        no₁ = No(1, -1.5, -2)
        no₂ = No(2, 3, 3, fx=1, fy=-1, fxy=1, rx=true, ry=true, rxy=true)

        # Fields
        @test no₁.id == 1
        @test no₁.x == -1.5
        @test no₁.y == -2
        @test no₁.fx == no₁.fy == 0
        @test no₁.fxy ≡ no₁.rxy ≡ nothing
        @test no₁.rx == no₁.ry == false
        # Erros
        # id
        @test_throws ArgumentError No(-1, 1, 1)
        # Compatibilidade entre fxy e rxy
        @test_throws ArgumentError No(1, 1, 1, fxy=2)
        @test_throws ArgumentError No(1, 1, 1, rxy=true)

        # Funções
        @test Analise.num_gls(no₁) == 2
        @test Analise.num_gls(no₂) == 3

        @test Analise.distancia(no₁, no₂) ≈ 6.726812023536855

    end
end

# =======================================================================
@testset "Material" begin
    # Fields
    m = Material(1, 5e9)
    #id
    @test m.id == 1
    @test m.E == 5e9

    # id
    @test_throws ArgumentError Material(0, 1e9)
    # E
    @test_throws ArgumentError Material(1, -1)

    # Métodos
end

# =======================================================================
@testset "Estrutura" begin
    # Unidades: kN, cm
    material = Material(1, 2e5)
    area = 13.796
    inercia = 8.1052e2

end
