import unittest

import numpy as np
from scipy.sparse import csc_matrix

from analise import No, Material, Elemento, Estrutura

nos = [No(1, 1, -1, rx=True, ry=True),
       No(2, 3, -1),
       No(3, 5, -1),
       No(4, 7, -1, rx=True, ry=True),
       No(5, 3, 1, fx=10, fy=-5),
       No(6, 5, 1)]

material = Material(1, 100e6)

area = 0.0014
elementos = [Elemento(1, nos[0], nos[1], area, material),
             Elemento(2, nos[0], nos[4], area, material),
             Elemento(3, nos[1], nos[2], area, material),
             Elemento(4, nos[1], nos[5], area, material),
             Elemento(5, nos[3], nos[2], area, material),
             Elemento(6, nos[4], nos[1], area, material),
             Elemento(7, nos[5], nos[2], area, material),
             Elemento(8, nos[5], nos[3], area, material),
             Elemento(9, nos[5], nos[4], area, material)]

estrutura = Estrutura(nos, elementos)

k = np.array([[164748.73734152917, 24748.73734152916, -70000.0, 0.0, -2.624579619658251e-28, 4.2862637970157364e-12,
               -24748.737341529166, -24748.737341529162],
              [24748.737341529155, 94748.73734152915, 0.0, 0.0, 4.2862637970157364e-12, -70000.0, -24748.73734152916,
               -24748.737341529155],
              [-70000.0, 0.0, 140000.0, -4.2862637970157364e-12, 0.0, 0.0, -2.624579619658251e-28,
               4.2862637970157364e-12],
              [0.0, 0.0, -4.2862637970157364e-12, 70000.0, 0.0, 0.0, 4.2862637970157364e-12, -70000.0],
              [-2.624579619658251e-28, 4.2862637970157364e-12, 0.0, 0.0, 94748.73734152917, 24748.73734152916, -70000.0,
               0.0],
              [4.2862637970157364e-12, -70000.0, 0.0, 0.0, 24748.737341529155, 94748.73734152915, 0.0, 0.0],
              [-24748.737341529166, -24748.737341529162, -2.624579619658251e-28, 4.2862637970157364e-12, -70000.0, 0.0,
               119497.47468305833, -3.637978807091713e-12],
              [-24748.73734152916, -24748.737341529155, 4.2862637970157364e-12, -70000.0, 0.0, 0.0,
               -3.637978807091713e-12, 119497.4746830583]]
             )


class TestNo(unittest.TestCase):
    def test_entrada(self):
        self.assertEqual(nos[1].idt, 2)
        self.assertEqual(nos[1].x, 3)
        self.assertEqual(nos[1].y, -1)
        self.assertEqual(nos[1].fx, 0)
        self.assertEqual(nos[1].fy, 0)
        self.assertFalse(nos[1].rx)
        self.assertFalse(nos[1].ry)

        with self.assertRaises(ValueError):
            No(0, 3, -1)

    def test_distancia(self):
        self.assertAlmostEqual(nos[0].distancia(nos[4]), 2.8284271247461903)

    def test_gls_no(self):
        self.assertTrue(np.allclose(nos[2].gls_no(), [5, 6]))

    def test_vetor_forcas(self):
        self.assertTrue(np.allclose(nos[0].vetor_forcas(), [0, 0]))
        self.assertTrue(np.allclose(nos[4].vetor_forcas(), [10, -5]))

    def test_vetor_apoios(self):
        self.assertTrue(np.allclose(nos[0].vetor_apoios(), [True, True]))
        self.assertTrue(np.allclose(nos[1].vetor_apoios(), [False, False]))


class TestMaterial(unittest.TestCase):
    def test_entrada(self):
        self.assertEqual(material.idt, 1)
        self.assertEqual(material.e, 100e6)

        with self.assertRaises(ValueError):
            Material(-1, 100e6)

        with self.assertRaises(ValueError):
            Material(1, -100e6)


class TestElemento(unittest.TestCase):
    def test_entrada(self):
        self.assertEqual(elementos[1].idt, 2)
        self.assertEqual(elementos[1].no1, nos[0])
        self.assertEqual(elementos[1].no2, nos[4])
        self.assertAlmostEqual(elementos[1].area, area)
        self.assertEqual(elementos[1].material, material)

    def test_comprimento(self):
        self.assertAlmostEqual(elementos[1].comprimento(), 2.8284271247461903)

    def test_gls_elemento(self):
        self.assertTrue(np.allclose(elementos[1].gls_elemento(), [1, 2, 9, 10]))

    def test_vetor_forcas(self):
        self.assertTrue(np.allclose(elementos[1].vetor_forcas(), [0, 0, 10, -5]))
        self.assertTrue(np.allclose(elementos[8].vetor_forcas(), [0, 0, 10, -5]))
        self.assertTrue(np.allclose(elementos[3].vetor_forcas(), [0, 0, 0, 0]))

    def test_angulo(self):
        self.assertAlmostEqual(elementos[3].angulo(), np.pi / 4)
        self.assertAlmostEqual(elementos[8].angulo(), np.pi)
        self.assertAlmostEqual(elementos[7].angulo(), -np.pi / 4)

    def test_matriz_rotacao(self):
        t = np.sqrt(2) / 2
        r = np.array([[t, t, 0, 0],
                      [-t, t, 0, 0],
                      [0, 0, t, t],
                      [0, 0, -t, t]])

        self.assertTrue(np.allclose(elementos[3].matriz_rotacao(), r))

    def test_ke(self):
        ke = np.array([[49497.474683058324, 0, -49497.474683058324, 0],
                       [0, 0, 0, 0],
                       [-49497.474683058324, 0, 49497.474683058324, 0],
                       [0, 0, 0, 0]])

        self.assertTrue(np.allclose(elementos[3].ke(), ke))

    def test_ke_global(self):
        ke = np.array([[24748.737341529166, 24748.737341529162, -24748.737341529166, -24748.737341529162],
                       [24748.73734152916, 24748.737341529155, -24748.73734152916, -24748.737341529155],
                       [-24748.737341529166, -24748.737341529162, 24748.737341529166, 24748.737341529162],
                       [-24748.73734152916, -24748.737341529155, 24748.73734152916, 24748.737341529155]])

        self.assertTrue(np.allclose(elementos[3].ke_global(), ke))


class TestEstrutura(unittest.TestCase):
    def test_entrada(self):
        self.assertTrue(estrutura.nos is nos)
        self.assertTrue(estrutura.elementos is elementos)

    def test_num_gls_estrutura(self):
        self.assertEqual(estrutura.num_gls_estrutura(), 12)

    def test_gls_livres(self):
        self.assertTrue(np.allclose(estrutura.gls_livres(), [3, 4, 5, 6, 9, 10, 11, 12]))

    def test_vetor_forcas(self):
        self.assertTrue(np.allclose(estrutura.vetor_forcas(), [0, 0, 0, 0, 0, 0, 0, 0, 10, -5, 0, 0]))
        self.assertTrue(np.allclose(estrutura.vetor_forcas(False), [0, 0, 0, 0, 10, -5, 0, 0]))

    def test_vetor_apoios(self):
        self.assertTrue(np.allclose(estrutura.vetor_apoios(), [1, 2, 7, 8]))

    def test_k_estrutura_1(self):
        self.assertTrue(np.allclose(estrutura.k_estrutura_1().toarray(), k))

    def test_k_estrutura_2(self):
        self.assertTrue(np.allclose(estrutura.k_estrutura_2().toarray(), k))
        self.assertTrue(np.allclose(estrutura.k_estrutura_2().toarray(), estrutura.k_estrutura_1().toarray()))

    def test_deslocamentos(self):
        desloc = [0.0, 0.0, 4.761904761904763e-5, -0.00019817906943235845, 2.3809523809523814e-5, -7.528001090665546e-5,
                  0.0, 0.0, 0.0002696076408609297, -0.0002696076408609298, 0.00012675049800378677,
                  -7.528001090665545e-5]

        self.assertTrue(np.allclose(estrutura.deslocamentos(), desloc))
        self.assertTrue(np.allclose(estrutura.deslocamentos(2), desloc))

    def test_forcas_internas(self):
        f = [3.3340206284604434, 0.0012720448909134228, -1.6664023403833994, 7.0710761922049326, -1.6665674915761208,
             -4.999137587831104, 0.00018544477509152557, -7.07105622101196, -9.999339096309345]
        self.assertTrue(np.allclose(estrutura.forcas_internas(), f))


if __name__ == '__main__':
    unittest.main()
