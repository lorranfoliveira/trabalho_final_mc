from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import numpy as np
from math import cos, sin, atan2

# Número de graus de liberdade por nó.
num_gls_no = 2


class No:
    """Define as propriedades de um  nó."""

    def __init__(self, idt: int, x: float, y: float, fx: float = 0, fy: float = 0, rx: bool = False, ry: bool = False):
        """Construtor.

        Args:
            idt: Identificação do nó.
            x: Coordenada x.
            y: Coordenada y.
            fx: Força externa na direção x.
            fy: Força externa na direção y.
            rx: Apoio na direção x. 'True' para impedido e 'False' para livre.
            ry: Apoio na direção y. 'True' para impedido e 'False' para livre.
        """
        self.idt = idt
        self.x = x
        self.y = y
        self.fx = fx
        self.fy = fy
        self.rx = rx
        self.ry = ry

    @property
    def idt(self) -> int:
        return self._idt

    @idt.setter
    def idt(self, value):
        if value < 1:
            raise ValueError(f'O número de identificação {value} não é válido!')
        else:
            self._idt = value

    def distancia(self, other) -> float:
        """Retorna a distância entre dois nós."""
        return np.linalg.norm([other.x - self.x, other.y - self.y])

    def gls_no(self) -> np.ndarray:
        """Retorna um vetor contendo os graus de liberdade do nó."""
        return np.array([num_gls_no * self.idt - 1, num_gls_no * self.idt])

    def vetor_forcas(self) -> np.ndarray:
        """Retorna um vetor contendo as forças nodais."""
        return np.array([self.fx, self.fy])

    def vetor_apoios(self) -> np.ndarray:
        """Retorna o vetor de apoios nodais."""
        return np.array([self.rx, self.ry])


class Material:
    """Define as propriedades do material que compõe os elementos."""

    def __init__(self, idt: int, e: float):
        """Construtor.

        Args:
            idt: Identificação do material.
            e: Módulo de elasticidade.
        """
        self.idt = idt
        self.e = e

    @property
    def idt(self) -> float:
        return self._idt

    @idt.setter
    def idt(self, value):
        if value < 1:
            raise ValueError('O número de identificação do material não pode ser menor que 1!')
        else:
            self._idt = value

    @property
    def e(self) -> float:
        return self._e

    @e.setter
    def e(self, value):
        if not value > 0:
            raise ValueError('O módulo de elasticidade deve ser maior que 0!')
        else:
            self._e = value


class Elemento:
    """Implementa as propriedades de um elemento de treliça."""

    def __init__(self, idt: int, no1: No, no2: No, area: float, material: Material):
        """Construtor.

        Args:
            idt: Número de identificação do elemento.
            no1: Nó inicial.
            no2: Nó final.
            area: Área da seção transversal.
            material: Material utilizado no elemento.
        """
        self.idt = idt
        self.no1 = no1
        self.no2 = no2
        self.area = area
        self.material = material

    @property
    def idt(self) -> int:
        return self._idt

    @idt.setter
    def idt(self, value):
        if value < 1:
            raise ValueError('O número de identificação do elemento não pode ser menor que 1!')
        else:
            self._idt = value

    @property
    def area(self) -> float:
        return self._area

    @area.setter
    def area(self, value):
        if value < 0:
            raise ValueError('A área da seção não pode ser menor que 0!')
        else:
            self._area = value

    def comprimento(self) -> float:
        """Retorna o comprimento do elemento."""
        return self.no1.distancia(self.no2)

    def gls_elemento(self) -> np.ndarray:
        """Retorna um vetor contendo os graus de liberdade do elemento."""
        return np.concatenate([self.no1.gls_no(),
                               self.no2.gls_no()])

    def vetor_forcas(self) -> np.ndarray:
        """Retorna o vetor de forças atuantes nos graus de liberdade do elemento."""
        return np.concatenate([self.no1.vetor_forcas(),
                               self.no2.vetor_forcas()])

    def angulo(self) -> float:
        """Retorna o ângulo de inclinação do elemento."""
        dx = self.no2.x - self.no1.x
        dy = self.no2.y - self.no1.y

        return atan2(dy, dx)

    def matriz_rotacao(self) -> np.ndarray:
        """Retorna a matriz de rotação do elemento."""
        theta = self.angulo()
        s = sin(theta)
        c = cos(theta)

        return np.array([[c, s, 0, 0],
                         [-s, c, 0, 0],
                         [0, 0, c, s],
                         [0, 0, -s, c]])

    def ke(self) -> np.ndarray:
        """Retorna a matriz de rigidez local do elemento."""
        c = self.comprimento()
        e = self.material.e
        a = self.area

        return (e * a / c) * np.array([[1, 0, -1, 0],
                                       [0, 0, 0, 0],
                                       [-1, 0, 1, 0],
                                       [0, 0, 0, 0]])

    def ke_global(self) -> np.ndarray:
        """Retorna a matriz de rigidez do elemento rotacionada para o sistema global."""
        k = self.ke()
        r = self.matriz_rotacao()

        return r.T @ k @ r


class Estrutura:
    """Define as propriedades de uma estrutura."""

    def __init__(self, nos: list[No], elementos: list[Elemento]):
        self.nos = nos
        self.elementos = elementos

    def num_gls_estrutura(self) -> int:
        """Retorna o número de graus de liberdade da estrutura."""
        return num_gls_no * len(self.nos)

    def gls_livres(self) -> np.ndarray:
        """Retorna os graus de liberdade livres."""
        return np.setdiff1d(np.arange(1, self.num_gls_estrutura() + 1), self.vetor_apoios())

    def vetor_forcas(self, incluir_gls_impedidos: bool = True):
        """Retorna o vetor de forças da estrutura."""
        forcas = np.zeros(self.num_gls_estrutura())

        for no in self.nos:
            forcas[no.gls_no() - 1] = [no.fx, no.fy]

        if incluir_gls_impedidos:
            return forcas
        else:
            return forcas[self.gls_livres() - 1]

    def vetor_apoios(self) -> np.ndarray:
        """Retorna um vetor com os graus de liberdade impedidos."""
        apoios = []

        for no in self.nos:
            vaps = no.vetor_apoios()

            if any(vaps):
                gls = no.gls_no()
                apoios += [gls[i] for i, j in enumerate(vaps) if j]

        return np.array(apoios)

    def k_estrutura_1(self) -> csc_matrix:
        """Retorna a matriz de rigidez esparsa da estrutura. Procedimento de montagem feito pela soma das
        rigidezes de cada elemento diretamente na matriz esparsa global da estrutura."""
        ngl = self.num_gls_estrutura()
        k = lil_matrix((ngl, ngl))

        for e in self.elementos:
            gle = e.gls_elemento()
            k[np.ix_(gle - 1, gle - 1)] += e.ke_global()

        # Eliminação dos graus de liberdade impedidos
        gl_liv = self.gls_livres()
        return k[np.ix_(gl_liv - 1, gl_liv - 1)].tocsc()

    def k_estrutura_2(self) -> csc_matrix:
        """Retorna a matriz de rigidez esparsa da estrutura. Procedimento de montagem com 3 vetores separados."""
        # Número de termos em cada matriz de rigidez de elementos.
        num_gle = (num_gls_no * 2)
        apoios = self.vetor_apoios()

        num_it = len(self.elementos) * num_gle ** 2

        linhas = np.zeros(num_it)
        colunas = np.zeros(num_it)
        termos = np.zeros(num_it)

        c = 0

        for elem in self.elementos:
            gle = elem.gls_elemento()
            ke = elem.ke_global()

            # Posições livres do elemento.
            pos_livres = np.array([i for i, gl in enumerate(gle) if gl not in apoios])

            if len(pos_livres) > 0:
                for i in pos_livres:
                    lin = gle[i] - len(list(filter(lambda x: x < gle[i], apoios))) - 1
                    for j in pos_livres:
                        linhas[c] = lin
                        colunas[c] = gle[j] - len(list(filter(lambda x: x < gle[j], apoios))) - 1
                        termos[c] = ke[i, j]
                        c += 1
        ngl_est = self.num_gls_estrutura() - len(apoios)
        return csc_matrix((termos, (linhas, colunas)), shape=(ngl_est, ngl_est))

    def deslocamentos(self, metodo: int = 1) -> np.ndarray:
        """Retorna o vetor dos deslocamentos nodais.

        Args:
            metodo: Método de montagem da matriz de rigidez. 1 para 'k_estrutura_1', 2 para 'k_estrutura_2'.
        """
        match metodo:
            case 1:
                kf = self.k_estrutura_1
            case _:
                kf = self.k_estrutura_2

        u = spsolve(kf(), self.vetor_forcas(False))

        uf = np.zeros(self.num_gls_estrutura())
        uf[self.gls_livres() - 1] = u

        return uf

    def nos_deslocados(self, escala: int = 1) -> list[No]:
        """Retorna um vetor contendo os nós em suas novas posições com a estrutura deformada."""
        num_nos = len(self.nos)
        u_pontos = np.reshape(self.deslocamentos(), (num_nos, 2))
        nos_desloc = []

        for i in range(num_nos):
            nos_desloc.append(No(i + 1, self.nos[i].x + escala * u_pontos[i, 0],
                                 self.nos[i].y + escala * u_pontos[i, 1]))

        return nos_desloc

    def forcas_internas(self):
        """Retorna um vetor contendo os esforços normais atuantes em cada elemento."""
        forcas = np.zeros(len(self.elementos))
        nos_def = self.nos_deslocados()

        for i, elem in enumerate(self.elementos):
            l0 = elem.comprimento()
            l1 = nos_def[elem.no1.idt - 1].distancia(nos_def[elem.no2.idt - 1])
            deform = (l1 - l0) / l0
            forcas[i] = elem.area * elem.material.e * deform

        return forcas
