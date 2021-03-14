# importando os pacotes necessários
from skimage import feature
import numpy as np

class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# número de pontos e raio
		self.numPoints = numPoints
		self.radius = radius

	def describe(self, image, eps=1e-7):
		# calcular a representação do padrão binário local
		# da imagem e, em seguida, usar a representação LBP
		# para construir o histograma de padrões
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))

		# normaliza o histograma
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)

		# retorna o histograma dos padrões binários locais
		return hist
