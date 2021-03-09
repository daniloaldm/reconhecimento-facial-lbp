# PDI

## Instalação e Execução
Clone o repositório:
```
git clone https://github.com/daniloaldm/reconhecimento-facial-lbp.git
```
Vá para o diretório:
```
cd reconhecimento-facial-lbp
```
Execute:
```
python3 -m venv .
. bin/activate
pip3 install -r requirements.txt
```
Para executar:
```
python3 knn.py
```

# Reconhecimento facial com LBP e NN
Este repositório contém o código-fonte dos experimentos realizados no artigo intitulado *** "Reconhecimento facial usando padrão binário local e classificação do vizinho mais próximo" ***.[2018 International Advanced Intelligent Computing Symposium (SAIN 2018)] (http://sain.ijain.org/) em 29-30 de agosto de 2018 em Yogyakarta, Indonésia.

## Estrutura do código fonte
- ** pasta db ** Contém três conjuntos de dados usados ​​neste experimento, a saber, JAFFE, AT&T e Yale
- ** localbinarypatterns.py ** A implementação LBP usando o módulo scikit-learn
- ** knn.py ** A implementação do sistema usando k-NN (k vizinho mais próximo)
- ** rnn.py ** A implementação do sistema usando RNN (Radius Nearest Neighbour)
- ** akurasi.py ** Analise a precisão do sistema (taxa de reconhecimento) em termos de parâmetros (número de pontos vizinhos P, o raio P e o valor k escolhido de k-NN)
- ** plotting.py ** Traçar o histograma de vetor de característica da saída LBP
- ** pasta de resultados ** Contém resultados de experimentos para k-NN e RNN, salvos em arquivos csv
