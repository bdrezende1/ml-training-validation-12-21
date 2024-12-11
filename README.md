# ml-training-validation-12-21
O código é parte de um script Python dedicado a análise de dados e validação de modelos de aprendizado de máquina, utilizando técnicas como *cross-validation*. Vou explicar brevemente algumas partes importantes do arquivo:

### Importação de bibliotecas
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```
Essas bibliotecas são usadas para:
- `pandas`: Manipulação de dados em formato tabular.
- `numpy`: Cálculos numéricos eficientes.
- `matplotlib.pyplot`: Visualização gráfica.

### Configuração de gráficos
```python
plt.rcParams['figure.figsize'] = (8, 8)
```
Configura o tamanho padrão das figuras para 8x8.

### Carregamento e amostragem de dados
```python
tic_tac_toe = pd.read_csv('./tic-tac-toe.csv')
sample1 = tic_tac_toe.sample(200, random_state=42)
sample2 = tic_tac_toe.sample(200, random_state=3)
```
- Os dados são carregados de um arquivo chamado `tic-tac-toe.csv`.
- São criadas duas amostras aleatórias de 200 observações com diferentes *seeds* (42 e 3).

### Comparação entre amostras
```python
print(len([index for index in sample1.index if index in sample2.index]))
print(sample1['Class'].value_counts())
print(sample2['Class'].value_counts())
```
- Calcula o número de observações comuns entre as duas amostras.
- Imprime a distribuição de classes (`Class`) em cada amostra.

### Cross-validation
O restante do script aborda técnicas como validação cruzada para evitar problemas de generalização e avaliar a robustez dos modelos. Essa abordagem divide os dados em diferentes subconjuntos para treinar e testar modelos repetidamente.
