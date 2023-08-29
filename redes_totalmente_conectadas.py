"""
Implementa redes neurais totalmente conectadas no PyTorch.
AVISO: você NÃO DEVE usar ".to()" ou ".cuda()" em cada bloco de implementação.
"""
from sys import exec_prefix, executable
import torch
import random
from pi import Solucionador
from torch.nn.functional import cross_entropy


def ola_redes_totalmente_conectadas():
  """
  Esta é uma função de exemplo que tentaremos importar e executar para garantir 
  que nosso ambiente esteja configurado corretamente no Google Colab.    
  """
  print('Olá do redes_totalmente_conectadas.py!')


class ConjuntoDeDados(torch.utils.data.Dataset):
  
  def __init__(self, X, y):
    """
    Inicializa o conjunto de dados.
    Entrada:
    - X: Tensor de dados de entrada de shape (N, d_1, ..., d_k)
    - y: int64 Tensor de rótulos, e shape (N,). y[i] fornece o rótulo para X[i].
    """
    self.X = None
    self.y = None
    ####################################################################
    # TODO: Armazene os parâmetros X e y nos atributos self.X e self.y #
    ####################################################################
    # Substitua a comando "pass" pelo seu código
    self.X = X
    self.y = y


    ####################################################################
    #                         FIM DO SEU CODIGO                        #
    ####################################################################

  def __getitem__(self, i):
    """
    Retorna a i-esima amostra do conjunto de dados.
    Entrada:
    - i: inteiro indicando o número da amostra desejada
    Retorno:
    - amostra: tupla contendo os dados e o rótulo da i-ésima amostra
    """
    amostra = None
    ################################################################## 
    # TODO: Armazene uma tupla com dados e rótulo da i-ésima amostra #
    ##################################################################
    # Substitua a comando "pass" pelo seu código
    amostra = [self.X[i], self.y[i]]
    ##################################################################
    #                        FIM DO SEU CODIGO                       #
    ##################################################################
    return amostra

  def __len__(self):
    """
    Retorna o número total de amostras no conjunto de dados.
    Retorno:
    - num_amostras: inteiro indicando o número total de amostras.
    """
    num_entradas = None
    ################################################################## 
    # TODO: Armazene um inteiro contendo o número total de amostras. #
    ##################################################################
    # Substitua a comando "pass" pelo seu código
    num_entradas = self.X.shape[0]
    ##################################################################
    #                        FIM DO SEU CODIGO                       #
    ##################################################################
    return num_entradas
    

class RedeDuasCamadas(torch.nn.Module):
  """
  Uma rede neural totalmente conectada de duas camadas com não linearidade ReLU. 
  Assumimos uma dimensão de entrada de D, uma dimensão oculta de H, e realizar a 
  classificação em C classes. A arquitetura deve ser linear - relu - linear.
  Observe que esta classe não implementa a decida de gradiente; em vez disso, ela
  irá interagir com um objeto separado que é responsável por executar a otimização.
  """

  def __init__(self, dim_entrada=3*32*32, dim_oculta=100, num_classes=10,
               escala_peso=1e-3, dtype=torch.float32, device='cpu'):
    """
    Inicializa o modelo. Pesos são inicializados com pequenos valores aleatórios 
    e vieses são inicializados com zero.            

    W1: Pesos da primeira camada; tem shape (D, H)
    b1: Vieses da primeira camada; tem shape (H,)
    W2: Pesos da segunda camada; tem shape (H, C)
    b2: Vieses da segunda camada; tem shape (C,)

    Entrada:
    - dim_entrada: A dimensão D dos dados de entrada.
    - dim_oculta: O número de neurônios H na camada oculta.
    - num_classes: O número de categorias C.
    - escala_peso: Escalar indicando o desvio padrão para inicialização dos pesos
    - dtype: Opcional, tipo de dados de cada parâmetro de peso.
    - device: Opcional, se os parâmetros de peso estão na GPU ou CPU.
    """
    super().__init__()

    self.W1 = None
    self.b1 = None
    self.W2 = None
    self.b2 = None
    ####################################################################
    # TODO: Inicialize os pesos e vieses da rede de duas camadas. Os   #
    # pesos devem ser inicializados a partir de uma Gaussiana centrada #
    # em 0,0 com desvio padrão igual a escala_peso e os vieses devem   #
    # ser inicializados com zero. Todos os pesos e vieses devem ser    #
    # armazenados no dicionário self.params, com os pesos e vieses da  #
    # primeira camada usando as chaves 'W1' e 'b1' e os pesos e vieses #
    # da segunda camada usando as chaves 'W2' e 'b2'.                  #
    ####################################################################
    # Substitua a comando "pass" pelo seu código
    #pass
    self.b1 = torch.nn.Parameter(torch.zeros(dim_oculta,dtype=dtype, device=device))
    self.b2 = torch.nn.Parameter(torch.zeros(num_classes,dtype=dtype, device=device))
    self.W1 = torch.nn.Parameter(torch.normal(0, escala_peso, size=(dim_entrada,dim_oculta),dtype=dtype, device=device))
    self.W2 = torch.nn.Parameter(torch.normal(0, escala_peso, size=(dim_oculta, num_classes),dtype=dtype, device=device))
    self.params = {}
    self.params['W1'] = self.W1
    self.params['W2'] = self.W2
    self.params['b1'] = self.b1
    self.params['b2'] = self.b2
    ####################################################################
    #                         FIM DO SEU CODIGO                        #
    ####################################################################

  def forward(self, X):
    """
    Executa o passo para frente da rede para calcular as pontuações de classe. 
    A arquitetura da rede deve ser:

    camada linear -> ReLU (rep_oculta) -> camada linear (pontuações)

    Entrada:
    - X: Dados de entrada de shape (N, D). Cada X[i] é uma amostra de treinamento.

    Retorno: 
    - pontuacoes: Tensor de shape (N, C) contendo pontuações de classe, em
      que pontuacoes[i, c] é a pontuação de categoria para X[i] e classe c.
    """
    # Calcule o passo para frente
    pontuacoes = None
    #############################################################
    # TODO: Implemente o passo para frente em uma rede de duas  #
    # camadas, calculando as pontuações de classe para X e      #
    # armazenando-as na variável de pontuações.                 #
    #############################################################
    # Substitua a comando "pass" pelo seu código
    #pass
    P = torch.add(torch.mm(X,self.W1), self.b1) #linear1 XW+ b
    P = torch.max(torch.zeros_like(P),P) #RElu
    pontuacoes = torch.add(torch.mm(P,self.W2), self.b2) 
    
    #############################################################
    #                      FIM DO SEU CODIGO                    #
    #############################################################    
    
    return pontuacoes

    
def perda_softmax(x, y):
  """
  Calcula a perda usando a classificação softmax.
    
  Entrada: 
  - x: Dados de entrada, de shape (N, C), onde x[i, j] é a pontuação para a
    j-ésima classe para a i-ésima amostra de treinamento.
  - y: Vetor de rótulos, de shape (N,), onde y[i] é o rótulo para x[i] e 
    0 <= y[i] < C.
    
  Retorno:
  - perda: escalar fornecendo a perda
  """
  # Calcular a perda
  perda = None
  ###############################################################################
  # TODO: Calcule a perda com base nas pontuações obtidas pelo modelo. Armazene #
  # o resultado na variável "perda", que deve ser um escalar. Use a perda do    # 
  # classificador Softmax.                                                      #
  ###############################################################################
  # Substitua a comando "pass" pelo seu código
  #pass

  perda = torch.sum(-torch.log(torch.exp(torch.diagonal(x[:,y[:]], 0)) / torch.exp(x).sum(dim = 1))) / x.shape[0]

  #x.shape[0] esta pegando as linhas
  #https://pytorch.org/docs/stable/generated/torch.diagonal.html 
  # a funçao cross_entropy_loss talvez podesse ser usada aqui
  
  
  ###############################################################################
  #                               FIM DO SEU CODIGO                             #
  ###############################################################################
  
  return perda  


class RedeTotalmenteConectada(torch.nn.Module):
  """
  Uma rede neural totalmente conectada com um número arbitrário de camadas 
  ocultas e ativações ReLU. Para uma rede com camadas L, a arquitetura será:

  {linear - relu - [descarte]} x (L - 1) - linear

  onde descarte é opcional, e o bloco {...} é repetido L - 1 vezes.
  """

  def __init__(self, dims_oculta, dim_entrada=3*32*32, num_classes=10, 
               descarte=0.0, escala_peso=1e-2):
    """
    Inicialize uma nova RedeTotalmenteConectada.

    Entradas:
    - dims_oculta: uma lista de inteiros indicando o tamanho de cada camada oculta.
    - dim_entrada: um inteiro indicando o tamanho da entrada.
    - num_classes: Um inteiro indicando o número de categorias a serem classificadas.
    - descarte: escalar entre 0 e 1 indicando a probabilidade de descarte para redes
      com camadas de descarte. Se descarte = 0, então a rede não deve usar descarte.
    - escala_peso: Escalar indicando o desvio padrão para inicialização dos pesos
    """
    super().__init__()

    # redefine a semente antes de começar
    random.seed(0)
    torch.manual_seed(0)
    
    self.usar_descarte = descarte != 0
    self.escala_peso = escala_peso
    self.num_camadas = 1 + len(dims_oculta)

    ############################################################################
    # TODO: Inicializa os parâmetros da rede.                                  #
    ############################################################################
    # Substitua a comando "pass" pelo seu código
    #pass
    ##########################################Arrumar########################
               ###########################################################
    if(self.usar_descarte):
      self.fc1 = torch.nn.Dropout(p = descarte) 
    

    
    n = 1
    self.b1 = torch.nn.Parameter(torch.zeros(dims_oculta[0]))
    self.W1 = torch.nn.Parameter(torch.normal(0, escala_peso, size=(dim_entrada,dims_oculta[0])))
    while(n < self.num_camadas-1):
     #vars('self.b'+str(n+1)) = torch.nn.Parameter(torch.zeros(dims_oculta[n]))
     # vars()['self.W'+str(n+1)] = torch.nn.Parameter(torch.normal(0, escala_peso, size=(dims_oculta[n-1],dims_oculta[n])))
     exec('self.b%d  = torch.nn.Parameter(torch.zeros(dims_oculta[n]))' % (n+1,))
     exec('self.W%d  = torch.nn.Parameter(torch.normal(0, escala_peso, size=(dims_oculta[n-1],dims_oculta[n])))' % (n+1,))
     n = n + 1
    #print(self.W2)
    exec('self.b%d = torch.nn.Parameter(torch.zeros(num_classes))' % (n+1,))

    exec('self.W%d = torch.nn.Parameter(torch.normal(0, escala_peso, size=(dims_oculta[n-1],num_classes)))'
    % (n+1,))
    

    ############################################################################
    #                             FIM DO SEU CODIGO                            #
    ############################################################################

    self.reset_parameters()

  def forward(self, X): ########arrumar ###################################
    """
    Executa o passo para frente da rede para calcular as pontuações de classe.

    Entrada:
    - X: Dados de entrada de shape (N, D). Cada X[i] é uma amostra de treinamento.

    Retorno: 
    - pontuacoes: Tensor de shape (N, C) contendo as pontuações de classe para X
    """
    # Calcule o passo para frente
    pontuacoes = None
    ############################################################################
    # TODO: Execute o passo para frente, calculando as pontuações de classe    #
    # para a entrada. Armazene o resultado na variável "pontuacoes", que deve  #
    # ser um tensor de shape (N, C).                                           #
    ############################################################################
    # Substitua a comando "pass" pelo seu código
    
    n = 0
    P = X
    while(n < self.num_camadas-1):
    #for n in range(self.num_camadas-1):
      P = torch.add(torch.mm(P,eval('self.W%d'%(n+1))),eval('self.b%d'%(n+1))) #linear1 XW+ b
      P = torch.nn.functional.relu(P) #relu no colab n tem restição nenhuma pra usar essa função
      if(self.usar_descarte):
        P = self.fc1(P)
      n = n+1
    pontuacoes = torch.add(torch.mm(P,eval('self.W%d'%(n+1))), eval('self.b%d'%(n+1)))

    ############################################################################
    #                             FIM DO SEU CODIGO                            #
    ############################################################################    
    
    return pontuacoes

  def reset_parameters(self):
    """
    Inicializa os pesos e vieses das camadas totalmente conectadas.
    """
    for nome, camada in self.named_modules():
      if isinstance(camada, torch.nn.Linear):
        torch.nn.init.normal_(camada.weight, std=self.escala_peso)
        torch.nn.init.zeros_(camada.bias)


def criar_instancia_solucionador(dic_dados, dtype, device):
  modelo = RedeDuasCamadas(dim_oculta=200, dtype=dtype, device=device)
  #############################################################
  # TODO: Use uma instância do Solucionador para treinar uma  #
  # RedeDuasCamadas que atinja pelo menos 50% de acurácia no  #
  # conjunto de validação.                                    #
  #############################################################
  solucionador = None
  # Substitua a instrução "pass" pelo seu código
  #pass
  #Solucionador(modelo, funcao_de_perda, otimizador, dados, **kwargs)
  # Define um otimizador SGD para atualizar os parâmetros
  otimizador = torch.optim.SGD(modelo.parameters(), lr=0.8, weight_decay=0.001)

  #Define um escalonador exponencial para ajustar a taxa de aprendizado
  escalonador = torch.optim.lr_scheduler.ExponentialLR(otimizador, 0.48)
  dados_de_treinamento = ConjuntoDeDados(dic_dados['X_treino'], dic_dados['y_treino'])
  dados_de_validacao = ConjuntoDeDados(dic_dados['X_val'], dic_dados['y_val'])
  solucionador = Solucionador(modelo, perda_softmax, otimizador,
  dados={'treinamento': dados_de_treinamento, 
  'validacao': dados_de_validacao}, escalonador=escalonador, device =device, imprime_cada = 500)
  

#ver se alguma das duas esta mais ou menos correta

  #############################################################
  #                    FIM DO SEU CODIGO                      #
  #############################################################
  return solucionador


def retorna_params_rede_tres_camadas():
  ###############################################################
  # TODO: Altere escala_peso e taxa_aprendizagem para que seu   #
  # modelo atinja 100% de acurácia de treinamento em 20 épocas. #
  ###############################################################
  taxa_aprendizagem = 1e-2   # Tente com este!
  escala_peso = 1e-4  # Tente com este!
  # Substitua a instrução "pass" pelo seu código
  taxa_aprendizagem = 0.083   
  escala_peso = 0.2  
  ################################################################
  #                       FIM DO SEU CODIGO                      #
  ################################################################
  return escala_peso, taxa_aprendizagem


def retorna_params_rede_cinco_camadas():
  ###############################################################
  # TODO: Altere escala_peso e taxa_aprendizagem para que seu   #
  # modelo atinja 100% de acurácia de treinamento em 20 épocas. #
  ###############################################################
  taxa_aprendizagem = 1e-2   # Tente com este!
  escala_peso = 1e-4  # Tente com este!
  # Substitua a instrução "pass" pelo seu código
  taxa_aprendizagem = 0.32   # Tente com este!
  escala_peso = 0.08  # Tente com este!
  ################################################################
  #                       FIM DO SEU CODIGO                      #
  ################################################################
  return escala_peso, taxa_aprendizagem
  
