"""
Implementa redes convolucionais no PyTorch.
AVISO: você NÃO DEVE usar ".to()" ou ".cuda()" em cada bloco de implementação.
"""
import torch
import random
from pi import Solucionador
from redes_totalmente_conectadas import *

def ola_redes_convolucionais():
  """
  Esta é uma função de exemplo que tentaremos importar e executar para garantir 
  que nosso ambiente esteja configurado corretamente no Google Colab.    
  """
  print('Olá do redes_convolucionais.py!')


class RedeConvTresCamadas(torch.nn.Module):
  """
  Uma rede convolucional de três camadas com a seguinte arquitetura:
  conv - relu - max pooling 2x2 - linear - relu - linear
  A rede opera em mini-lotes de dados que têm shape (N, C, H, W)
  consistindo em N imagens, cada uma com altura H e largura W e com C
  canais de entrada.
  """

  def __init__(self, dims_entrada=(3, 32, 32), num_filtros=32, tamanho_filtro=7,
               dim_oculta=100, num_classes=10, escala_peso=1e-3):
    """
    Inicializa a nova rede.
    Entrada:
    - dims_entrada: Tupla (C, H, W) indicando o tamanho dos dados de entrada
    - num_filtros: Número de filtros a serem usados na camada de convolução
    - tamanho_filtro: Largura/altura dos filtros a serem usados na camada de convolução
    - dim_oculta: Número de unidades a serem usadas na camada oculta totalmente conectada
    - num_classes: Número de pontuações a serem produzidas na camada linear final.
    - escala_peso: Escalar indicando o desvio padrão para inicialização 
      aleatória de pesos.
    """
    super().__init__()

    # redefine a semente antes de começar
    random.seed(0)
    torch.manual_seed(0)
    
    self.escala_peso = escala_peso

    ########################################################################
    # TODO: Inicialize pesos, vieses para a rede convolucional de três     #
    # camadas. Os pesos devem ser inicializados a partir de uma Gaussiana  #
    # centrada em 0,0 com desvio padrão igual a escala_peso; vieses devem  #
    # ser inicializados com zero.                                          #
    #                                                                      #
    # IMPORTANTE: Para esta tarefa, você pode assumir que o preenchimento  #
    # e o passo da primeira camada de convolução são escolhidos para que   #
    # **a largura e a altura da entrada sejam preservada**.                #
    ########################################################################
    # Substitua a comando "pass" pelo seu código
    pass
    ########################################################################
    #                           FIM DO SEU CODIGO                          #
    ########################################################################

    self.reset_parameters()

  def forward(self, X):
    """
    Executa o passo para frente da rede para calcular as pontuações de classe.

    Entrada:
    - X: Dados de entrada de shape (N, D). Cada X[i] é uma amostra de treinamento.

    Retorno: 
    - pontuacoes: Tensor de shape (N, C) contendo as pontuações de classe para X
    """
    # Calcule o passo para frente
    pontuacoes = None
    ########################################################################
    # TODO: Implemente o passo para frente para rede convolucional de três #
    # camadas, calculando as pontuações de classe para X e armazenando-as  #
    # na variável pontuações.                                              #
    ########################################################################
    # Substitua a comando "pass" pelo seu código
    pass
    ########################################################################
    #                           FIM DO SEU CODIGO                          #
    ########################################################################    
    
    return pontuacoes
    
  def reset_parameters(self):
    """
    Inicializa os pesos e vieses das camadas convolucionais e totalmente conectadas.
    """
    for param in self.parameters():
      if isinstance(param, torch.nn.Conv2d) or isinstance(param, torch.nn.Linear):
        torch.nn.init.normal_(param.weight, std=self.escala_peso)
        torch.nn.init.zeros_(param.bias)


class RedeConvProfunda(torch.nn.Module):
  """
  Uma rede neural convolucional com um número arbitrário de camadas de 
  convolução no estilo da rede VGG. Todas as camadas de convolução usarão 
  filtro de tamanho 3 e preenchimento de 1 para preservar o tamanho do mapa 
  de ativação, e todas as camadas de agrupamento serão camadas de agrupamento 
  por máximo com campos receptivos de 2x2 e um passo de 2 para reduzir pela 
  metade o tamanho do mapa de ativação.

  A rede terá a seguinte arquitetura:

  {conv - [normlote?] - relu - [agrup?]} x (L - 1) - linear

  Cada estrutura {...} é uma "camada macro" que consiste em uma camada de 
  convolução, uma camada de normalização de lote opcional, uma não linearidade 
  ReLU e uma camada de agrupamento opcional. Depois de L-1 dessas macrocamadas, 
  uma única camada totalmente conectada é usada para prever pontuações de classe.

  A rede opera em minilotes de dados que possuem shape (N, C, H, W) consistindo 
  de N imagens, cada uma com altura H e largura W e com C canais de entrada.
  """
  def __init__(self, dims_entrada=(3, 32, 32),
               num_filtros=[8, 8, 8, 8, 8],
               agrups_max=[0, 1, 2, 3, 4],
               normlote=False,
               num_classes=10, escala_peso=1e-3):
    """
    Inicializa uma nova rede.

    Entrada:
    - dims_entrada: Tupla (C, H, W) indicando o tamanho dos dados de entrada
    - num_filtros: Lista de comprimento (L - 1) contendo o número de filtros
      de convolução para usar em cada macrocamada.
    - agrups_max: Lista de inteiros contendo os índices (começando em zero) das 
      macrocamadas que devem ter agrupamento por máximo.
    - normlote: Booleano dizendo se normalização do lote deve ou não ser 
      incluída em cada macrocamada.
    - num_classes: Número de pontuações a serem produzidas na camada linear final.
    - escala_peso: Escalar indicando o desvio padrão para inicialização 
      aleatória de pesos, ou a string "kaiming" para usar a inicialização Kaiming.
    """
    super().__init__()

    # redefine a semente antes de começar
    random.seed(0)
    torch.manual_seed(0)
    
    self.num_camadas = len(num_filtros)+1
    self.escala_peso = escala_peso
    self.agrups_max = agrups_max
    self.normlote = normlote

    #######################################################################
    # TODO: Inicialize os parâmetros para o RedeConvProfunda.             #
    #                                                                     #
    # Pesos para camadas de convolução e totalmente conectadas devem ser  #
    # inicializados de acordo com escala_peso. Os vieses devem ser        #
    # inicializados com zero. Parâmetros de escala (gamma) e deslocamento #
    # (beta) de camadas de normalização de lote devem ser inicializados   #
    # com um e zero, respectivamente.                                     #
    #######################################################################
    # Substitua a comando "pass" pelo seu código
    pass
    #######################################################################
    #                           FIM DO SEU CODIGO                         #
    #######################################################################

    # Verifique se obtivemos o número correto de parâmetros
    if not self.normlote:
      params_por_camada_macro = 2  # peso e viés
    else:
      params_por_camada_macro = 4  # peso, viés, escala, deslocamento
    num_params = params_por_camada_macro * len(num_filtros) + 2
    msg = 'self.parameters() tem o número errado de ' \
          'elementos. Obteve %d; esperava %d'
    msg = msg % (len(list(self.parameters())), num_params)
    assert len(list(self.parameters())) == num_params, msg
    
    self.reset_parameters()

  def forward(self, X):
    """
    Executa o passo para frente da rede para calcular as pontuações de classe.

    Entrada:
    - X: Dados de entrada de shape (N, D). Cada X[i] é uma amostra de treinamento.

    Retorno: 
    - pontuacoes: Tensor de shape (N, C) contendo as pontuações de classe para X
    """
    # Calcule o passo para frente
    pontuacoes = None
    ##################################################################
    # TODO: Implemente o passo para frente para a RedeConvProfunda,  #
    # calculando as pontuações de classe para X e armazenando-as na  #
    # variável pontuacoes.                                           #
    ##################################################################
    # Substitua a comando "pass" pelo seu código
    pass
    ##################################################################
    #                        FIM DO SEU CODIGO                       #
    ##################################################################    
    
    return pontuacoes
    
  def reset_parameters(self):
    """
    Inicializa os pesos e vieses das camadas convolucionais e totalmente conectadas.
    """
    for nome, camada in self.named_modules():
      if isinstance(camada, torch.nn.Conv2d) or isinstance(camada, torch.nn.Linear):
        if isinstance(self.escala_peso, str) and self.escala_peso == "kaiming":
          ############################################################################
          # TODO: Inicializa os pesos das camadas de convolução e lineares usando o  #
          # método de Kaiming.                                                       #
          ############################################################################
          # Substitua a comando "pass" pelo seu código
          pass
          ############################################################################
          #                             FIM DO SEU CODIGO                            #
          ############################################################################          
        else:
          torch.nn.init.normal_(camada.weight, std=self.escala_peso)
        torch.nn.init.zeros_(camada.bias)


def encontrar_parametros_sobreajuste():
  taxa_aprendizagem = 1e-5  # Tente com este!
  escala_peso = 2e-3   # Tente com este!
  ###############################################################
  # TODO: Altere escala_peso e taxa_aprendizagem para que seu   #
  # modelo atinja 100% de acurácia de treinamento em 30 épocas. #
  ###############################################################
  # Substitua a instrução "pass" pelo seu código
  pass
  ###############################################################
  #                        FIM DO SEU CODIGO                    #
  ###############################################################
  return escala_peso, taxa_aprendizagem


def criar_instancia_solucionador_convolucional(dic_dados, device):
  modelo = None
  solucionador = None
  #########################################################
  # TODO: Treine a melhor RedeConvProfunda possível na    #
  # CIFAR-10 em 60 segundos.                              #
  #########################################################
  # Substitua a instrução "pass" pelo seu código
  pass
  #########################################################
  #                  FIM DO SEU CODIGO                    #
  #########################################################
  return solucionador
