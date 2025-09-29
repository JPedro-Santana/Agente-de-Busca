# Agente-de-Busca
Programa que sem utilizar nenhuma biblioteca extra das presentes na linguagem, automatiza um agente.

Atividade:
Crie um programa, na linguagem java ou python. 
Sem utilizar NENHUMA biblioteca extra das presentes na linguagem, que automatize um agente com funções abaixo. 
Use uma classe para implementar seu agente.

Sensor:
- getSensor() : retorna uma matriz de caracteres de dimensão 3x3. 
Para posições fronteira [(0,0), (0,1), (0,2), (2,0), (3,0), (3,1), (3,2)] 
podem ter 'X' - parede, 'o' - comida, '_' - corredor, 'E' - entrada e 'S' para saída.. 
Para posição (2,2), o valor representa para onde o agente está girado. 
Os valores podem ser 'N' - norte, 'S'- sul, 'L'- leste, 'O' - oeste.

Atuadores:
- setDirection(dir): muda a direção do agente considerando os valores 'N' - norte, 'S'- sul, 'L'- leste, 'O' - oeste.
- move() : move o agente para direção que o agente está apontado.

Instruções:
Voce deve implementar as funções de sensores e implementar o ambiente onde o agente vai rodar. 
Tanto o ambiente quanto o agente deve ser implementado em classes separadas. 
Programa deve receber um arquivo TXT representando o ambiente e o agente deve ser instanciado com a informação de quantas 'comidas' tem dentro do labirinto. 
O objetivo é pegar todas as comidas e ir para saída. 
O agente não deve 'conhecer' o mapa a priori. 
Ele pode ir criando uma 'memória' do mapa enquanto anda. 
No final deve imprimir o total de recompensas ganho usando a seguinte tabela: 1 comida = 10pts, 1 passo = -1pts. 
Os labirintos pode ter tamanhos variáveis. 
Voce deve gerar um video com seu agente caminhando pelo labirinto. (https://opencv.org/blog/reading-and-writing-videos-using-opencv/?authuser=1#h-writing-a-video-in-opencv). 