# Projeto Libras - Ambiente e Documentação

## Ambiente Computacional
**Versão do Python:** 3.8.10

## Bibliotecas e Dependências
As principais bibliotecas utilizadas neste projeto são:

1.  **mindspore**
    *   Framework de Inteligência Artificial utilizado para criar, treinar e executar a rede neural.
    *   Arquivos: `src/modelo_libras.py`, `src/treinar_mindspore_simples.py`, `src/reconhecer_mindspore.py`.
2.  **mediapipe**
    *   Utilizado para detecção de mãos e extração dos pontos (landmarks) das articulações.
    *   Arquivos: `detectar_maos.py`, `src/coletar_dados.py`, `src/reconhecer_mindspore.py`.
3.  **opencv-python (cv2)**
    *   Biblioteca de Visão Computacional usada para capturar vídeo da webcam, processar imagens e desenhar na tela.
    *   Arquivos: `detectar_maos.py`, `src/coletar_dados.py`.
4.  **numpy**
    *   Fundamental para manipulação de arrays e operações matemáticas nos dados dos landmarks.
5.  **joblib**
    *   Utilizado para salvar e carregar objetos (como o `LabelEncoder`) de forma eficiente.

Outros módulos padrão: `os`, `sys`, `pickle`, `collections`.

## Estrutura de Arquivos e Documentos
Esta é a lista dos arquivos de código fonte (scripts) presentes no projeto:

### Raiz do Projeto
*   `detectar_maos.py`: Demonstração simples da detecção de mãos.
*   `test_tudo.py`: Teste de importação de todas as bibliotecas.
*   `teste_rapido.py`: Teste rápido de integração.
*   `verificar.py`: Verificação de dependências instaladas.
*   `testar.py`: Script de teste.

### Diretório `src/` (Código Fonte Principal)
*   `src/coletar_dados.py`: Script para capturar exemplos de sinais e salvar em disco para treinamento.
*   `src/modelo_libras.py`: Define a classe da Rede Neural em MindSpore.
*   `src/treinar_mindspore_simples.py`: Carrega os dados coletados e treina o modelo.
*   `src/reconhecer_mindspore.py`: Executa o reconhecimento de sinais em tempo real usando a webcam e o modelo treinado.
*   `src/reconhecer.py`: Versão alternativa/antiga do reconhecedor.
*   `src/treinar_modelo.py`: Versão alternativa/antiga do treinamento.
