# IFC Audit Agent

Aplicacao acadêmica em Python + Streamlit para analisar arquivos IFC com `ifcopenshell` e gerar uma auditoria tecnica com Agno + LLM.

O foco do projeto e mostrar, de forma simples e didatica, a combinacao de:

- upload de IFC no Streamlit;
- extração estruturada de dados com `ifcopenshell`;
- auditoria automatizada com um agente Agno;
- chat simples para perguntas sobre o modelo analisado.

## Funcionalidades

- Upload de arquivo `.ifc`.
- Leitura do modelo IFC com tratamento de erro.
- Extração de:
  - nome do projeto;
  - total de entidades;
  - contagem de elementos por classe IFC;
  - lista de `IfcBuildingStorey`;
  - elementos sem `Name`;
  - elementos sem material associado;
  - elementos sem propriedades IFC;
  - elementos sem quantidades;
  - resumo dos principais tipos `IfcWall`, `IfcSlab`, `IfcDoor`, `IfcWindow`, `IfcBeam` e `IfcColumn`.
- Exibição dos dados em tabelas no Streamlit.
- Relatório técnico gerado por um agente Agno.
- Chat simples para perguntas sobre o IFC analisado.

## Estrutura

```text
app.py
src/ifc_analyzer.py
src/agent.py
src/__init__.py
requirements.txt
README.md
```

## Variáveis de ambiente

Escolha um provider e informe a chave correspondente:

```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash
GEMINI_API_KEY=sua-chave-aqui

# ou
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sua-chave-aqui
```

Se quiser, você também pode usar `GOOGLE_API_KEY` para Gemini.
O app não exige `OPENAI_API_KEY` quando `LLM_PROVIDER=gemini`.

## Como rodar

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

No Linux/macOS, use `source .venv/bin/activate` no lugar do comando do Windows.

## Como a aplicação funciona

1. O usuário faz upload do arquivo IFC.
2. `ifcopenshell` abre o modelo e extrai os dados estruturais.
3. A interface mostra tabelas com os resultados da auditoria.
4. O resumo estruturado vai para o agente Agno.
5. O agente gera:
   - resumo tecnico do modelo;
   - pontos criticos;
   - nivel de completude informacional;
   - recomendacoes de melhoria;
   - conclusao de auditoria.
6. O chat permite perguntas adicionais sobre o mesmo IFC.

## Observacoes

- A aplicação nao faz edição do IFC.
- Nao ha analise geométrica complexa.
- O objetivo e demonstrar o uso combinado de `ifcopenshell` + `Agno` + IA.
- Se o arquivo nao puder ser lido, a interface mostra uma mensagem de erro amigavel.

## Dica para apresentação

Se quiser uma demo simples para vídeo, o fluxo mais claro e:

1. subir o IFC;
2. mostrar as tabelas;
3. clicar em "Gerar análise técnica com Agno";
4. abrir o chat e fazer 1 ou 2 perguntas.
