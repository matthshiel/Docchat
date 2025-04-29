# Docchat With an LLM about documents

![test cases](https://github.com/matthshiel/docchat/workflows/tests/badge.svg)

This Docchat allows the user to send a document to an LLM, and engange in an
conversation with the LLM. The Docchat enables the user to ask questions or ask 
LLM to summarize key details in the document




Example of it working well:
```
docchat> file:docs/research_paper.pdf
Document loaded successfully.
Relevant chunks identified and added to the conversation.
```
```
docchat> tell me the general idea of the research paper
result= The general idea of the research paper appears to be about improving the performance of models for processing large documents. The paper discusses the limitations of existing models, which are often based on the BERT architecture and require O(n^2) memory and computational resources, making it difficult to process documents longer than a certain length.

The paper proposes the DOCSPLIT pretraining method, which can be used to improve the performance of models on document-level tasks, and introduces a contrastive objective for pretraining language models on large documents. The authors suggest that this approach could be used to enable the processing of longer documents and improve the performance of models on document-level tasks.
```



An exmple of the model falling short 
```
docchat> file:https://indianembassy-moscow.gov.in/pdf/russia-fact-sheet-oct-2022.pdf
Detected PDF file. Content-Type: application/pdf
Document loaded successfully.
Relevant chunks identified and added to the conversation.
```
```
docchat> what is russia's population?
result= According to the text, Russia's population is 145.6 million (as of July 2022 census), with an estimated decline to 143 million by 2036.
```
Some of the information got lost when we split the chunks for processing and the pick out bits of information

<p align="center"><img src="/demo.gif?raw=true"/></p>
