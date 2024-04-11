Used Weaviate vector database and LangChain to augment school textbooks context and generate answers to chapter end questions.
```
1. Process Flow
```
We visit each page of a pdf, adding it to a list dcouments [].
We then chunk the text and store it in the Weavite vector database. We then retrieve the embeddings and feed them through Langchain to generate responses (answers).
```
2. RAG
```
We retrieve the vectores from database, augment them to chain and generate answers.
```
2.1 Text Splitting
```  
We use LangChain text splitter with chunk size of 500 and overlap of 50.
```
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        loader = PyPDFLoader(os.path.join(dirname, filename))
        documents.extend(loader.load())
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
```
```
2.2 Vectoring and Storage
```
Generating vector embeddings and storing them to Weaviate vector database. We use OpenAI embeddings; .from_documents() populates the database with chunks.
```
client = weaviate.Client(
  embedded_options = EmbeddedOptions()
)

vectorstore = Weaviate.from_documents(
    client = client,    
    documents = docs,
    embedding = OpenAIEmbeddings(),
    by_text = False
)
```
```
2.3 Retrieval
```
We then retrieve the vectors as follows. After you fill up the vector database, you can set it up as the retriever. This retriever gets more information by comparing how similar the user's question is to the stored pieces of information. It uses this comparison to fetch relevant additional context.
```
retriever = vectorstore.as_retriever()
```
```
2.4 Augment
```
We create LangChain ChatPrompt template with question and context, briefing the model on how to generate the responses. 
```
template = """You are an assistant for question-answering tasks in context of school English 
stories textbooks. You are expected to generate answer based on a user query.
Question: {question} 
Context: {context} 
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

print(prompt)
```
```
2.5 Generation
```
Finally, we invoke RAG LangChain to generate responses. We use gpt-4-0125-preview as it offers a large token size. We provide context with retrieved embeddings and invoke rag LangChain for each query and store answers.
```
llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0)

rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser() 
)
def question_answer(question):
    answer = rag_chain.invoke(question)
    return answer
question = """Which one of the following sums up the story best?
(i) A bird in hand is worth two in the bush.
(ii) One is known by the company one keeps.
(iii) A friend in need is a friend indeed.
"""
print(question_answer(question))
```
