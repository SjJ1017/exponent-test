### Document Ingestion:

Documents should be safely fetched and scanned to avoid viruses or other risks. The SME internal files may include: PDF, DOCX, internal wiki exports (HTML/Markdown), plaintext. So in the preprocessing step all the files should be converted into texts.

As for chunking, a normal chunk size of around 500 or 1000 tokens, and 10% overlap would be fine. Chunking should respect the hierarchical structure of documents, as they are highly structured, unlike novels and newspapers. After chunking, each chunk should contain metadata, including:

* Provenance: source, source type, title, creation time…
* security: RBAC, confidential tags
* lexical: language…

For embedding, as the languages may include English, French, German, and Italian, multilingual embedding models should be preferred. Besides, to ensure security, API calls like OpenAI embeddings should be avoided. So deploying open-source multilingual models like `intfloat/multilingual-e5-large` on locally on Swiss Compute is necessary.

### Vector Store:

I will choose `pgvector` because it is better suited to the scale and technical architecture of small and medium-sized enterprises. As a PostgreSQL extension, pgvector can be deployed on a local or private server, allowing vector data and business data to be stored and managed within the same database system. This setup provides stronger data security and compliance control. In contrast, some vector database services, such as Pinecone, primarily operate under a cloud-based model, resulting in less direct physical control over enterprise data. Therefore, from the perspectives of cost efficiency for SME and data security, pgvector is the more appropriate choice.

### LLM Orchestrator

Given the strict data security requirements (nFADP / LPD), `LangChain` is a strong choice as its modular and extensible design makes it easy to add control layers into the pipeline. We can integrate preprocessing (e.g., query normalization), enforce RBAC-based filtering before retrieval, add auditing mechanisms, and apply post-processing steps to reduce hallucinations (such as citation checks or abstention rules). Each step can be structured clearly in the chain, and it also allows future extensions, such as adding internal tools or workflow automation, without redesigning the system.

While LlamaIndex is very convenient for building standard RAG pipelines and is strong in document retrieval, it is more retrieval-focused, with flexibility and engineering control compared to LangChain.

### 100% Switzerland-based Hosting

Three options according to the companies:

* locally on the company's devices. If the company has strict confidentiality requirements, it can use its own computing and storage resources, but this requires the enterprise to purchase GPUs and storage infrastructure, which can be very costly.
* On-prem in Switzerland: Using local compute and GPU servers for inference, like CSCS. Use Swiss local storage. Suitable for most of the companies.
* Swiss cloud provider, including Swiss Ecoscale, AWS Zurich (still need to pay attention and lock it to prohibit cross-border data transfer).  Suitable for companies without very strict security requirements and with relatively limited budgets.

### Security

For RBAC, employees must only retrieve chunks they are allowed to see. The documents should be tagged with different RBAC tags, and a filter should be applied before retrieving.

For data residency, it is necessary to hardcode region selection (Swiss zones only) and block egress to non-CH endpoints at the network layer. Disk and database encryption and encryption in transit are also needed. It is also important to record user informations (id, time…) and detect anomalies, such as repeated access attempts to a large volumne of restricted docs.

### Hallucinations:

Possible techniques:

1. Generate citations for non-trivial responses to improve verifiability, and ask the LLM to abstain from answering when the retrieved documents are not informative. They can also generate confidence after the generation to help verify and improve trustworthiness
2. Verify that statements can be entailed from the retrieved documents (or cited documents) using an external model, such as `google/t5_xxl_true_nli_mixture` or an agentic model.
3. Using reranking or a post-hoc filter to filter out irrelevant documents. Too many irrelevant documents can be noisy, exacerbating the hallucination problem. Reranking can help find more relevant documents.


### Diagram of the pipeline:

![Pipeline](pipeline.svg)