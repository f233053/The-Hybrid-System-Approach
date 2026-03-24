# The-Hybrid-System-Approach
PyTorch-Hybrid-NLP-Engine
PyTorch Hybrid NLP Engine: Multi-Stage Ticket RetrievalA high-performance, multi-stage NLP pipeline built from first principles using base PyTorch and NumPy. This system bridges the gap between traditional statistical retrieval (TF-IDF) and modern neural semantic search (GloVe), optimized for dual-GPU execution.
Why This Project?Most modern NLP projects rely on high-level libraries like Scikit-Learn or HuggingFace. This repository demonstrates a deep-tier understanding of the underlying mathematics by implementing tokenization, vectorization, and similarity scoring from scratch.
Key Features1. The "From Scratch" FoundationCustom Tokenizer: Regex-based processing with lowercasing, punctuation removal, and N-Gram generation (Bigrams/Trigrams).Manual TF-IDF: Implementation of CountVectorizer and TfidfTransformer using raw NumPy/PyTorch logic (No Scikit-Learn allowed).Categorical Encoders: Custom Label and One-Hot encoders with robust error handling for unseen inference data.2. Neural Semantic LayerGloVe Embeddings: 300-dimensional vectors loaded into a torch.nn.Embedding layer.Mean Pooling: Strategy for converting variable-length sequences into fixed-dimension sentence vectors.OOV Strategy: Advanced handling of Out-of-Vocabulary tokens to ensure system stability.3. Hybrid Retrieval LogicThe system uses a weighted scoring mechanism to balance keyword precision with semantic intent:$$FinalScore = \alpha(TF\text{-}IDF) + (1 - \alpha)(GloVe)$$Default $\alpha$: 0.4 (Optimized for semantic-heavy retrieval).4. Performance OptimizationDual-GPU Parallelization: Similarity calculations for query batches are distributed across Dual NVIDIA T4 GPUs, significantly reducing latency for high-volume ticket processing.
Evaluation & ResultsMetricPerformancePrecision @ 5[Insert your % here, e.g., 88%]Inference Time (Batch 100)[Insert your time, e.g., 140ms]AcceleratorDual T4 GPUsQualitative Insight: The GloVe-based search successfully retrieves "Billing" issues when queried with "Payment problems," even when there is 0% keyword overlap.
Interactive AppThe project includes a Gradio/Streamlit deployment featuring:Dynamic Alpha Slider: Adjust the balance between Keyword and Semantic search in real-time.Side-by-Side Comparison: View how TF-IDF and GloVe interpret the same query differently.Ticket Classification: Automatic prediction of Ticket Type and Priority.
Repository StructurePlaintext├── data/               # Customer Support Ticket Dataset
├── notebooks/          # Documented PyTorch Implementation
├── src/
│   ├── encoders.py     # Custom OHE and Label Encoders
│   ├── tfidf.py        # From-scratch TF-IDF & Tokenizer
│   └── embeddings.py   # GloVe & Mean Pooling logic
├── app.py              # Gradio/Streamlit Application
└── requirements.txt    # Project Dependencies
🚀 Getting StartedClone the repo:Bashgit clone https://github.com/your-username/PyTorch-Hybrid-NLP-Engine.git
Install dependencies:Bashpip install -r requirements.txt
Run the interactive app:Bashpython app.py
📧 Contact & LinksLinkedIn: Ahmad Mujtaba| www.linkedin.com/in/ahmad-mujtaba-79a17b280
Medium: https://medium.com/@f233053/beyond-the-library-building-a-hybrid-nlp-search-engine-from-scratch-in-pytorch-0bbc90625aaa
website: https://9639a8dccafe9f2f40.gradio.live/
