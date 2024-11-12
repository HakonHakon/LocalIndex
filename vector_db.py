# vector_db.py

import os
import json
from datetime import datetime
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv

class LocalVectorDBChat:
    def __init__(self, collection_name: str = "documents", use_persistent: bool = True):
        # Set up storage paths
        self.base_path = "./vector_db_data"
        self.docs_path = os.path.join(self.base_path, "documents")
        self.index_path = os.path.join(self.base_path, "index.json")
        
        # Create necessary directories
        os.makedirs(self.docs_path, exist_ok=True)
        
        # Initialize document index
        self.doc_index = self._load_index()
        
        # Set up embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Set up chat model
        model_name = "facebook/opt-350m"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.chat_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        # Set up vector database
        if use_persistent:
            storage_path = os.path.join(self.base_path, "qdrant_data")
            os.makedirs(storage_path, exist_ok=True)
            self.qdrant_client = QdrantClient(path=storage_path)
        else:
            self.qdrant_client = QdrantClient(":memory:")
        
        self.collection_name = collection_name
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self._init_collection()

    def _load_index(self) -> Dict:
        """Load document index from file"""
        if os.path.exists(self.index_path):
            with open(self.index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"documents": [], "categories": {}}

    def _save_index(self):
        """Save document index to file"""
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(self.doc_index, f, indent=2)

    def _save_document(self, text: str, category: str = "general") -> str:
        """Save document to file and return document ID"""
        doc_id = str(len(self.doc_index["documents"]))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{doc_id}_{timestamp}.txt"
        filepath = os.path.join(self.docs_path, filename)
        
        # Save document content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Update index
        doc_info = {
            "id": doc_id,
            "filename": filename,
            "category": category,
            "timestamp": timestamp,
            "text_preview": text[:100] + "..." if len(text) > 100 else text
        }
        self.doc_index["documents"].append(doc_info)
        
        # Update categories
        if category not in self.doc_index["categories"]:
            self.doc_index["categories"][category] = []
        self.doc_index["categories"][category].append(doc_id)
        
        self._save_index()
        return doc_id

    def list_documents(self, category: str = None) -> List[Dict]:
        """List all stored documents, optionally filtered by category"""
        if category:
            doc_ids = self.doc_index["categories"].get(category, [])
            return [doc for doc in self.doc_index["documents"] if doc["id"] in doc_ids]
        return self.doc_index["documents"]

    def list_categories(self) -> List[str]:
        """List all available categories"""
        return list(self.doc_index["categories"].keys())

    def _init_collection(self):
        """Initialize Qdrant collection"""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            if collection_info.config.params.vectors.size != self.embedding_dim:
                self.qdrant_client.delete_collection(self.collection_name)
                raise ValueError("Collection exists with wrong dimensions")
        except Exception:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
            )

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding using Sentence Transformers"""
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()

    def _generate_response(self, context: str, query: str) -> str:
        """Generate response using local language model"""
        prompt = f"""Based on the following context, please provide a natural response to the question.
        
Context: {context}

Question: {query}

Response:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = inputs.to(self.chat_model.device)
        
        outputs = self.chat_model.generate(
            **inputs,
            max_length=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            answer = response.split("Response:")[-1].strip()
        except:
            answer = response
        
        return answer.replace(prompt, "").strip()

    def train(self, documents: List[Dict[str, str]], category: str = "general"):
        """Store and index documents"""
        points = []
        
        for doc in documents:
            # Save document and get ID
            doc_id = self._save_document(doc['text'], category)
            
            # Create embedding
            embedding = self._get_embedding(doc['text'])
            
            # Create point for Qdrant
            point = PointStruct(
                id=int(doc_id),
                vector=embedding,
                payload={"text": doc['text'], "doc_id": doc_id}
            )
            points.append(point)
        
        # Upload points to Qdrant
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Stored {len(documents)} documents in category '{category}'")

    def chat(self, query: str, k: int = 3) -> str:
        """Chat with the stored documents"""
        query_embedding = self._get_embedding(query)
        
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k
        )
        
        if not search_result:
            return "I don't have enough information to answer that question."
        
        context = "\n\n".join([hit.payload['text'] for hit in search_result])
        response = self._generate_response(context, query)
        
        return response