import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from src.model.data_models import MyChunkData
from src.core.filter_predict import predict_topic

class QdrantRAGDeployer:
    def __init__(self, 
                 qdrant_url: str = "http://localhost:6333",
                 qdrant_api_key: Optional[str] = None,
                 embedding_model: str = "dangvantuan/vietnamese-document-embedding"):
        """
        Khởi tạo Qdrant RAG Deployer
        
        Args:
            qdrant_url: URL của Qdrant server
            qdrant_api_key: API key (nếu có)
            embedding_model: Model để tạo embeddings
        """
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.embedding_model = SentenceTransformer(embedding_model, trust_remote_code=True)
        self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
        
    def create_collection(self, collection_name: str):
        """
        Tạo collection trong Qdrant
        
        Args:
            collection_name: Tên collection
        """
        try:
            # Kiểm tra collection đã tồn tại chưa
            collections = self.client.get_collections()
            existing_collections = [c.name for c in collections.collections]
            
            if collection_name in existing_collections:
                print(f"Collection '{collection_name}' đã tồn tại")
                return True
            
            # Tạo collection mới
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"Đã tạo collection '{collection_name}' thành công với vector size {self.vector_size}")
            return True
            
        except Exception as e:
            print(f"Lỗi khi tạo collection: {e}")
            return False
    
    def prepare_point(self, chunk_data: MyChunkData) -> PointStruct:
        """
        Chuẩn bị point cho Qdrant từ ChunkData
        
        Args:
            chunk_data: ChunkData object
            
        Returns:
            PointStruct: Point để upload lên Qdrant
        """
        # Tạo embedding từ content
        embedding = self.embedding_model.encode(chunk_data.content).tolist()
        
        # Tạo payload từ metadata và content
        payload = chunk_data.metadata
        payload['content'] = chunk_data.content  # Thêm content vào payload
        
        return PointStruct(
            id=chunk_data.metadata['data_id'],
            vector=embedding,
            payload=payload
        ) 
    
    def upload_chunks(self, collection_name: str, chunk_data_list: List[MyChunkData], batch_size: int = 100):
        """
        Upload chunks lên Qdrant
        
        Args:
            collection_name: Tên collection
            chunk_data_list: Danh sách ChunkData
            batch_size: Kích thước batch
        """
        points = []
        total_chunks = len(chunk_data_list)
        
        print(f"Bắt đầu upload {total_chunks} chunks...")
        
        for i, chunk_data in enumerate(chunk_data_list):
            try:
                point = self.prepare_point(chunk_data)
                points.append(point)
                
                # Upload khi đạt batch_size hoặc là chunk cuối cùng
                if len(points) >= batch_size or i == total_chunks - 1:
                    self.client.upsert(
                        collection_name=collection_name,
                        points=points
                    )
                    print(f"Đã upload {i + 1}/{total_chunks} chunks")
                    points = []
                    
            except Exception as e:
                print(f"Lỗi khi xử lý chunk {chunk_data.metadata['data_id']}: {e}")
                continue
        
        print("Hoàn thành upload")
    
    def search_chunks(self, 
                     collection_name: str, 
                     query: str, 
                     limit: int = 5,
                     folder_filter: Optional[str] = None,
                     topic_filter: Optional[str] = None) -> List[Dict]:
        """
        Tìm kiếm chunks với filter
        
        Args:
            collection_name: Tên collection
            query: Câu truy vấn
            limit: Số lượng kết quả
            folder_filter: Filter theo folder
            topic_filter: Filter theo topic
            
        Returns:
            List[Dict]: Kết quả tìm kiếm
        """
        try:
            # Tạo embedding cho query
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # # Dự đoán topic
            # if topic_filter is None:
            #     predicted_topic, _ = predict_topic(query)
            #     topic_filter = predicted_topic
            #     print(f"Dự đoán topic: {topic_filter}")

            # Tạo filter conditions
            filter_conditions = []
            if folder_filter:
                filter_conditions.append(
                    FieldCondition(key="folder", match=MatchValue(value=folder_filter))
                )
            if topic_filter:
                filter_conditions.append(
                    FieldCondition(key="topic", match=MatchValue(value=topic_filter))
                )
            
            # Tạo filter object
            search_filter = None
            if filter_conditions:
                search_filter = Filter(must=filter_conditions)
            
            # Tìm kiếm
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                query_filter=search_filter,
                limit=limit,
                with_payload=True
            )
            
            results = []
            for result in search_result:
                results.append({
                    'score': result.score,
                    'content': result.payload.get('content', ''),
                    'metadata': {k: v for k, v in result.payload.items() if k != 'content'}
                })
            
            return results
            
        except Exception as e:
            print(f"Lỗi khi tìm kiếm: {e}")
            return []
    
    def get_collection_info(self, collection_name: str):
        """Lấy thông tin collection"""
        try:
            info = self.client.get_collection(collection_name)
            print(f"Thông tin collection '{collection_name}':")
            print(f"- Số lượng points: {info.points_count}")
            print(f"- Kích thước vector: {info.config.params.vectors.size}")
            print(f"- Distance metric: {info.config.params.vectors.distance}")
            
        except Exception as e:
            print(f"Lỗi khi lấy thông tin collection: {e}")
