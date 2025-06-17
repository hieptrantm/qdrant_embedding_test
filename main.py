from src.core.qdrant_deployer import QdrantRAGDeployer
from src.utils.data_loader import DataLoader
from src.model.data_models import MyChunkData, MyMetaData

import json

def main():
    """Hàm main để chạy toàn bộ pipeline"""
    
    # Cấu hình
    QDRANT_URL = "http://localhost:32768/"
    QDRANT_API_KEY = "171749a1-8c0f-4f34-9d60-24ff614342ca|0fgtRc_aAwnzR9Lh2Cki71jR9ETk2_jpclL-NCQ9nMyM7jIpOwOr0A"
    COLLECTION_NAME = "new_rag_documents"  # Tên collection mới
    
    # Đường dẫn file dữ liệu
    CHUNKS_FILE = "data/chunks.json"  
    METADATA_FILE = "data/metadata.json"
    
    print("=== RAG Dataset Deployment to Qdrant ===")

    print("1. Khởi tạo Qdrant deployer...")
    deployer = QdrantRAGDeployer(
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY
    )

    print("2. Load dữ liệu...")
    chunk_data_list = DataLoader.load_from_separated_files(CHUNKS_FILE, METADATA_FILE)
    print(f"Đã load {len(chunk_data_list)} chunks")

    print("3. Tạo collection...")
    deployer.create_collection(COLLECTION_NAME)
 
    print("4. Upload dữ liệu...")
    deployer.upload_chunks(COLLECTION_NAME, chunk_data_list)

    print("5. Kiểm tra collection...")
    deployer.get_collection_info(COLLECTION_NAME)

    print("\n6. Test tìm kiếm...")

    with open ("trash/query.json", "r", encoding="utf-8") as f:
        queries = json.load(f)

    num_query = len(queries)
    
    for index in range(num_query):
        data = queries[index]['query']
        folder_name = queries[index]['folder']
        topic_name = queries[index]['topic']
        print(f"\nTìm kiếm: '{data}'")
        results = deployer.search_chunks(COLLECTION_NAME, data, limit=3)
        right_answers = 0
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result['score']:.4f}")
            print(f"     Data ID: {result['metadata']['data_id']}")
            print(f"     Topic: {result['metadata']['topic']}")
            print(f"     Folder: {result['metadata']['folder']}")
            print(f"     Content: {result['content'][:100]}...")
            if(topic_name == result['metadata']['topic'] and folder_name == result['metadata']['folder']):
                right_answers += 1
                print("Kết quả đúng!")
    
    print(f"Tổng số kết quả đúng: {right_answers}/{len(results)}")
    print("\n=== Hoàn thành! ===")

if __name__ == "__main__":
    main()