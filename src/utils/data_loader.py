import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple
from src.model.data_models import MyChunkData, MyMetaData

class DataLoader:
    """Class to load data from various formats."""
    
    @staticmethod
    def load_from_separated_files(chunk_file: str, metadata_file: str) -> List[MyChunkData]:
        """Load data from a separated file"""
        chunks = DataLoader._load_chunks(chunk_file)
        metadata = DataLoader._load_metadata(metadata_file)
        
        chunk_data_list = []
        metadata_dict = {meta['data_id']: meta for meta in metadata}
        
        for content in chunks:
            data_id = content['data_id']
            if data_id in metadata_dict:
                chunk_data = MyChunkData(
                    content=content['content'],
                    metadata=metadata_dict[data_id]
                )
                chunk_data_list.append(chunk_data)
        return chunk_data_list
    
    
    @staticmethod
    def _load_chunks(chunks_file: str) -> Dict[str, str]:
        """Load chunks từ file"""
        with open(chunks_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    
    @staticmethod
    def _load_metadata(metadata_file: str) -> List[Dict[str, Any]]:
        """Load metadata từ file"""
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
        
