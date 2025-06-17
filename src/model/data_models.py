from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime

@dataclass
class MyMetaData:
    """Metadata for a data model."""
    data_id: int
    folder: str
    topic: str
    
    def __post_init__(self):
        if self.create_at is None:
            self.create_at = datetime.now().isoformat()
        if self.tags is None:
            self.tags = []
            
    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_id": self.data_id,
            "folder": self.folder,
            "topic": self.topic
        }
        
@dataclass
class MyChunkData:
    """Data model for a chunk of data."""
    content: str
    metadata: MyMetaData
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "metadata": self.metadata.to_dict()
        }