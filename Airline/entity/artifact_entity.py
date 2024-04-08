from dataclasses import dataclass

# data ingestion artifact start here
@dataclass
class DataIngestionArtifact:
    train_file_path:str
    test_file_path:str

# data ingestion artifact end here