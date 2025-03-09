#!/usr/bin/env python3
"""
Weaviate Document Loader

A command line utility to load embedded document chunks into a Weaviate collection.
"""

import os
import sys
import json
import logging
import traceback
from typing import Dict, Any, List
import weaviate
import weaviate.classes as wvc
from weaviate.exceptions import WeaviateQueryError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
COLLECTION_NAME = "Manuals"
ROOT_FOLDER = "output/embeddings"  # Default path

def connect_to_weaviate() -> weaviate.WeaviateClient:
    try:
        client = weaviate.connect_to_local()
        if not client.is_ready():
            logger.error("Weaviate connection is not ready. Please ensure Weaviate is running.")
            sys.exit(1)
        logger.info("Connected to Weaviate successfully.")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Weaviate: {str(e)}")
        sys.exit(1)


def check_and_setup_collection(client: weaviate.WeaviateClient):
    try:
        if client.collections.exists(COLLECTION_NAME):
            logger.info(f"Collection '{COLLECTION_NAME}' already exists.")
            return client.collections.get(COLLECTION_NAME)
        
        # Create a new collection
        logger.info(f"Creating collection '{COLLECTION_NAME}'...")
        collection = client.collections.create(
            COLLECTION_NAME,
            vectorizer_config=wvc.config.Configure.Vectorizer.none()
        )
        logger.info(f"Collection '{COLLECTION_NAME}' created successfully.")
        return collection
    except Exception as e:
        logger.error(f"Error setting up collection: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


def load_metadata(metadata_file: str) -> Dict[str, Any]:
    try:
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Metadata file not found: {metadata_file}")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in metadata file: {metadata_file}")
        return {}
    except Exception as e:
        logger.error(f"Error reading metadata file {metadata_file}: {str(e)}")
        return {}


def load_embeddings(collection, root_folder: str) -> None:
    """Load embeddings from JSON files into Weaviate collection."""
    total_files = 0
    successful_imports = 0
    
    try:
        # Check if root folder exists
        if not os.path.exists(root_folder):
            logger.error(f"Root folder not found: {root_folder}")
            sys.exit(1)
            
        # Process each subfolder in the root folder
        for subdir_name in os.listdir(root_folder):
            subdir_path = os.path.join(root_folder, subdir_name)
            
            # Skip if not a directory
            if not os.path.isdir(subdir_path):
                continue
                
            # Load metadata for this subdirectory
            metadata_file = os.path.join(subdir_path, 'metadata.json')
            metadata = load_metadata(metadata_file)
            
            if not metadata:
                logger.warning(f"Skipping subdirectory {subdir_name} due to missing or invalid metadata.")
                continue
                
            # Find all JSON files (except metadata.json)
            json_files = [f for f in os.listdir(subdir_path) 
                         if f.lower().endswith('.json') and f != "metadata.json"]
                
            logger.info(f"Processing {len(json_files)} files in {subdir_name}...")
            
            # Process each JSON file
            for json_file in json_files:
                total_files += 1
                file_path = os.path.join(subdir_path, json_file)
                
                try:
                    # Load the JSON data
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        
                    # Validate required fields
                    if not all(key in data for key in ["id", "embeddings", "content"]):
                        logger.warning(f"Skipping file {json_file}: Missing required fields.")
                        continue
                        
                    # Prepare properties with lowercase values for consistent querying
                    properties = {
                        "content": data["content"],
                        "doc_type": "chunk",
                        "brand": metadata.get("brand", "").lower(),
                        "model": metadata.get("model", "").lower(),
                        "product_type": metadata.get("product_type", "").lower(),
                    }
                    
                    # Add keywords if available
                    if "keywords" in metadata and isinstance(metadata["keywords"], list):
                        properties["keywords"] = ",".join(metadata["keywords"]).lower()
                    
                    # Insert data into Weaviate
                    uuid = collection.data.insert(
                        uuid=data["id"],
                        vector=data["embeddings"],
                        properties=properties
                    )
                    
                    successful_imports += 1
                    if successful_imports % 10 == 0:
                        logger.info(f"Imported {successful_imports} documents so far...")
                        
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON format in file: {file_path}")
                except KeyError as ke:
                    logger.error(f"Missing key in file {file_path}: {str(ke)}")
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error during embedding import: {str(e)}")
        traceback.print_exc()
    
    logger.info(f"Import summary: Successfully imported {successful_imports} out of {total_files} files.")


def main():
    """Main execution function."""
    try:
        logger.info("Starting Weaviate document loader utility")
        
        # Connect to Weaviate
        client = connect_to_weaviate()
        
        # Check/setup collection
        collection = check_and_setup_collection(client)
        
        # Load embeddings
        load_embeddings(collection, ROOT_FOLDER)
        
        # Close connection
        client.close()
        logger.info("Document loading completed successfully.")
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()