#!/usr/bin/env python3
"""
Embedding Generator
------------------
This script processes JSON files from specified folders, generates embeddings
using the Voyage AI API, and saves the results to an output directory.
"""

import voyageai
import os
import json
import sys
from pathlib import Path
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function to process JSON files and generate embeddings."""
    try:
        # Initialize the Voyage AI client
        try:
            # Check for API key in environment
            if 'VOYAGE_API_KEY' not in os.environ:
                logger.error("VOYAGE_API_KEY environment variable not set")
                sys.exit(1)
                
            embedding_client = voyageai.Client()
        except voyageai.error.AuthenticationError as e:
            logger.error(f"Authentication failed with Voyage AI: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to initialize Voyage AI client: {e}")
            sys.exit(1)
        
        # Define the root folder path
        root_folder = "output/chunks"
        
        # Check if the root folder exists
        if not os.path.exists(root_folder):
            logger.error(f"Root folder '{root_folder}' does not exist.")
            sys.exit(1)
        
        # Process each subfolder
        processed_count = 0
        error_count = 0
        
        # List all items in the root folder
        try:
            subdirs = os.listdir(root_folder)
        except Exception as e:
            logger.error(f"Failed to list contents of '{root_folder}': {e}")
            sys.exit(1)
        
        for subdir_name in subdirs:
            subdir_path = os.path.join(root_folder, subdir_name)
            
            # Check if this is a subfolder (not a file)
            if os.path.isdir(subdir_path):
                logger.info(f"Processing subfolder: {subdir_name}")
                
                # Create output folder if it doesn't exist
                output_folder = os.path.join("output/embeddings", subdir_name)
                try:
                    Path(output_folder).mkdir(exist_ok=True, parents=True)
                except Exception as e:
                    logger.error(f"Failed to create output folder '{output_folder}': {e}")
                    continue
                
                # Get all JSON files in the current subfolder
                try:
                    json_files = [f for f in os.listdir(subdir_path) if f.lower().endswith('.json')]
                except Exception as e:
                    logger.error(f"Failed to list JSON files in '{subdir_path}': {e}")
                    continue
                
                # Process each JSON file
                for json_file in json_files:
                    file_path = os.path.join(subdir_path, json_file)
                    logger.info(f"Processing: {file_path}")
                    
                    # Handle metadata.json specially
                    if json_file == "metadata.json":
                        try:
                            # Read the metadata file
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            
                            # Create the output path
                            output_path = os.path.join(output_folder, "metadata.json")
                            
                            # Save metadata file as is
                            with open(output_path, 'w', encoding='utf-8') as f:
                                json.dump(data, f, ensure_ascii=False, indent=2)
                            
                            processed_count += 1
                            continue
                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid JSON in metadata file {file_path}: {e}")
                            error_count += 1
                            continue
                        except Exception as e:
                            logger.error(f"Error processing metadata file {file_path}: {e}")
                            error_count += 1
                            continue
                    
                    # Process regular JSON files
                    try:
                        # Read and parse the JSON file
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Prepare document for embedding
                        if 'content' not in data or 'contextualization' not in data:
                            logger.warning(f"File {file_path} is missing required fields 'content' or 'contextualization'")
                            error_count += 1
                            continue
                        
                        documents = [data['content'] + "\n\n" + data['contextualization']]
                        
                        # Generate embeddings
                        try:
                            # Add retry logic for API calls
                            max_retries = 3
                            retry_delay = 2
                            
                            for retry in range(max_retries):
                                try:
                                    embedding_response = embedding_client.embed(
                                        documents,
                                        model="voyage-3",
                                        input_type="document"
                                    )
                                    break
                                except voyageai.error.RateLimitError:
                                    if retry < max_retries - 1:
                                        logger.warning(f"Rate limit hit, retrying in {retry_delay} seconds (attempt {retry+1}/{max_retries})")
                                        time.sleep(retry_delay)
                                        retry_delay *= 2  # Exponential backoff
                                    else:
                                        raise
                            
                            # Validate embedding response
                            if not hasattr(embedding_response, 'embeddings') or not embedding_response.embeddings:
                                logger.error(f"Empty or invalid embedding response for {file_path}")
                                error_count += 1
                                continue
                                
                            embeddings = embedding_response.embeddings[0]
                        except voyageai.error.VoyageError as e:
                            logger.error(f"Voyage API error for {file_path}: {e}")
                            error_count += 1
                            continue
                        except Exception as e:
                            logger.error(f"Failed to generate embeddings for {file_path}: {e}")
                            error_count += 1
                            continue
                        
                        # Add embeddings to data
                        data['embeddings'] = embeddings
                        
                        # Define the output filename and path
                        if 'id' not in data:
                            logger.warning(f"File {file_path} is missing 'id' field, using original filename")
                            filename = json_file
                        else:
                            filename = f"{data['id']}.json"
                        
                        output_path = os.path.join(output_folder, filename)
                        
                        # Save the enriched data
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        
                        processed_count += 1
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in {file_path}: {e}")
                        error_count += 1
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                        error_count += 1
        
        # Log completion summary
        logger.info(f"Embedding generation completed.")
        logger.info(f"Processed {processed_count} files successfully.")
        if error_count > 0:
            logger.warning(f"Encountered errors in {error_count} files.")
            
    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Exiting...")
        sys.exit(130)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()