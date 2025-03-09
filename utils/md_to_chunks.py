#!/usr/bin/env python3
import os
from pathlib import Path
import json
import uuid
import re
from datetime import datetime
import sys
from tqdm import tqdm  # Using standard tqdm instead of notebook version
from unstructured.partition.md import partition_md
from unstructured.chunking.title import chunk_by_title
from unstructured.chunking.basic import chunk_elements
import anthropic

def situate_context(client, doc: str, chunk: str):
    try:
        DOCUMENT_CONTEXT_PROMPT = """
        <document>
        {doc_content}
        </document>
        """

        CHUNK_CONTEXT_PROMPT = """
        Here is the chunk we want to situate within the whole document
        <chunk>
        {chunk_content}
        </chunk>

        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
        Answer only with the succinct context and nothing else.
        """

        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            temperature=0.0,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc),
                            "cache_control": {"type": "ephemeral"} # Cache full document context
                        },
                        {
                            "type": "text",
                            "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk),
                        },
                    ]
                },
            ]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Error in situate_context: {e}")
        return "Error generating context"

def classify_content(client, doc: str):
    try:
        CLASSIFIER_PROMPT = """
        You will be analyzing a technical manual for a product to extract specific information. The manual content is provided below:

        <technical_manual>
        {doc_content}
        </technical_manual>

        Your task is to carefully read through the manual and extract the following information:
        1. Company name / brand
        2. Model name of the product being documented
        3. Type of product (e.g., synthesizer, guitar pedal, software plugin)
        4. Keywords that describe the purpose and utility of the product (to aid in BM25 search)

        Follow these steps to complete the task:

        1. Thoroughly read the entire technical manual.

        2. Look for the company name or brand. This is often found on the cover page, in headers, or in copyright notices.

        3. Identify the model name of the product. This is typically prominently displayed near the beginning of the manual or in product descriptions.

        4. Determine the type of product based on the descriptions and features mentioned in the manual.

        5. Extract keywords that describe the product's purpose and utility. Focus on terms that highlight its main features, functions, and applications.

        6. Organize your findings into a JSON object with the following structure:
        {{
            "brand": "",
            "model": "",
            "product_type": "",
            "keywords": []
        }}

        Important notes:
        - If you cannot find a specific piece of information, use "Unknown" as the value.
        - For the "keywords" field, include an array of relevant terms (at least 3, but no more than 10).
        - Ensure that the extracted information is accurate and directly supported by the content in the manual.
        - Leave out legal designations like LLC or TM.

        Present your final output within <json_output> tags, formatted as a valid JSON object.
        """

        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": CLASSIFIER_PROMPT.format(doc_content=doc)
                        }
                    ]
                },
            ]
        )

        # Pattern to match content between <json_output> tags
        pattern = r'<json_output>(.*?)</json_output>'
        
        # Use re.DOTALL to make '.' match newlines as well
        match = re.search(pattern, response.content[0].text, re.DOTALL)

        empty_object = {
            "brand": "",
            "model": "",
            "product_type": "",
            "keywords": []
        }

        if match:
            try:
                # Extract the JSON string and parse it
                json_str = match.group(1).strip()
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                return empty_object
        else:
            return empty_object
    except Exception as e:
        print(f"Error in classify_content: {e}")
        return {
            "brand": "",
            "model": "",
            "product_type": "",
            "keywords": []
        }

def process_document(client, file_path, input_folder, output_folder):
    try:
        md_file = os.path.basename(file_path)
        print(f"Processing: {md_file}")
        filename_base = os.path.splitext(md_file)[0]
        output_folder_chunk_path = os.path.join(output_folder, filename_base)
        
        # Create chunk folder if it doesn't exist
        Path(output_folder_chunk_path).mkdir(exist_ok=True, parents=True)

        with open(file_path, "r", encoding="utf-8") as f:
            document = f.read()
        
        elements = partition_md(filename=file_path)

        # Not chunking tables, so pulling them out
        tables = []
        for element in elements:
            if element.category == "Table":
                tables.append(element)
            
        for table in tqdm(tables, desc="Processing tables"):
            # Create metadata
            chunk_data = {
                "id": str(uuid.uuid4()),
                "source_file": md_file,
                "category": "Table",
                "content": table.text,
                "contextualization": situate_context(client, document, table.text),
                "raw_table": table.metadata.text_as_html,
                "created_at": datetime.now().isoformat()
            }

            # Generate unique filename using UUID
            filename = f"{chunk_data['id']}.json"
            output_path = os.path.join(output_folder_chunk_path, filename)

            # Save chunk as JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f, ensure_ascii=False, indent=2)

        # "For technical manuals, I recommend larger chunk sizes around 300-500 tokens with semantic boundaries."
        # "Use 10% overlap to preserve cross-references."
        chunks = chunk_by_title(
            elements,
            multipage_sections=True,
            combine_text_under_n_chars=1200,
            max_characters=2000,
            overlap=60
        )

        for chunk in tqdm(chunks, desc="Processing chunks"):
            if chunk.category in ["Table", "TableChunk"]:
                continue
            # Create metadata
            chunk_data = {
                "id": str(uuid.uuid4()),
                "source_file": md_file,
                "category": chunk.category,
                "content": chunk.text,
                "contextualization": situate_context(client, document, chunk.text),
                "created_at": datetime.now().isoformat()
            }

            # Generate unique filename using UUID
            filename = f"{chunk_data['id']}.json"
            output_path = os.path.join(output_folder_chunk_path, filename)

            # Save chunk as JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f, ensure_ascii=False, indent=2)

        # Create metadata.json file to classify the pdf as a whole
        print("Generating document metadata...")
        doc_metadata = classify_content(client, document)
        metadata_filename = f"metadata.json"
        metadata_output_path = os.path.join(output_folder_chunk_path, metadata_filename)

        # Save metadata
        with open(metadata_output_path, 'w', encoding='utf-8') as f:
            json.dump(doc_metadata, f, ensure_ascii=False, indent=2)
            
        print(f"Completed processing {md_file}")
        return True
    except Exception as e:
        print(f"Error processing document {file_path}: {e}")
        return False

def main():
    # Configure paths
    input_folder = "output/extractions"
    output_folder = "output/chunks"

    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(exist_ok=True, parents=True)

    # Initialize Claude client
    try:
        client = anthropic.Client(
            max_retries=4
        )
    except Exception as e:
        print(f"Error initializing Anthropic client: {e}")
        print("Make sure your API key is set in the environment variable ANTHROPIC_API_KEY")
        sys.exit(1)

    # Get all Markdown files in the input folder
    md_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith('.md')]

    if not md_files:
        print(f"No markdown files found in {input_folder}")
        sys.exit(0)

    print(f"Found {len(md_files)} markdown files to process")
    
    # Process each Markdown file
    success_count = 0
    for file_path in md_files:
        result = process_document(client, file_path, input_folder, output_folder)
        if result:
            success_count += 1

    print(f"Processing complete. Successfully processed {success_count}/{len(md_files)} files.")

if __name__ == "__main__":
    main()