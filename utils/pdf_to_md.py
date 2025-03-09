#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
import fitz  # PyMuPDF
import anthropic
import time
from tqdm import tqdm  # Standard tqdm instead of notebook version
import base64
import re
import sys

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert PDF files to markdown using Claude API')
    parser.add_argument('input_folder', help='Directory containing PDF files to process', default="documents", nargs='?')
    args = parser.parse_args()
    
    # Configuration
    input_folder = args.input_folder
    output_folder = "output/extractions"
    prompt = """Please follow these instructions carefully:

1. Analyze the PDF content thoroughly.

2. Convert the content to markdown format, following these rules:
   - Ignore page headers at the top of each page and page footers at the bottom of each page. These often contain page numbers, document names, or section titles.
   - Preserve semantic markup such as headings, bold text, italics, and bullet points.
   - If you encounter pictures or diagrams, describe their purpose in markdown instead of including the actual images.
   - For multi-column layouts, treat the columns as one continuous page, maintaining the logical flow of the content.
   - Do not exclude any sections, summarize them, or truncate for length.

3. Before providing the final markdown output, wrap your analysis in a <pdf_analysis> tag to show your thought process and ensure you've addressed all requirements. In your analysis:
   - List the main sections or chapters of the PDF content.
   - Identify and quote examples of headers and footers you'll be ignoring.
   - List and describe any images or diagrams you've found.
   - Note any special formatting or semantic markup you've encountered.
   - Explain how you'll handle multi-column layouts, if present.
   - Double check that you didn't truncate any content.
   - Outline your plan for converting the content to markdown.

4. After your analysis, provide the converted markdown content in <markdown_output></markdown_output> tags without any additional commentary. Don't forget the closing tag.

Example output structure:

<pdf_analysis>
[Your detailed analysis of the PDF content, including:
- List of main sections or chapters
- Examples of headers and footers
- Description of images or diagrams
- Notes on special formatting or semantic markup
- Approach for handling multi-column layouts
- Check that all required content is present and not truncated.
- Conversion plan]
</pdf_analysis>

<markdown_output>
[Your converted markdown content here]
</markdown_output>

Please proceed with your analysis and conversion of the PDF content."""

    # Check if input directory exists
    if not os.path.isdir(input_folder):
        print(f"Error: Input directory '{input_folder}' does not exist.")
        sys.exit(1)

    # Initialize Claude client
    try:
        client = anthropic.Client()
    except Exception as e:
        print(f"Error initializing Claude client: {e}")
        print("Make sure you have set your ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(exist_ok=True, parents=True)

    # Get all PDF files in the input folder
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in '{input_folder}'.")
        sys.exit(0)
    
    print(f"Found {len(pdf_files)} PDF files to process.")

    # Process each PDF file
    for pdf_file in tqdm(pdf_files, desc="Processing PDF files"):
        file_path = os.path.join(input_folder, pdf_file)
        print(f"Processing: {pdf_file}")
        
        # Prepare variables
        filename_base = os.path.splitext(pdf_file)[0]
        doc = None
        
        # Prepare the results container for this PDF
        pdf_results = {
            'filename': pdf_file,
            'path': file_path,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'pages': []
        }

        markdown_output = ""
        
        try:
            # Open the PDF
            doc = fitz.open(file_path)
            
            # Process each page
            for page_num in tqdm(range(len(doc)), desc=f"Pages in {pdf_file}"):
                # Create empty pdf, insert page, and then get binary info
                page_pdf = fitz.open()
                page_pdf.insert_pdf(doc, from_page=page_num, to_page=page_num)
                page_bytes = page_pdf.tobytes()
                base64_string = base64.b64encode(page_bytes).decode("utf-8")
                
                # Call Claude API with retries
                max_retries = 3
                retry_count = 0
                success = False

                while retry_count < max_retries and not success:
                    try:
                        if retry_count > 0:
                            print(f"  Retry {retry_count} for page {page_num+1}")
                            # Longer delay for retries
                            time.sleep(2)
                        
                        response = client.messages.create(
                            model="claude-3-7-sonnet-latest",
                            max_tokens=8192,
                            system="You are an advanced AI assistant specializing in PDF content analysis and conversion. Your task is to convert the provided PDF content into markdown format while adhering to specific guidelines.",
                            messages=[
                                {
                                    "role": "user", 
                                    "content": [
                                        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": base64_string}},
                                        {"type": "text", "text": prompt}
                                    ]
                                 }
                            ]
                        )
                        
                        # Store the response in our results
                        if not response.content or not response.content[0].text:
                            raise Exception("Empty response received from Claude API")
                        
                        pattern = r'<markdown_output>(.*?)</markdown_output>'
                        match = re.search(pattern, response.content[0].text, re.DOTALL)

                        if match:
                            pdf_results['pages'].append({
                                'page_number': page_num + 1,
                                'status': 'success',
                                'retry_count': retry_count,
                                'response': response.content[0].text
                            })
                            markdown_output += match.group(1) + "\n"
                            success = True
                        else:
                            if retry_count == max_retries - 1:
                                print(f"  Warning: Could not extract markdown output after {max_retries} attempts on page {page_num+1}")
                                pdf_results['pages'].append({
                                    'page_number': page_num + 1,
                                    'status': 'warning',
                                    'retry_count': retry_count,
                                    'warning': "Could not extract markdown output after maximum retries",
                                    'response': response.content[0].text
                                })
                            retry_count += 1
                        
                        # Respect rate limits - add a small delay
                        time.sleep(0.5)
                    
                    except Exception as api_error:
                        if retry_count == max_retries - 1:
                            print(f"  API error on page {page_num+1} after {max_retries} attempts: {api_error}")
                            # Record the error but continue with next page
                            pdf_results['pages'].append({
                                'page_number': page_num + 1,
                                'status': 'error',
                                'retry_count': retry_count,
                                'error_message': str(api_error)
                            })
                        else:
                            print(f"  API error on page {page_num+1} (attempt {retry_count+1}): {api_error}")
                            retry_count += 1
            
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            pdf_results['error'] = str(e)
        
        finally:
            # Close the document if it was successfully opened
            if doc is not None:
                try:
                    doc.close()
                except Exception as close_error:
                    print(f"Warning: Could not close document properly: {close_error}")
            
            # Save results for this PDF, even if partial due to errors
            output_file = os.path.join(output_folder, f"{filename_base}.md")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_output)
            debug_file = os.path.join(output_folder, f"{filename_base}_results.json")
            with open(debug_file, 'w', encoding='utf-8') as f:
                json.dump(pdf_results, f, indent=2)
                
            print(f"Saved results for {pdf_file} to {output_file}")

    print(f"Processing complete. Results saved to {output_folder}")

if __name__ == "__main__":
    main()