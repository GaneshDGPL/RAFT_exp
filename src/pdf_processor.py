import re
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence.models import DocumentContentFormat
from copy import deepcopy
import logging
import json

# Setup logging
logger = logging.getLogger(__name__)

def ensure_dir_exists(file_path):
    """Create directory if it doesn't exist"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def get_relative_path(file_path, base_dir):
    """Get the relative path of a file from the base directory"""
    return os.path.relpath(file_path, base_dir)

class PDFTextExtractor:
    """
    Simplified PDF text extraction class focusing only on text content extraction.
    """
    
    def __init__(self):
        pass

    async def extract_content_from_pdf(self, document: bytes, endpoint: str, key: str) -> Any:
        """
        Helper function to call Azure Form Recognizer and extract the document content.
        """
        if not endpoint or not key:
            raise ValueError("Azure Form Recognizer 'endpoint' and 'key' must be provided.")

        client = DocumentIntelligenceClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )
        try:
            poller = client.begin_analyze_document(
                "prebuilt-layout",
                body=document,
                content_type="application/pdf",
                output_content_format=DocumentContentFormat.MARKDOWN,
            )
            return poller.result()
        except Exception as e:
            logger.error(f"Error occurred in extracting document: {str(e)}")
            return None

    async def convert_polygon_to_points(self, polygon: list) -> list:
        """Convert a flat polygon list into (x, y) point pairs."""
        return [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]

    async def detect_overlap(
        self,
        paragraph_polygon: List[float],
        table_polygons_by_page: dict[int, List[List[float]]],
        page_number: int,
    ) -> Optional[int]:
        """
        Detects overlap between a paragraph polygon and tables on the same page.
        """
        para_points = await self.convert_polygon_to_points(paragraph_polygon)
        para_x_min = min(x for x, _ in para_points)
        para_x_max = max(x for x, _ in para_points)
        para_y_min = min(y for _, y in para_points)
        para_y_max = max(y for _, y in para_points)

        if page_number not in table_polygons_by_page:
            return None

        for idx, table_polygon in enumerate(table_polygons_by_page[page_number]):
            table_points = await self.convert_polygon_to_points(table_polygon)
            table_x_min = min(x for x, _ in table_points)
            table_x_max = max(x for x, _ in table_points)
            table_y_min = min(y for _, y in table_points)
            table_y_max = max(y for _, y in table_points)

            if not (
                para_x_max < table_x_min
                or para_x_min > table_x_max
                or para_y_max < table_y_min
                or para_y_min > table_y_max
            ):
                return idx

        return None

    async def get_structured_table_text(self, table) -> str:
        """
        Converts a table into a pipe-separated format.
        """
        table_matrix = [
            ["" for _ in range(table.column_count)] for _ in range(table.row_count)
        ]
        for cell in table.cells:
            table_matrix[cell.row_index][cell.column_index] = cell.content
        return "\n".join(["| " + " | ".join(row) + " |" for row in table_matrix])

    async def get_text_from_azure_pdf(self, content_results, file_name: str) -> dict:
        """
        Parse the PDF file and extract text and tables using Azure Form Recognizer.
        
        Args:
            content_results: The parsed content results from Azure Form Recognizer.
            file_name: Name of the PDF file being processed.
            
        Returns:
            Dictionary with page numbers as keys and list of content dictionaries as values.
        """
        page_content = defaultdict(list)
        tables_by_page = defaultdict(list)
        table_polygons_by_page = defaultdict(list)

        # Extract tables
        if hasattr(content_results, "tables") and content_results.tables:
            for idx, table in enumerate(content_results.tables):
                if not table.bounding_regions:
                    logger.warning(f"Table {idx} has no bounding regions. Skipping.")
                    continue

                page_number = table.bounding_regions[0].page_number
                polygon = table.bounding_regions[0].polygon
                tables_by_page[page_number].append(table)
                table_polygons_by_page[page_number].append(polygon)

        # Extracting Text
        ignore_para_till = None
        if hasattr(content_results, "paragraphs") and content_results.paragraphs:
            title = None
            section_heading = None
            
            for paragraph in content_results.paragraphs:
                if not paragraph.bounding_regions:
                    logger.warning("Paragraph has no bounding regions. Skipping.")
                    continue

                page_number = paragraph.bounding_regions[0].page_number
                paragraph_polygon = paragraph.bounding_regions[0].polygon
                para_start = min(span.offset for span in paragraph.spans)

                # Handle different paragraph roles
                if paragraph.role == "title":
                    title = paragraph.content
                elif paragraph.role == "sectionHeading":
                    section_heading = paragraph.content
                elif paragraph.role == "pageNumber":
                    page_content[page_number].append({
                        "type": "pageNumber",
                        "content": paragraph.content,
                        "polygon": paragraph_polygon,
                        "file_name": file_name,
                    })

                # Skip paragraphs already covered by tables
                if ignore_para_till and para_start < ignore_para_till:
                    continue

                # Detect overlap with tables
                overlap_table_idx = await self.detect_overlap(
                    paragraph_polygon, table_polygons_by_page, page_number
                )
                
                if overlap_table_idx is not None:
                    # Handle table content
                    table = tables_by_page[page_number][overlap_table_idx]
                    caption = table.caption
                    table_text = await self.get_structured_table_text(table)

                    table_entry = {
                        "type": "table",
                        "content": table_text,
                        "title": title,
                        "sectionHeading": section_heading,
                        "file_name": file_name,
                        "caption": caption.content if caption else None,
                    }

                    if table_entry not in page_content[page_number]:
                        page_content[page_number].append(table_entry)

                    ignore_para_till = max(span.offset + span.length for span in table.spans)
                else:
                    # Handle regular paragraph content
                    if page_content[page_number]:
                        # Check if we can merge with existing paragraph
                        for para in page_content[page_number]:
                            if (
                                para["type"] == "paragraph"
                                and para["title"] == title
                                and para["sectionHeading"] == section_heading
                            ):
                                para["content"] += "\n" + paragraph.content
                                break
                        else:
                            # Add new paragraph
                            page_content[page_number].append({
                                "type": "paragraph",
                                "content": paragraph.content,
                                "title": title,
                                "sectionHeading": section_heading,
                                "file_name": file_name,
                            })
                    else:
                        # First paragraph on page
                        page_content[page_number].append({
                            "type": "paragraph",
                            "content": paragraph.content,
                            "title": title,
                            "sectionHeading": section_heading,
                            "file_name": file_name,
                        })

        return page_content

    async def remove_extra_new_line(self, text: str) -> str:
        """Remove excessive newlines from text."""
        extra_newlines_pattern = r'\n{3,}'
        cleaned_text = re.sub(extra_newlines_pattern, '\n\n', text)
        return cleaned_text

    async def extract_outlines(self, file_path: str):
        """Extract all headers and associate content within each header."""
        outlines = []
        header_obj = dict(header="_root", content="")
        outlines.append(header_obj)
        
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    if "# Page" in line:
                        continue
                    header_obj = dict(header=line.strip(), content="")
                    outlines.append(header_obj)
                else:
                    header_obj["content"] += line.strip() + "\n"
                    
        for element in outlines:
            element["content"] = await self.remove_extra_new_line(element["content"])

        return outlines

    async def remove_table_and_figure_tags(self, file_path: str) -> tuple:
        """
        Remove table and figure tags and their contents from a Markdown file.
        
        Args:
            file_path: Path to the Markdown file.
            
        Returns:
            Tuple of (cleaned_content, new_file_path)
        """
        with open(file_path, 'r') as file:
            content = file.read()

        # Regex patterns to match table and figure tags and their contents
        table_pattern = r'(?s)<table.*?>.*?</table>'
        figure_pattern = r'(?s)<figure.*?>.*?</figure>'
        pagebreak_pattern = r'<!--\s*PageBreak\s*-->'

        # Remove the matched patterns
        cleaned_content = re.sub(table_pattern, '', content)
        cleaned_content = re.sub(figure_pattern, '', cleaned_content)
        cleaned_content = re.sub(pagebreak_pattern, '', cleaned_content)

        # Write the cleaned content back to a new file
        new_path = file_path.split(".md")[0] + "_cleaned" + ".md"
        with open(new_path, 'w') as file:
            file.write(cleaned_content)
            
        return cleaned_content, new_path

    async def get_semantically_chunked_text(self, cleaned_data: list, chunk_size: int = 2000):
        """
        Process text data and chunk large content.
        Note: This is a simplified version - you'll need to implement semantic chunking.
        """
        semantic_data = []
        
        for element in cleaned_data:
            if element.get('content') == "" or len(element.get('content')) < 100:
                continue
                
            content = element.get('content')
            if len(content) > chunk_size:
                # Simple chunking - replace with actual semantic chunking
                chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
                for chunk in chunks:
                    new_element = deepcopy(element)
                    new_element['content'] = chunk
                    semantic_data.append(new_element)
            else:
                semantic_data.append(element)
                
        return semantic_data

    async def extract_text_from_pdf(
        self, 
        document: bytes, 
        endpoint: str, 
        key: str, 
        input_path: str
    ) -> List[Dict]:
        """
        Main method to extract text from PDF.
        
        Args:
            document: PDF document as bytes
            endpoint: Azure endpoint
            key: Azure key
            input_path: Full path to the input PDF file
            
        Returns:
            List of dictionaries containing extracted text content
        """
        # Get file name from input path
        file_name = os.path.basename(input_path)
        
        # Convert input_path to absolute path
        input_path = os.path.abspath(input_path)
        input_dir = os.path.dirname(input_path)
        
        # Generate output directory names based on input directory
        base_name = os.path.basename(input_dir)
        parent_dir = os.path.dirname(input_dir)
        
        # Create output directories
        original_dir = os.path.join(parent_dir, f"{base_name}_original")
        cleaned_dir = os.path.join(parent_dir, f"{base_name}_text")
        chunks_dir = os.path.join(parent_dir, f"{base_name}_chunks")
        
        # Get relative path for output files
        rel_path = os.path.relpath(os.path.dirname(input_path), input_dir)
        
        # Construct output paths maintaining directory structure
        base_name = os.path.splitext(file_name)[0]
        original_path = os.path.join(original_dir, rel_path, f"{base_name}.md")
        cleaned_path = os.path.join(cleaned_dir, rel_path, f"{base_name}.txt")
        chunks_path = os.path.join(chunks_dir, rel_path, f"{base_name}.json")
        
        # Ensure output directories exist
        ensure_dir_exists(original_path)
        ensure_dir_exists(cleaned_path)
        ensure_dir_exists(chunks_path)
        
        # Extract content using Azure
        content_results = await self.extract_content_from_pdf(document, endpoint, key)
        if not content_results:
            logger.error("Failed to extract content from PDF")
            return []

        # Save original markdown content
        with open(original_path, 'w', encoding='utf-8') as f:
            f.write(content_results.content)
        logger.info(f"Saved original content to: {original_path}")

        # Remove tables and figures for text-only extraction
        cleaned_content, _ = await self.remove_table_and_figure_tags(original_path)
        
        # Save cleaned text
        with open(cleaned_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        logger.info(f"Saved cleaned text to: {cleaned_path}")
        
        # Extract outlines/headers
        outline_dict = await self.extract_outlines(cleaned_path)
        
        # Process and chunk the content
        semantic_data = await self.get_semantically_chunked_text(outline_dict)
        
        # Format the final output
        final_documents = []
        for content in semantic_data:
            doc = {
                "content": content["content"],
                "context-type": "text",
                "metadata": {
                    "page_number": 0,
                    "title": content.get("header", ""),
                    "file_name": file_name,
                    "file_path": input_path,
                    "relative_path": os.path.join(rel_path, file_name)
                }
            }
            final_documents.append(doc)
            
        # Save the extracted data to a JSON file
        try:
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "chunks": final_documents,
                    "file_name": file_name,
                    "file_path": input_path,
                    "relative_path": os.path.join(rel_path, file_name)
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved extracted data to: {chunks_path}")
        except Exception as e:
            logger.error(f"Error saving extracted data: {str(e)}")
            
        logger.info(f"Extracted {len(final_documents)} text chunks from PDF")
        return final_documents


# # # Usage example:
# # async def main():
# #     """
# #     Example usage of the PDF text extractor.
# #     """
# #     extractor = PDFTextExtractor()
    
# #     # Your Azure credentials
# #     endpoint = "your_azure_endpoint"
# #     key = "your_azure_key"
    
# #     # Example input directory structure
# #     input_dir = "/path/to/your/pdfs"
# #     pdf_file = "example.pdf"
# #     input_path = os.path.join(input_dir, pdf_file)
    
# #     # Read PDF file
# #     with open(input_path, "rb") as f:
# #         pdf_bytes = f.read()
    
# #     # Extract text
# #     extracted_data = await extractor.extract_text_from_pdf(
# #         document=pdf_bytes,
# #         endpoint=endpoint,
# #         key=key,
# #         input_path=input_path
# #     )
    
# #     # Print results
# #     for i, doc in enumerate(extracted_data):
# #         print(f"Document {i+1}:")
# #         print(f"Title: {doc['metadata']['title']}")
# #         print(f"Content: {doc['content'][:200]}...")
# #         print("-" * 50)

# # if __name__ == "__main__":
# #     import asyncio
# #     asyncio.run(main())