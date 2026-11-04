"""
Generates a valid, multi-page sample technical PDF whitepaper for testing the RAG system.
Uses raw PDF 1.4 specification without external dependencies.
"""

import os


def generate_pdf(output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 3-page technical AI whitepaper
    pages_content = [
        # Page 1
        "Page 1: Executive Summary - The Rise of Multimodal AI & RAG\n\n"
        "Modern artificial intelligence systems are transitioning from unimodal text models to comprehensive multimodal architectures. "
        "A multimodal system processes text, speech, and visual information in tandem. "
        "In enterprise document intelligence, Retrieval-Augmented Generation (RAG) bridges foundational LLMs with proprietary enterprise knowledge. "
        "Key challenges in vanilla RAG pipelines include semantic hallucination, out-of-domain query failures, and high inference latency. "
        "To mitigate these issues, multi-stage retrieval workflows employ cosine similarity scoring and strict confidence thresholds.",

        # Page 2
        "Page 2: Architecture & Vector Database Benchmarking\n\n"
        "Embedding models transform unstructured chunked text into dense vector representations. "
        "The all-MiniLM-L6-v2 model maps sentences into a 384-dimensional dense vector space and is optimized for low-latency similarity tasks. "
        "ChromaDB serves as the persistent vector database, utilizing Hierarchical Navigable Small World (HNSW) indexing for approximate nearest neighbor search. "
        "During multi-stage retrieval, an initial candidate pool of 2k chunks is filtered by cosine similarity. "
        "Chunks exhibiting similarity scores below the 0.40 confidence threshold are eliminated before prompting the LLM.",

        # Page 3
        "Page 3: Ultra-Low-Latency Inference with Groq & Guardrails\n\n"
        "Inference speed is critical for interactive document intelligence. "
        "By integrating the Groq LPU (Language Processing Unit), inference speeds exceed 250 tokens per second for Llama 3.3 70B models. "
        "Every retrieved chunk is linked to source metadata, including exact page numbers and chunk hashes. "
        "If no chunks meet the confidence threshold, the hallucination guardrail triggers an automated refusal response. "
        "This architecture guarantees that all generated answers are auditable, verifiable, and strictly grounded in source documents."
    ]

    def escape_pdf(text):
        return text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')

    # Minimal valid PDF writer
    objects = []
    
    # Object 1: Catalog
    # Object 2: Pages
    # Object 3, 4, 5: Page objects
    # Object 6, 7, 8: Content streams
    # Object 9: Font
    num_pages = len(pages_content)
    page_obj_ids = [3 + i for i in range(num_pages)]
    content_obj_ids = [3 + num_pages + i for i in range(num_pages)]
    font_obj_id = 3 + 2 * num_pages

    # 1: Catalog
    catalog = "<< /Type /Catalog /Pages 2 0 R >>"
    
    # 2: Pages
    kids_str = " ".join([f"{pid} 0 R" for pid in page_obj_ids])
    pages_obj = f"<< /Type /Pages /Kids [{kids_str}] /Count {num_pages} >>"

    # Pages & Streams
    page_objs = []
    stream_objs = []
    
    for i in range(num_pages):
        page_obj = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_obj_ids[i]} 0 R /Resources << /Font << /F1 {font_obj_id} 0 R >> >> >>"
        )
        page_objs.append(page_obj)

        lines = pages_content[i].split("\n\n")
        stream_text = "BT /F1 12 Tf 50 720 Td 18 TL "
        for paragraph in lines:
            # Wrap text roughly every 75 characters
            words = paragraph.split(" ")
            curr_line = ""
            for w in words:
                if len(curr_line) + len(w) + 1 > 75:
                    stream_text += f"({escape_pdf(curr_line)}) ' "
                    curr_line = w
                else:
                    curr_line = (curr_line + " " + w).strip()
            if curr_line:
                stream_text += f"({escape_pdf(curr_line)}) ' "
            stream_text += "T* "
        stream_text += "ET"

        stream_data = f"<< /Length {len(stream_text)} >>\nstream\n{stream_text}\nendstream"
        stream_objs.append(stream_data)

    font_obj = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    all_objs = [catalog, pages_obj] + page_objs + stream_objs + [font_obj]

    # Assemble PDF with cross-reference table
    pdf_output = "%PDF-1.4\n"
    xref_offsets = []

    for idx, obj in enumerate(all_objs, 1):
        xref_offsets.append(len(pdf_output))
        pdf_output += f"{idx} 0 obj\n{obj}\nendobj\n"

    xref_start = len(pdf_output)
    pdf_output += f"xref\n0 {len(all_objs) + 1}\n0000000000 65535 f \n"
    for offset in xref_offsets:
        pdf_output += f"{offset:010d} 00000 n \n"

    pdf_output += (
        f"trailer\n<< /Size {len(all_objs) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF"
    )

    with open(output_path, "wb") as f:
        f.write(pdf_output.encode("latin1"))

    print(f"Generated sample PDF at: {output_path}")


if __name__ == "__main__":
    out_file = os.path.join(os.path.dirname(__file__), "multimodal_ai_whitepaper.pdf")
    generate_pdf(out_file)
