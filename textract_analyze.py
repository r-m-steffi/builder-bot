import boto3
import json
import pandas as pd
from trp import Document

from opensearchpy import OpenSearch

import json
import boto3

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

region = 'us-east-2'
bucket_name = 'ottawa-builder-bucket'
document_name = 'sample_builder_report_with_table.pdf'
json_path = "/home/steffi/aisd/Hackathon/fake_reports/sample_reports_pdf/sample_builder_report_with_table.json"

# OpenSearch config
from dotenv import load_dotenv
import os

load_dotenv()

opensearch_host = os.getenv("OPENSEARCH_HOST")
opensearch_index = os.getenv("OPENSEARCH_INDEX")
opensearch_auth = (
    os.getenv("OPENSEARCH_USER"),
    os.getenv("OPENSEARCH_PASS")
 ) # Or use IAM auth if secured
os_client = OpenSearch(
    hosts=[{"host": opensearch_host, "port": 443}],
    http_auth=opensearch_auth,
    use_ssl=True,
    verify_certs=True
)
# Setup Textract client
textract = boto3.client('textract', region_name=region)  # or your region
#bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")


# Call Textract
response = textract.analyze_document(
    Document={
        'S3Object': {
            'Bucket': bucket_name,
            'Name': document_name
        }
    },
    FeatureTypes=["TABLES", "FORMS"]
)

# Print full JSON (you can save it too)

print("Textract response keys:", response.keys())
# Parse using trp
doc = Document(response)

# Extract and save tables
for page_num, page in enumerate(doc.pages, 1):
    for table_num, table in enumerate(page.tables, 1):
        rows = []
        for row in table.rows:
            rows.append([cell.text for cell in row.cells])

        header, *data = rows
        df = pd.DataFrame(data, columns=header)

        # Save
        base = json_path.replace('.json', f'_page{page_num}_table{table_num}')
        df.to_json(base + '.json', orient="records", indent=2)
        df.to_csv(base + '.csv', index=False)

        print(f"✅ Saved table: {base + '.csv'}")
        records = df.to_dict(orient='records')
        document = {
            "builder": "ABC Homes",  # You could extract this from FORM fields
            "year": 2024,            # Or infer from filename/metadata
            "page": page_num,
            "table": table_num,
            "issues": records
        }

        res = os_client.index(index=opensearch_index, body=document)
        print(f"📤 Indexed to OpenSearch: ID={res['_id']}")

# Print just the detected lines of text
#for block in response.get('Blocks', []):
 #   if block.get('BlockType') == 'LINE':
  #      print(block.get('Text', ''))

lines = []
for block in response.get('Blocks', []):
    if block.get('BlockType') == 'LINE':
        lines.append(block.get('Text', ''))

full_text = "\n".join(lines)
with open("full_text_report.txt", "w") as f:
    f.write(full_text)

search_term = "cracks and plumbing issues"
search_results = os_client.search(index=opensearch_index, body={
    "query": {
        "match": {
            "text": search_term
        }
    }
})

def chunk_text(text, max_length=500):
    sentences = text.split(". ")
    chunks, chunk = [], ""
    for sentence in sentences:
        if len(chunk) + len(sentence) <= max_length:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + ". "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

text_chunks = chunk_text(full_text)


    
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(text_chunks)
import faiss
import numpy as np

dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
print("FAISS Index Size:", index.ntotal)


chunk_lookup = {i: chunk for i, chunk in enumerate(text_chunks)}
question = "What kind of issues were found in the reports?"

query_embedding = model.encode([question])
D, I = index.search(np.array(query_embedding), k=3)

retrieved_chunks = [chunk_lookup[i] for i in I[0] if i != -1]

context = "\n".join(retrieved_chunks)
print(context)
import faiss, json, numpy as np

# Save FAISS index
faiss.write_index(index, "builder_index.faiss")

# Save chunk mapping
with open("chunk_lookup.json", "w") as f:
    json.dump(chunk_lookup, f)