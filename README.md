# builder-bot
Building a Rag-based multimodal chatbot

# RAG Chatbot for City of Ottawa Builder Reports

## Description
Chatbot to answer questions from builder reports using AWS Textract, OpenSearch, and FAISS retrieval.

### Task
 #### Store reports on aws S3 bucket.
 #### Process the reports by loading, chuncking, embedding .
 #### Retrieve answer specific to the question.
 #### Whenever a new report is uploaded on cloud, The processing script should run
 #### Every question answer pair should be logged for monitoring.
 #### Idea is make everythig work on aws.

## Setup
1. Clone the repo
2. Install dependencies
3. Add `.env` file with your own OpenSearch/Textract credentials
4. Run `textract_analyze.py` to analyze and query documents

## Team
- @Priya
- @Sonam
- @Steffi
- @Yosufi

