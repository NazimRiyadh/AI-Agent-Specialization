import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import * as dotenv from 'dotenv';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';

dotenv.config();

async function indexDocument() {
  // Load PDF
  const PDF_PATH = './Dsa.pdf';
  const pdfLoader = new PDFLoader(PDF_PATH);
  const rawDocs = await pdfLoader.load();
  console.log("document loaded:", rawDocs.length, "pages");

  // Chunk documents
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const chunkDocs = await textSplitter.splitDocuments(rawDocs);
  console.log("document chunked into", chunkDocs.length, "chunks");

  // Create embeddings
  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model: 'text-embedding-004',
  });
  console.log("embeddings object created");

  // Initialize Pinecone client (new way)
  const pinecone = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  });

  const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

  // Index documents into Pinecone
  try {
    await PineconeStore.fromDocuments(chunkDocs, embeddings, {
      pineconeIndex,
      maxConcurrency: 5,
    });
    console.log("documents indexed into Pinecone");
  } catch (err) {
    console.error("Error indexing into Pinecone:", err);
  }
}

// Run the function
indexDocument()
  .then(() => console.log("Done indexing document!"))
  .catch((err) => console.error(err));
