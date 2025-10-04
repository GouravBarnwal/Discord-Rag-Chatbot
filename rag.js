import dotenv from "dotenv";
dotenv.config();
import faiss from "faiss-node";
import fs from "fs";
import crypto from "crypto";
import fetch from 'node-fetch';

// Ollama API configuration
const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || 'http://localhost:11434/api';

// Check if OLLAMA_BASE_URL is set
if (!process.env.OLLAMA_BASE_URL) {
  console.warn('OLLAMA_BASE_URL not set in .env, using default: http://localhost:11434/api');
}

// Function to get embeddings using Ollama
async function getEmbedding(text) {
  try {
    const response = await fetch(`${OLLAMA_BASE_URL}/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'all-minilm',
        prompt: text
      })
    });
    
    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.statusText}`);
    }
    
    const data = await response.json();
    return data.embedding || fallbackEmbeddingSync(text, 768);
  } catch (error) {
    console.error('Error in getEmbedding:', error.message);
    return fallbackEmbeddingSync(text, 768);
  }
}

// Fallback embedding function
function fallbackEmbeddingSync(text, dim = 768) {
  const floats = new Array(dim).fill(0);
  let i = 0, counter = 0;
  while (i < dim) {
    const h = crypto.createHash("sha256").update(text + "|" + counter).digest();
    for (let b = 0; b < h.length && i < dim; b++) {
      floats[i++] = (h[b] / 255.0) * 2 - 1; // Scale to [-1, 1]
    }
    counter++;
  }
  return floats;
}

// Function to generate response using Mistral
async function generateResponse(context, question) {
  console.log('[DEBUG] Inside generateResponse');
  const controller = new AbortController();
  const timeoutId = setTimeout(() => {
    console.log('[DEBUG] Request timed out after 60 seconds');
    controller.abort();
  }, 60000); // Increased to 60 seconds

  try {
    // Create a more focused prompt with limited context
    const prompt = `[INST] <|system|>You are a helpful assistant. Answer based on the resume context.</s>
<|user|>Context: ${context.substring(0, 1000)}...

Question: ${question}

Answer in 1 sentence: [/INST]`;
    
    console.log(`[DEBUG] Sending request to Ollama with prompt length: ${prompt.length}`);

    const requestBody = {
      model: process.env.OLLAMA_MODEL || 'phi3:mini',
      prompt: prompt,
      stream: false,
      options: {
        num_ctx: 2048,
        temperature: 0.7,
        top_p: 0.9,
        num_predict: 50,  // Shorter responses
        stop: ['\n']     // Stop at newlines
      }
    };
    
    console.log(`[DEBUG] Using model: ${requestBody.model}`);
    
    console.log('[DEBUG] Sending request to Ollama...');
    const startTime = Date.now();
    
    const response = await fetch(`${OLLAMA_BASE_URL}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody),
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    const endTime = Date.now();
    console.log(`[DEBUG] Response received in ${(endTime - startTime) / 1000} seconds`);
    
    console.log(`[DEBUG] Response status: ${response.status}`);

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`[ERROR] Ollama API error (${response.status}): ${errorText}`);
      throw new Error(`Ollama API error (${response.status}): ${response.statusText}`);
    }

    const data = await response.json();
    console.log('[DEBUG] Response data:', JSON.stringify(data, null, 2));
    
    if (!data || !data.response) {
      throw new Error('Empty or invalid response from Ollama');
    }
    
    return data.response.trim();
  } catch (error) {
    console.error('Error in generateResponse:', error.message);
    return "I'm having trouble generating a response right now. Please try again later.";
  }
}

// Initialize RAG system
const dataDir = "./data";
let docs = [];
let embeddings = [];
let index;
let validDocs = [];

async function initRAG() {
  // Ensure data directory exists
  if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true });
  }

  // Create a sample file if no files exist
  let files = fs.readdirSync(dataDir);
  if (files.length === 0) {
    const sampleContent = `This is a sample document for the RAG bot. 
It contains information about how the bot works and what it can do.
The bot uses semantic search to find relevant information and generate responses.
You can add more documents to the data folder to expand its knowledge base.`;
    
    const sampleFilePath = `${dataDir}/sample.txt`;
    fs.writeFileSync(sampleFilePath, sampleContent);
    console.log('Created sample document in data directory');
    files = [sampleFilePath];
  }

  // Load and process documents
  const pdfModule = await import("pdf-parse/lib/pdf-parse.js").catch(() => null);
  const pdf = pdfModule ? (pdfModule.default || pdfModule) : null;

  docs = await Promise.all(
    files.map(async (f) => {
      const full = `${dataDir}/${f}`;
      if (f.toLowerCase().endsWith(".pdf")) {
        if (!pdf) throw new Error("pdf-parse not loaded.");
        const buf = fs.readFileSync(full);
        const out = await pdf(buf);
        return out.text;
      } else {
        return fs.readFileSync(full, "utf8");
      }
    })
  );

  // Generate embeddings
  console.log('Starting to generate embeddings...');
  try {
    embeddings = await Promise.all(
      docs.map(async (text, i) => {
        console.log(`Processing document ${i + 1}/${docs.length}`);
        if (!text || !text.trim()) {
          console.warn(`Document ${i} is empty`);
          return null;
        }
        return await getEmbedding(text.slice(0, 8000));
      })
    );

    // Filter out failed embeddings
    const validEmbeddings = embeddings.filter(e => e && e.length > 0);
    if (validEmbeddings.length === 0) {
      throw new Error("No valid embeddings were generated");
    }

    // Initialize FAISS index
    const dim = validEmbeddings[0].length;
    index = new faiss.IndexFlatL2(dim);
    
    // Add valid embeddings to index
    const flat = Array.prototype.concat(...validEmbeddings);
    if (flat.length % dim !== 0) {
      throw new Error("Invalid flattened embeddings length for FAISS index.");
    }
    
    console.log(`Adding ${validEmbeddings.length} embeddings to FAISS index...`);
    index.add(flat);
    validDocs = docs.filter((_, i) => embeddings[i] && embeddings[i].length > 0);
    console.log('FAISS index created successfully');
  } catch (error) {
    console.error('Error initializing RAG system:', error);
    throw error;
  }
}

// Query the RAG system
async function queryRAG(question, k = 3) {
  if (!index || !validDocs || validDocs.length === 0) {
    throw new Error("RAG system not initialized. Call initRAG() first.");
  }

  console.log(`[DEBUG] Starting query for: "${question}"`);
  
  try {
    console.log('[DEBUG] Getting embedding for question...');
    const queryEmbedding = await getEmbedding(question);
    console.log('[DEBUG] Got embedding, searching index...');
    
    // Determine the actual number of documents to retrieve (min of k and available docs)
    const actualK = Math.min(k, validDocs.length);
    
    // Search for similar documents
    const { labels, distances } = index.search(queryEmbedding, actualK);
    
    if (!labels || labels.length === 0 || !validDocs || validDocs.length === 0) {
      return "I couldn't find any relevant information to answer that question.";
    }

    // Get the most relevant context
    console.log('[DEBUG] Found matching documents, extracting context...');
    const contextIndices = Array.isArray(labels[0]) ? labels[0] : [labels[0]];
    const context = contextIndices
      .filter(i => i >= 0 && i < validDocs.length)
      .map(i => validDocs[i])
      .join("\n\n");

    if (!context) {
      console.log('[DEBUG] No relevant context found');
      return "I couldn't find any relevant information to answer that question.";
    }

    console.log('[DEBUG] Generating response...');
    const response = await generateResponse(context, question);
    console.log('[DEBUG] Response generated successfully');
    return response;
  } catch (error) {
    console.error('Error in queryRAG:', error);
    return "I encountered an error while processing your request.";
  }
}

// Export the functions that need to be used by other modules
export { initRAG, queryRAG };