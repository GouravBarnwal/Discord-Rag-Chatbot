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
  console.log(`[DEBUG] Context length: ${context ? context.length : 0} chars`);
  console.log(`[DEBUG] Question: ${question}`);
  
  const controller = new AbortController();
  const timeoutMs = 30000; // Reduced from 60s to 30s
  const timeoutId = setTimeout(() => {
    console.log(`[DEBUG] Request timed out after ${timeoutMs/1000} seconds`);
    controller.abort();
  }, timeoutMs);

  try {
    let prompt;
    
    if (context) {
      // For project/experience related questions with context
      prompt = `[INST] <|system|>You are a helpful assistant. Answer based on the provided context.</s>
<|user|>Context: ${context.substring(0, 1000)}...

Question: ${question}

Answer helpfully: [/INST]`;
    } else {
      // For general questions without specific context
      prompt = `[INST] <|system|>You are a helpful AI assistant. Answer the following question helpfully and concisely.</s>
<|user|>${question}[/INST]`;
    }
    
    console.log(`[DEBUG] Sending request to Ollama with prompt length: ${prompt.length}`);

    const requestBody = {
      model: process.env.OLLAMA_MODEL || 'phi3:mini',
      prompt: prompt,
      stream: false,
      options: {
        num_ctx: 2048,
        temperature: 0.7,
        top_p: 0.9,
        num_predict: context ? 100 : 200,  // Longer responses for general questions
        stop: context ? ['\n'] : undefined  // Only stop at newlines for project questions
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

// Function to clean and normalize text from PDF
function cleanText(text) {
  if (!text) return '';
  
  // First, normalize line breaks and spaces
  let cleaned = text
    .replace(/\r\n|\r/g, '\n')      // Normalize line endings
    .replace(/\f/g, '\n')            // Form feed to newline
    .replace(/\u00A0/g, ' ')         // Replace non-breaking spaces
    .replace(/[\u200B-\u200D\uFEFF]/g, '') // Remove zero-width spaces
    .replace(/[^\x00-\x7F]/g, ' ');  // Replace non-ASCII characters with space

  // Handle bullet points and lists
  cleaned = cleaned
    .replace(/•|\*|\-\s+/g, '• ')  // Normalize bullet points
    .replace(/\s*\n\s*•/g, '\n•')  // Ensure proper bullet point formatting
    .replace(/\s*\n\s*/g, '\n')    // Clean up spaces around newlines
    .replace(/\s+/g, ' ')           // Replace multiple spaces with single space
    .trim();

  // Remove common PDF artifacts and weird patterns
  const artifacts = [
    /\b(?:page|pg|p\.?)\s*\d+\b/gi,  // Page numbers
    /\b\d{1,2}\/\d{1,2}\/\d{2,4}\b/g,  // Dates
    /\b\d{1,2}[\-\/]\d{1,2}[\-\/]\d{2,4}\b/g,  // More dates
    /\b\w{1,2}\.\s*\d+\b/g,  // Section numbers (e.g., "1.1", "A.1")
    /\b\d{10,}\b/g,  // Long numbers (likely phone numbers or IDs)
    /\b\w{1,2}\/\w{1,2}\b/g  // Fractions or ratios
  ];

  artifacts.forEach(regex => {
    cleaned = cleaned.replace(regex, '');
  });

  // Remove empty lines and trim again
  return cleaned
    .split('\n')
    .map(line => line.trim())
    .filter(line => line.length > 0)
    .join('\n');
}

// Function to split text into meaningful chunks
function chunkText(text, chunkSize = 1000, overlap = 200) {
  const chunks = [];
  let start = 0;
  
  while (start < text.length) {
    let end = start + chunkSize;
    
    // Try to find a good breaking point (end of sentence)
    if (end < text.length) {
      const nextPeriod = text.indexOf('.', end);
      const nextNewline = text.indexOf('\n', end);
      
      if (nextPeriod !== -1 && (nextPeriod - end) < 100) {
        end = nextPeriod + 1;
      } else if (nextNewline !== -1 && (nextNewline - end) < 100) {
        end = nextNewline + 1;
      }
    }
    
    chunks.push(text.substring(start, end).trim());
    start = end - overlap; // Overlap chunks for better context
    
    if (start >= text.length) break;
  }
  
  return chunks;
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
    console.log(`Created data directory at: ${dataDir}`);
  } else {
    console.log(`Using existing data directory: ${dataDir}`);
  }

  // Get all files from data directory
  let files = fs.readdirSync(dataDir);
  
  // Filter for .txt files only
  files = files.filter(f => f.toLowerCase().endsWith('.txt'));

  if (files.length === 0) {
    console.warn('No .txt files found in data directory. Please add .txt files to process.');
    return;
  }

  // Process each file
  for (const file of files) {
    try {
      const fullPath = `${dataDir}/${file}`;
      console.log(`Processing file: ${file} (${fs.statSync(fullPath).size} bytes)`);
      
      // Read text file content
      const content = fs.readFileSync(fullPath, 'utf8');
      if (!content.trim()) {
        console.warn(`  File ${file} is empty`);
        continue;
      }
      
      // Clean and process the content
      let cleanedContent = cleanText(content);
      if (!cleanedContent) {
        console.warn(`No content extracted from ${file}`);
        continue;
      }
      
      // Add special markers for resume files
      if (file.toLowerCase().includes('resume') || file.toLowerCase().includes('cv')) {
        cleanedContent = `[RESUME_START]\n${cleanedContent}\n[RESUME_END]`;
      }
      
      // Split into chunks and add to docs
      const chunks = chunkText(cleanedContent);
      console.log(`Split ${file} into ${chunks.length} chunks`);
      docs.push(...chunks);
      
    } catch (error) {
      console.error(`Error processing file ${file}:`, error);
    }
  }

  // Generate embeddings in batches to avoid memory issues
  console.log('Starting to generate embeddings...');
  try {
    const batchSize = 10; // Process 10 documents at a time
    embeddings = [];
    
    for (let i = 0; i < docs.length; i += batchSize) {
      const batch = docs.slice(i, i + batchSize);
      console.log(`Processing documents ${i + 1} to ${Math.min(i + batchSize, docs.length)} of ${docs.length}`);
      
      const batchEmbeddings = await Promise.all(
        batch.map(async (text, idx) => {
          if (!text || !text.trim()) {
            console.warn(`Document ${i + idx} is empty`);
            return null;
          }
          try {
            // Take first 8000 characters to avoid context window issues
            const truncatedText = text.slice(0, 8000);
            const embedding = await getEmbedding(truncatedText);
            console.log(`  - Document ${i + idx + 1}: Generated embedding (${embedding.length} dimensions)`);
            return embedding;
          } catch (error) {
            console.error(`Error generating embedding for document ${i + idx}:`, error.message);
            return null;
          }
        })
      );
      
      embeddings.push(...batchEmbeddings);
    }

    // Filter out failed embeddings
    const validEmbeddings = [];
    validDocs = [];
    
    for (let i = 0; i < embeddings.length; i++) {
      if (embeddings[i] && embeddings[i].length > 0) {
        validEmbeddings.push(embeddings[i]);
        validDocs.push(docs[i]);
      }
    }
    
    if (validEmbeddings.length === 0) {
      throw new Error("No valid embeddings were generated");
    }

    console.log(`Initializing FAISS index with ${validEmbeddings.length} valid embeddings...`);
    const dim = validEmbeddings[0].length;
    index = new faiss.IndexFlatL2(dim);
    
    // Add valid embeddings to index
    const flat = Array.prototype.concat(...validEmbeddings);
    if (flat.length % dim !== 0) {
      throw new Error("Invalid flattened embeddings length for FAISS index.");
    }
    
    console.log(`Adding ${validEmbeddings.length} embeddings to FAISS index...`);
    index.add(flat);
    console.log(`FAISS index created successfully with ${validDocs.length} documents`);
  } catch (error) {
    console.error('Error initializing RAG system:', error);
    throw error;
  }
}

// Function to format context from multiple documents
function formatContext(docs, scores = []) {
  return docs.map((doc, i) => {
    const scoreInfo = scores[i] !== undefined ? `[Relevance: ${scores[i].toFixed(2)}] ` : '';
    return `${scoreInfo}${doc}`;
  }).join('\n\n---\n\n');
}

// Function to check if a question is about skills
function isSkillsQuestion(question) {
  const skillKeywords = [
    'skill', 'technology', 'tech stack', 'programming', 'language',
    'framework', 'tool', 'expertise', 'proficient', 'experience with',
    'technologies', 'tools', 'languages', 'what can you do', 'what do you know',
    'what are you good at', 'what are your skills', 'what are you experienced in',
    'technical skills', 'technical expertise', 'proficiency', 'knowledge of'
  ];
  const questionLower = question.toLowerCase();
  return skillKeywords.some(keyword => questionLower.includes(keyword));
}

// Query the RAG system
async function queryRAG(question, k = 5) {
  if (!index || !validDocs || validDocs.length === 0) {
    throw new Error("RAG system not initialized. Call initRAG() first.")
  }

  console.log(`[DEBUG] Starting query for: "${question}"`);
  
  // Special handling for skills-related questions
  if (isSkillsQuestion(question)) {
    console.log('[DEBUG] Detected skills-related question');
    
    try {
      // First try to find skills in the most relevant chunks
      const queryEmbedding = await getEmbedding("technical skills and technologies");
      const { labels } = index.search(queryEmbedding, 5); // Get top 5 most relevant chunks
      
      const relevantChunks = [];
      const seenTexts = new Set();
      
      for (const idx of Array.isArray(labels[0]) ? labels[0] : [labels[0]]) {
        if (idx >= 0 && idx < validDocs.length) {
          const text = validDocs[idx];
          if (!seenTexts.has(text)) {
            relevantChunks.push(text);
            seenTexts.add(text);
          }
        }
      }
      
      // If no relevant chunks found, use all documents
      const contextText = relevantChunks.length > 0 
        ? relevantChunks.join('\n\n---\n\n')
        : validDocs.join('\n\n---\n\n');
      
      const skillsPrompt = `Extract and list all technical skills, programming languages, 
        frameworks, and tools mentioned in the following text. 
        Group them into categories if possible (e.g., "Programming Languages", "Frameworks", "Tools").
        Only include actual technologies and skills, no explanations or additional text.
        
        ${contextText}`;
      
      console.log('[DEBUG] Extracting skills from context...');
      const skills = await generateResponse("", skillsPrompt);
      
      // Clean up the response
      const cleanedSkills = skills
        .split('\n')
        .filter(line => line.trim().length > 0)
        .map(line => line.replace(/^[-•*]\s*/, '').trim())
        .join('\n');
      
      return `Here are the skills and technologies mentioned in the resume:\n\n${cleanedSkills}`;
      
    } catch (error) {
      console.error('Error extracting skills:', error);
      // Fall through to normal processing
    }
  }
  
  try {
    console.log('[DEBUG] Getting embedding for question...');
    const queryEmbedding = await getEmbedding(question);
    
    // Search for more documents to get better context
    const searchK = Math.min(k * 2, validDocs.length);
    console.log(`[DEBUG] Searching top ${searchK} most relevant documents...`);
    
    const { labels, distances } = index.search(queryEmbedding, searchK);
    
    if (!labels || labels.length === 0) {
      console.log('[DEBUG] No matching documents found, using general response');
      return await generateResponse("", question);
    }
    
    // Get the most relevant contexts with their scores
    const contextIndices = Array.isArray(labels[0]) ? labels[0] : [labels[0]];
    const contextScores = Array.isArray(distances[0]) ? distances[0] : [distances[0]];
    
    // Get the actual document chunks with additional metadata
    const contexts = contextIndices
      .map((idx, i) => {
        if (idx < 0 || idx >= validDocs.length) return null;
        
        const text = validDocs[idx];
        const score = contextScores[i];
        
        // Calculate text metrics for better ranking
        const wordCount = text.split(/\s+/).length;
        const lineCount = (text.match(/\n/g) || []).length + 1;
        const avgLineLength = text.length / Math.max(1, lineCount);
        
        // Penalize very short or very long chunks
        let qualityScore = score;
        if (wordCount < 5 || wordCount > 200) {
          qualityScore += 0.5; // Penalize very short or very long chunks
        }
        
        return {
          text: text,
          score: qualityScore,
          originalScore: score,
          wordCount: wordCount,
          lineCount: lineCount,
          avgLineLength: avgLineLength
        };
      })
      .filter(ctx => ctx !== null) // Filter out invalid indices
      .sort((a, b) => a.score - b.score)  // Sort by quality score (lower is better)
      .slice(0, k);  // Take top k most relevant
    
    if (contexts.length === 0) {
      console.log('[DEBUG] No valid contexts found');
      return await generateResponse("", question);
    }
    
    // Prepare context for the model
    const contextText = contexts.map(ctx => ctx.text).join('\n\n---\n\n');
    const averageScore = contexts.reduce((sum, ctx) => sum + ctx.score, 0) / contexts.length;
    
    console.log(`[DEBUG] Found ${contexts.length} relevant contexts (avg score: ${averageScore.toFixed(2)})`);
    
    // Adjust threshold based on query type
    const lowerQuestion = question.toLowerCase();
    const isResumeQuery = lowerQuestion.includes('resume') || 
                         lowerQuestion.includes('cv') ||
                         lowerQuestion.includes('experience') ||
                         lowerQuestion.includes('skill') ||
                         lowerQuestion.includes('bio') ||
                         lowerQuestion.includes('about me');
    
    // Be more lenient with resume queries since they're important
    const relevanceThreshold = isResumeQuery ? 2.5 : 1.0;
    
    if (contexts[0].score > relevanceThreshold) {
      console.log(`[DEBUG] Context not relevant enough (score: ${contexts[0].score}), using general response`);
      if (isResumeQuery) {
        try {
          // For resume queries, use a simpler prompt and limit context
          const allText = validDocs.join('\n\n');
          // Limit to first 2000 chars to prevent timeout
          const limitedResume = allText.length > 2000 
            ? allText.substring(0, 2000) + '... [truncated]' 
            : allText;
            
          // Super simple prompt with minimal context
          const extractPrompt = `Extract just these details from the resume (be brief):\n` +
            `- Name\n` +
            `- Current role\n` +
            `- Latest work experience\n` +
            `- Top 3 skills\n\n` +
            `Resume (first 1000 chars):\n${limitedResume.substring(0, 1000)}`;
            
          return await generateResponse("", extractPrompt);
        } catch (error) {
          console.error('Error processing resume:', error);
          return "I had trouble processing the resume. Please try asking something more specific.";
        }
      }
      return await generateResponse("", question);
    }
    
    // Generate response with the found context
    console.log('[DEBUG] Generating response with context...');
    try {
      // Use very small context to prevent timeouts
      let maxContextLength = 800;
      let limitedContext = contextText;
      
      // If we have resume content but it's too long, try to extract most relevant parts
      if (contextText.includes('[RESUME]') && contextText.length > maxContextLength) {
        // Try to keep the beginning and end of the resume
        const halfLength = Math.floor(maxContextLength / 2);
        limitedContext = 
          contextText.substring(0, halfLength) + 
          '... [middle truncated] ...' + 
          contextText.substring(contextText.length - halfLength);
      } else if (contextText.length > maxContextLength) {
        limitedContext = contextText.substring(0, maxContextLength) + '... [truncated]';
      }
      
      console.log(`[DEBUG] Sending context (${limitedContext.length} chars) to generate response...`);
      
      const response = await Promise.race([
        generateResponse(limitedContext, question),
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Response generation timed out after 30 seconds')), 30000)
        )
      ]);
      return response;
    } catch (error) {
      console.error('Error generating response:', error);
      // Try with just the most relevant chunk if the full context times out
      if (error.message.includes('timed out') && contexts.length > 1) {
        console.log('[DEBUG] Full context timed out, trying with just the most relevant chunk...');
        try {
          return await generateResponse(contexts[0].text, question);
        } catch (e) {
          console.error('Error with single chunk response:', e);
        }
      }
      throw error;
    }
    
  } catch (error) {
    console.error('Error in RAG query:', error);
    // Fall back to general response if RAG fails
    try {
      if (error.message.includes('timed out')) {
        return "I'm taking too long to process your request. The resume might be large. " +
               "Could you try asking a more specific question about the resume?";
      }
      return await generateResponse("", question);
    } catch (e) {
      console.error('Error in RAG fallback response:', e);
      return "I'm having trouble processing your request right now. The resume might be too large or complex. " +
             "Could you try asking a more specific question about the resume?";
    }
  }
}

// Export the functions that need to be used by other modules
export { initRAG, queryRAG };