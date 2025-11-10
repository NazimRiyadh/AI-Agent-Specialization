import * as dotenv from 'dotenv';
dotenv.config();
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import readline from 'readline';
import { Pinecone } from '@pinecone-database/pinecone';
import OpenAI from 'openai';

console.log("Initializing...");

// Initialize outside the function so history persists
const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model: 'text-embedding-004',
});

const pinecone = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY
});
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY
});

const conversationHistory = [];

console.log("Initialization complete!");

// Create readline interface
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

function askQuestion(query) {
    return new Promise(resolve => rl.question(query, resolve));
}

async function chatting(question) {
    try {
        console.log("Processing your question...");
        
        // Generate embedding for the question
        const queryVector = await embeddings.embedQuery(question);
        console.log("Embedding generated, searching Pinecone...");

        // Search Pinecone
        const searchResults = await pineconeIndex.query({
            topK: 10,
            vector: queryVector,
            includeMetadata: true,
        });
        
        console.log(`Found ${searchResults.matches.length} matches`);

        // Extract context from search results
        const context = searchResults.matches
            .map(match => match.metadata.text)
            .join("\n\n---\n\n");

        console.log("Generating response...");

        // Add user message to history
        conversationHistory.push({
            role: 'user',
            content: question
        });

        // Generate response using GPT-4o
        const completion = await openai.chat.completions.create({
            model: "gpt-4o",
            messages: [
                {
                    role: "system",
                    content: `You are a Data Structure and Algorithm Expert.
You will be given a context of relevant information and a user question.
Your task is to answer the user's question based ONLY on the provided context.
If the answer is not in the context, you must say "I could not find the answer in the provided document."
Keep your answers clear, concise, and educational.

Context:
${context}`
                },
                ...conversationHistory
            ],
            temperature: 0.7,
            max_tokens: 1000
        });

        const responseText = completion.choices[0].message.content;

        // Add assistant response to history
        conversationHistory.push({
            role: 'assistant',
            content: responseText
        });

        console.log("\n=== Answer ===");
        console.log(responseText);
        console.log("==============\n");
        
    } catch (error) {
        console.error("Error in chatting function:", error.message);
        
        if (error.status === 429) {
            console.error("\n⚠️  Rate limit exceeded. Please wait and try again.");
        } else if (error.status === 401) {
            console.error("\n⚠️  Invalid API key. Please check your OPENAI_API_KEY in .env file.");
        } else {
            console.error(error);
        }
    }
}

async function main() {
    try {
        const userProblem = await askQuestion("\nAsk me anything about DSA (or 'exit' to quit) --> ");
        
        if (!userProblem.trim()) {
            console.log("Please enter a question.");
            return main();
        }
        
        if (userProblem.toLowerCase() === 'exit' || userProblem.toLowerCase() === 'quit') {
            console.log("Goodbye!");
            rl.close();
            process.exit(0);
        }
        
        await chatting(userProblem);
        main();
    } catch (error) {
        console.error("Error in main function:", error.message);
        console.error(error);
        rl.close();
        process.exit(1);
    }
}

console.log("Starting chat interface...");
main();