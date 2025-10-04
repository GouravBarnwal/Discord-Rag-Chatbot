# RAG-Powered Discord Resume Bot

A smart Discord bot that showcases my professional background using Retrieval-Augmented Generation (RAG). Built with Node.js, Ollama, and FAISS, this bot provides interactive access to my resume, answering questions about my experience, skills, and projects with context-aware responses.

## ✨ Features

- 🔍 RAG-powered responses using local LLM (phi3:mini)
- 🤖 Discord slash commands for easy interaction
- 📄 Processes and indexes resume data from documents
- 🚀 Local AI processing with Ollama
- ⚡ Efficient similarity search with FAISS

## 🛠️ Tech Stack

- **Backend**: Node.js, Discord.js
- **AI/ML**: Ollama, phi3:mini
- **Search**: FAISS
- **Environment**: dotenv

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/GouravBarnwal/Discord-Rag-Chatbot.git
   cd Discord-Rag-Chatbot
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Set up environment variables in `.env`:
   ```
   DISCORD_TOKEN=your_discord_bot_token
   CLIENT_ID=your_discord_client_id
   GUILD_ID=your_discord_server_id
   OLLAMA_BASE_URL=http://localhost:11434/api
   OLLAMA_MODEL=phi3:mini
   ```

4. Add your resume documents to the `./data` directory

5. Start the bot:
   ```bash
   node bot.js
   ```

## 🤖 Usage

- Use the `/ask` command in your Discord server to ask questions about my professional background
- Example: `/ask What projects have you worked on?`

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Made with ❤️ by Gourav Barnwal | [Portfolio](https://gouravs-portfolio.vercel.app/)
