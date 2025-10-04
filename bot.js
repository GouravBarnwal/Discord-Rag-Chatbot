import { Client, GatewayIntentBits } from "discord.js";
import { REST } from "@discordjs/rest";
import { Routes } from "discord-api-types/v10";
import { SlashCommandBuilder } from "@discordjs/builders";
import dotenv from "dotenv";
import { queryRAG, initRAG } from "./rag.js";

dotenv.config();

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.MessageContent,
  ],
});

client.once("ready", (c) => console.log(`Bot logged in as ${c.user.tag}`));

client.on("messageCreate", async (msg) => {
  if (msg.author.bot) return;
  const text = (msg.content || "").trim();
  if (!text) return;

  const pending = await msg.reply("Processing...");
  try {
    const answer = await queryRAG(text);
    await pending.edit(answer);
  } catch (err) {
    console.error(err);
    await pending.edit("Error while generating answer.");
  }
});

client.on("interactionCreate", async (interaction) => {
  if (!interaction.isChatInputCommand()) return;
  if (interaction.commandName === "ask") {
    const question = interaction.options.getString("question", true);
    await interaction.deferReply();
    try {
      const answer = await queryRAG(question);
      await interaction.editReply(answer);
    } catch (err) {
      console.error(err);
      await interaction.editReply("Error while generating answer.");
    }
  }
});

async function main() {
  const { DISCORD_TOKEN, CLIENT_ID, GUILD_ID } = process.env;

  if (!DISCORD_TOKEN) {
    console.error("Missing DISCORD_TOKEN in .env");
    process.exit(1);
  }

  try {
    console.log("Initializing RAG system...");
    await initRAG();
    console.log("RAG system initialized successfully");
  } catch (err) {
    console.error("Failed to initialize RAG system:", err);
    process.exit(1);
  }

  if (!DISCORD_TOKEN) {
    console.error("Missing DISCORD_TOKEN in .env");
    process.exit(1);
  }

  try {
    await client.login(DISCORD_TOKEN);
  } catch (err) {
    console.error("Failed to login:", err.message);
    process.exit(1);
  }

  if (CLIENT_ID && GUILD_ID) {
    const rest = new REST({ version: "10" }).setToken(DISCORD_TOKEN);
    const commands = [
      new SlashCommandBuilder()
        .setName("ask")
        .setDescription("Ask the RAG bot a question")
        .addStringOption((opt) =>
          opt.setName("question").setDescription("Your question").setRequired(true)
        )
        .toJSON(),
    ];
    try {
      console.log("Registering /ask command to guild", GUILD_ID);
      await rest.put(Routes.applicationGuildCommands(CLIENT_ID, GUILD_ID), {
        body: commands,
      });
      console.log("Registered commands.");
    } catch (err) {
      console.error("Command registration failed:", err);
    }
  } else {
    console.log("CLIENT_ID or GUILD_ID missing; skipping slash command registration.");
  }
}

main();
