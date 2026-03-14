# IRC-AI-bot
A self learning intelligent IRC bot
Here is a **clean GitHub repository description + README-style intro** you can use.

---

# IRC Self-Learning Chat Bot

A lightweight **self-learning IRC bot written in pure Python** that participates in channel conversations, learns from chat history, and builds long-term memory about users, slang, and topics.

The bot uses a remote LLM (via HuggingFace API) for generating replies while maintaining its own **local learning system** for context, memory, and behavior.

No external Python dependencies are required — it runs using only the **Python standard library**.

---

## Features

### Self-Learning Memory

The bot continuously learns from the IRC channel:

* remembers facts users mention
* tracks frequently discussed topics
* learns channel slang and memes
* builds simple user profiles
* remembers conversation context

Memory is stored locally and persists across restarts.

---

### Semantic Memory Retrieval

The bot includes a **pure-Python semantic memory system**:

* hash-based vector embeddings
* cosine similarity search
* approximate nearest-neighbor index
* long-term memory scoring

This allows the bot to recall relevant facts from past conversations when replying.

---

### Conversation Awareness

The bot keeps track of:

* recent chat history
* active topics
* common phrases used in the channel
* frequently repeated jokes or memes

Replies are generated using both **recent chat context** and **learned knowledge**.

---

### Personality System

The bot uses a built-in persona:

* casual IRC user
* short lowercase replies
* dry humor
* mild sarcasm
* linux/devops nerd personality

It also learns communication style from the channel and can imitate common short phrases.

---

### Human-Like Behavior

To avoid behaving like a typical bot:

* random reply chance
* cooldown between replies
* simulated typing delay
* spam detection
* waits for conversation activity before speaking

---

### Persistent Learning

The bot saves learned information to disk:

```
profiles.json
memories.json
topics.json
slang.json
styles.json
memes.json
chat_log.txt
```

This allows the bot to **improve over time as it observes the channel**.

---

## Requirements

* Python 3.8+
* HuggingFace API key

No additional libraries are required.

---

## Setup

Clone the repository:

```bash
git clone https://github.com/yourname/irc-learning-bot
cd irc-learning-bot
```

Add your HuggingFace API key in the script:

```python
HF_TOKEN = "hf_your_api_key_here"
```

Run the bot:

```bash
python bot.py
```

---

## Configuration

Inside the script you can configure:

```
SERVER
PORT
CHANNEL
NICK
MODEL
REPLY_COOLDOWN
RANDOM_REPLY_CHANCE
```

---

## Example Behavior

The bot may learn things like:

```
<alice> i run arch on my laptop
<bob> docker networking is cursed
<charlie> lol skill issue
```

Later it might reply:

```
lol docker networking is cursed yeah
```

or

```
alice runs arch btw
```

---

## Goals of the Project

This project explores how far a **self-learning chat bot can go without heavy ML libraries** by combining:

* local heuristic learning
* lightweight vector embeddings
* remote LLM reasoning
* persistent memory

---

## License

MIT License

---

If you want, I can also write a **much more viral GitHub README** (the kind that gets **tons of stars**) with:

* demo GIF section
* architecture diagram
* example conversation logs
* "How it learns" visual explanation.
