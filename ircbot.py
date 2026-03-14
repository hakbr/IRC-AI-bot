import socket
import time
import threading
import random
import json
import os
import http.client
import re
import math
import hashlib

# ---------------- CONFIG ----------------

SERVER = "SERVER HERE"
PORT = 6667
CHANNEL = "INPUT_CHANNEL_HERE"
NICK = "INPUT_NICK_HERE"

HF_TOKEN = "HUGGING_FACE_API_KEY_HERE"

HF_HOST = "router.huggingface.co"
HF_PATH = "/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-7B-Instruct"

LOG_FILE = "chat_log.txt"

PROFILE_FILE = "profiles.json"
MEMORY_FILE = "memories.json"
TOPIC_FILE = "topics.json"
SUMMARY_FILE = "summaries.json"
SLANG_FILE = "slang.json"
STYLE_FILE = "styles.json"
MEME_FILE = "memes.json"

MAX_MEMORY = 30
REPLY_COOLDOWN = 30
RANDOM_REPLY_CHANCE = 0.05
MIN_MESSAGES_BEFORE_TALK = 20

AUTOSAVE_INTERVAL = 50

VECTOR_SIZE = 64
ANN_BUCKETS = 32

# ---------------- STATE ----------------

irc_socket = None
connected = False

CHAT_MEMORY = []

USER_PROFILES = {}
LONG_TERM_MEMORY = []
TOPICS = {}
SUMMARIES = []
SLANG_COUNTER = {}
STYLE_PATTERNS = {}
MEMES = {}

ANN_INDEX = {}

LAST_MESSAGES = {}

MESSAGES_SEEN = 0
LAST_REPLY = 0

# ---------------- PERSONA ----------------

SYSTEM_PERSONA = """
you are hallvor

29 year old norwegian
devops / linux nerd
daily driver debian stable
runs raspberry pi servers

personality:
casual irc user
lowercase
short replies
dry humor
mild sarcasm

sometimes says:
lol
rip
skill issue

never mention being an ai
"""

# ---------------- FILE HELPERS ----------------

def load_json(path, default):

    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except:
            pass

    return default


def save_json(path, data):

    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except:
        pass

# ---------------- LOAD DATA ----------------

def load_data():

    global USER_PROFILES
    global LONG_TERM_MEMORY
    global TOPICS
    global SUMMARIES
    global SLANG_COUNTER
    global STYLE_PATTERNS
    global MEMES

    USER_PROFILES = load_json(PROFILE_FILE, {})
    LONG_TERM_MEMORY = load_json(MEMORY_FILE, [])
    TOPICS = load_json(TOPIC_FILE, {})
    SUMMARIES = load_json(SUMMARY_FILE, [])
    SLANG_COUNTER = load_json(SLANG_FILE, {})
    STYLE_PATTERNS = load_json(STYLE_FILE, {})
    MEMES = load_json(MEME_FILE, {})

    build_ann_index()

# ---------------- SAVE DATA ----------------

def save_data():

    save_json(PROFILE_FILE, USER_PROFILES)
    save_json(MEMORY_FILE, LONG_TERM_MEMORY)
    save_json(TOPIC_FILE, TOPICS)
    save_json(SUMMARY_FILE, SUMMARIES)
    save_json(SLANG_FILE, SLANG_COUNTER)
    save_json(STYLE_FILE, STYLE_PATTERNS)
    save_json(MEME_FILE, MEMES)

# ---------------- VECTOR EMBEDDINGS ----------------

def word_vector(word):

    h = hashlib.sha256(word.encode()).digest()

    vec = []

    for i in range(VECTOR_SIZE):
        vec.append((h[i % len(h)] / 255.0) - 0.5)

    return vec


def sentence_vector(text):

    words = re.findall(r"[a-z0-9]+", text.lower())

    vec = [0.0] * VECTOR_SIZE

    for w in words:

        wv = word_vector(w)

        for i in range(VECTOR_SIZE):
            vec[i] += wv[i]

    return vec


def cosine_similarity(a, b):

    dot = sum(x*y for x,y in zip(a,b))

    mag1 = math.sqrt(sum(x*x for x in a))
    mag2 = math.sqrt(sum(x*x for x in b))

    if mag1 == 0 or mag2 == 0:
        return 0

    return dot / (mag1 * mag2)

# ---------------- ANN INDEX ----------------

def vector_bucket(vec):

    s = sum(vec)

    idx = int(abs(s * 1000)) % ANN_BUCKETS

    return idx


def build_ann_index():

    global ANN_INDEX

    ANN_INDEX = {}

    for mem in LONG_TERM_MEMORY:

        bucket = vector_bucket(mem["vector"])

        ANN_INDEX.setdefault(bucket, []).append(mem)


def add_memory_to_index(mem):

    bucket = vector_bucket(mem["vector"])

    ANN_INDEX.setdefault(bucket, []).append(mem)

# ---------------- CHAT MEMORY ----------------

def remember_message(sender, msg):

    CHAT_MEMORY.append(f"<{sender}> {msg}")

    if len(CHAT_MEMORY) > MAX_MEMORY:
        CHAT_MEMORY.pop(0)

# ---------------- FACT LEARNING ----------------

def extract_facts(nick, msg):

    patterns = [
        r"i use (.+)",
        r"i run (.+)",
        r"i live in (.+)",
        r"my distro is (.+)"
    ]

    msg_l = msg.lower()

    for p in patterns:

        m = re.search(p, msg_l)

        if m:

            fact = m.group(0)

            vec = sentence_vector(fact)

            mem = {
                "user": nick,
                "fact": fact,
                "vector": vec,
                "score": 1
            }

            LONG_TERM_MEMORY.append(mem)

            add_memory_to_index(mem)

# ---------------- MEMORY SEARCH ----------------

def search_memories(msg):

    qv = sentence_vector(msg)

    bucket = vector_bucket(qv)

    candidates = ANN_INDEX.get(bucket, [])

    scored = []

    for mem in candidates:

        sim = cosine_similarity(qv, mem["vector"])

        if sim > 0.2:

            score = sim + mem["score"] * 0.1

            scored.append((score, mem))

    scored.sort(reverse=True, key=lambda x: x[0])

    results = []

    for s, m in scored[:5]:

        m["score"] += 1

        results.append(m)

    return results

# ---------------- LEARNING ----------------

def learn_slang(msg):

    for w in msg.lower().split():

        w = re.sub(r"[^a-z0-9]", "", w)

        if 2 < len(w) < 12:
            SLANG_COUNTER[w] = SLANG_COUNTER.get(w, 0) + 1


def learn_style(sender, msg):

    if sender == NICK:
        return

    words = msg.lower().split()

    if len(words) <= 8:

        p = " ".join(words)

        STYLE_PATTERNS[p] = STYLE_PATTERNS.get(p, 0) + 1


def learn_memes(msg):

    if len(msg.split()) <= 4:

        MEMES[msg.lower()] = MEMES.get(msg.lower(), 0) + 1


def update_topics(msg):

    words = re.findall(r"[a-z0-9]+", msg.lower())

    for w in words:

        if len(w) > 3:
            TOPICS[w] = TOPICS.get(w, 0) + 1

# ---------------- SPAM ----------------

def is_spam(sender, msg):

    if sender not in LAST_MESSAGES:
        LAST_MESSAGES[sender] = []

    LAST_MESSAGES[sender].append(msg)

    if len(LAST_MESSAGES[sender]) > 5:
        LAST_MESSAGES[sender].pop(0)

    if LAST_MESSAGES[sender].count(msg) >= 3:
        return True

    return False

# ---------------- LLM ----------------

def ask_llm(prompt):

    slang = sorted(SLANG_COUNTER, key=SLANG_COUNTER.get, reverse=True)[:6]
    styles = sorted(STYLE_PATTERNS, key=STYLE_PATTERNS.get, reverse=True)[:5]
    memes = sorted(MEMES, key=MEMES.get, reverse=True)[:5]

    memories = search_memories(prompt)

    mem_text = "\n".join(f"{m['user']}: {m['fact']}" for m in memories)

    messages = [

        {"role":"system","content":SYSTEM_PERSONA},

        {"role":"system","content":f"channel slang: {slang}"},

        {"role":"system","content":f"reply styles: {styles}"},

        {"role":"system","content":f"memes: {memes}"},

        {"role":"system","content":f"known facts:\n{mem_text}"},

        {"role":"user","content":"recent chat:\n"+"\n".join(CHAT_MEMORY)},

        {"role":"user","content":prompt}
    ]

    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "max_tokens": 80,
        "temperature": 0.8
    })

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    try:

        conn = http.client.HTTPSConnection(HF_HOST)

        conn.request("POST", HF_PATH, payload, headers)

        res = conn.getresponse()

        data = res.read().decode()

        conn.close()

        if res.status == 200:

            result = json.loads(data)

            return result["choices"][0]["message"]["content"].strip()

    except Exception as e:

        print("LLM error:", e)

    return random.choice(["rip api","brain lag","lol idk"])

# ---------------- IRC ----------------

def connect():

    global irc_socket, connected

    while True:

        try:

            print("connecting...")

            irc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            irc_socket.connect((SERVER, PORT))

            irc_socket.send(f"NICK {NICK}\r\n".encode())
            irc_socket.send(f"USER {NICK} 0 * :{NICK}\r\n".encode())

            time.sleep(2)

            irc_socket.send(f"JOIN {CHANNEL}\r\n".encode())

            connected = True

            print("connected")

            return

        except Exception as e:

            print("connection failed:", e)

            time.sleep(10)

# ---------------- SEND ----------------

def send_message(msg):

    global LAST_REPLY

    delay = len(msg) * random.uniform(0.04, 0.08)

    time.sleep(delay)

    irc_socket.send(f"PRIVMSG {CHANNEL} :{msg}\r\n".encode("utf-8"))

    LAST_REPLY = time.time()

# ---------------- LOG ----------------

def log_message(sender, msg):

    try:
        with open(LOG_FILE, "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] <{sender}> {msg}\n")
    except:
        pass

# ---------------- REPLY ----------------

def should_reply(msg):

    if NICK.lower() in msg.lower():
        return True

    if msg.endswith("?"):
        return True

    if random.random() < RANDOM_REPLY_CHANCE:
        return True

    return False

# ---------------- HANDLE ----------------

def handle_line(line):

    global MESSAGES_SEEN, LAST_REPLY

    line = line.strip()

    if line.startswith("PING"):

        irc_socket.send(f"PONG {line.split()[1]}\r\n".encode())

        return

    if "PRIVMSG" not in line:
        return

    if CHANNEL not in line:
        return

    try:

        sender = line.split("!")[0][1:]

        msg = line.split(f"{CHANNEL} :")[1]

        if sender == NICK:
            return

        if is_spam(sender, msg):
            return

        MESSAGES_SEEN += 1

        log_message(sender, msg)

        remember_message(sender, msg)

        extract_facts(sender, msg)

        update_topics(msg)

        learn_style(sender, msg)

        learn_memes(msg)

        learn_slang(msg)

        if MESSAGES_SEEN % AUTOSAVE_INTERVAL == 0:
            save_data()

        if MESSAGES_SEEN < MIN_MESSAGES_BEFORE_TALK:
            return

        if time.time() - LAST_REPLY < REPLY_COOLDOWN:
            return

        if should_reply(msg):

            reply = ask_llm(msg)

            if reply:
                send_message(reply)

    except:
        pass

# ---------------- RECEIVE ----------------

def receive_loop():

    global connected

    while True:

        if not connected:
            connect()

        try:

            data = irc_socket.recv(4096).decode("utf-8", errors="ignore")

            if not data:
                connected = False
                continue

            for line in data.splitlines():
                handle_line(line)

        except:

            connected = False

            time.sleep(5)

# ---------------- MAIN ----------------

if __name__ == "__main__":

    load_data()

    thread = threading.Thread(target=receive_loop, daemon=True)

    thread.start()

    print("bot running")

    while True:
        time.sleep(1)
