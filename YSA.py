import streamlit as st
from agno.agent import Agent
from agno.models.groq import Groq
import os
from dotenv import load_dotenv
import yt_dlp
import re
import requests
import traceback
import time
from datetime import datetime
import random

# Load API key
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

# ===== Helper Functions =====
def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|/embed/|/v/|shorts/)([A-Za-z0-9_-]{11})", url)
    if not match:
        raise ValueError("Couldn't extract a valid YouTube video ID.")
    return match.group(1)

def format_date(date_str):
    if not date_str:
        return "Unknown Date"
    try:
        return datetime.strptime(date_str, "%Y%m%d").strftime("%b %d, %Y")
    except:
        return date_str

def fetch_video_details(video_url: str):
    ydl_opts = {"skip_download": True, "quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
    return {
        "title": info.get("title", "Unknown Title"),
        "thumbnail": info.get("thumbnail", ""),
        "uploader": info.get("uploader", "Unknown Uploader"),
        "upload_date": info.get("upload_date", ""),
        "view_count": info.get("view_count", 0),
        "duration": info.get("duration", 0),
        "description": info.get("description", "")
    }

def fetch_transcript_yt_dlp(video_url: str) -> str:
    try:
        ydl_opts = {"skip_download": True, "quiet": True, "no_warnings": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
        subs = info.get("automatic_captions") or info.get("subtitles") or {}
        if not subs:
            return None
        lang_key = "en" if "en" in subs else next(iter(subs.keys()))
        entry = subs[lang_key][0]
        sub_url = entry.get("url")
        if not sub_url:
            return None
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(sub_url, headers=headers, timeout=15)
        r.raise_for_status()
        text = re.sub(r"<[^>]+>", " ", r.text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception:
        traceback.print_exc()
        return None

def chunk_text(text: str, max_words: int = 150):  # smaller chunks for token safety
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, cur, cur_len = [], [], 0
    for s in sentences:
        s_len = len(s.split())
        if cur_len + s_len > max_words and cur:
            chunks.append(" ".join(cur))
            cur, cur_len = [s], s_len
        else:
            cur.append(s)
            cur_len += s_len
    if cur:
        chunks.append(" ".join(cur))
    return chunks

# ===== AI Model =====
youtube_agent = Agent(
    name="YouTube Summarizer",
    role="Detailed and Chronological YouTube Summarizer",
    model=Groq(id="openai/gpt-oss-120b"),  # ONLY this model
    instructions=(
        "Create a detailed and accurate summary of the transcript, without skipping any events or facts. "
        "Include all key actions, dialogues, and information in chronological order exactly as they happen in the video. "
        "Do not add opinions, interpretations, or assumptions; only describe what is actually in the transcript."
    ),
    show_tool_calls=False,
    markdown=True,
)

def summarize_using_agent(chunks: list):
    summaries = []
    batch_size = 1  # smaller batches to prevent token overflow
    delay_seconds = 1.0

    # Step 1: Summarize each chunk
    for i, chunk in enumerate(chunks, start=1):
        prompt = (
            f"From the following transcript chunk, write a detailed chronological description "
            f"covering all events, actions, and statements exactly as they appear, without skipping any details:\n\n{chunk}"
        )
        out = youtube_agent.run(prompt)
        summaries.append(out.strip() if isinstance(out, str) else str(out))
        time.sleep(delay_seconds)

    # Step 2: Gradually merge summaries in multiple passes
    while len(summaries) > 1:
        merged_summaries = []
        for i in range(0, len(summaries), batch_size):
            batch = summaries[i:i+batch_size]
            batch_prompt = (
                "Combine the following detailed summaries into one chronological description, "
                "preserving all events and facts without removing any important points:\n\n"
                + "\n\n".join(batch)
            )
            combined = youtube_agent.run(batch_prompt)
            merged_summaries.append(combined.strip() if isinstance(combined, str) else str(combined))
            time.sleep(delay_seconds)
        summaries = merged_summaries

    return summaries[0] if summaries else ""

# ===== Streamlit UI =====
st.set_page_config(page_title="VidNexus — YouTube Summarizer", page_icon="🪐", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🪐 VidNexus")
    st.markdown("### 📌 About")
    st.write("VidNexus is your **AI-powered YouTube Summarizer**. Paste a link, get a detailed real-world summary — fast and easy.")
    
    st.markdown("---")
    st.markdown("### 🎯 How to Use")
    st.write("1. Paste a YouTube or Shorts URL\n2. Click **Summarize**\n3. Get a chronological, detailed summary!")
    
    st.markdown("---")
    st.markdown("### 🎲 Random Fun Fact")
    facts = [
        "🍌 Bananas are berries, but strawberries aren't.",
        "🦈 Sharks existed before trees.",
        "🗼 The Eiffel Tower can be 15 cm taller during summer.",
        "🐱 Some cats are allergic to humans.",
    ]
    if st.button("Get Fun Fact"):
        st.info(random.choice(facts))
    
    st.markdown("---")
    st.markdown("### 💬 Feedback")
    feedback = st.text_area("Your thoughts about VidNexus:")
    if st.button("Submit Feedback"):
        st.success("Thanks for your feedback! 🚀")

# --- Custom CSS ---
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #1f1c2c, #928dab);
        color: #f0f0f0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .title {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        color: #ccc;
    }
    .justified-text {
        text-align: justify;
    }
    .scrollable-summary {
        text-align: justify;
        max-height: 400px;
        overflow-y: auto;
        padding-right: 10px;
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Header ---
st.markdown("<div class='title'>🪐 VidNexus — Pro YouTube Summarizer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Paste a YouTube video URL and get a detailed chronological summary instantly!</div>", unsafe_allow_html=True)

# --- Main Input ---
video_url = st.text_input(
    "🎥 Paste YouTube URL (normal or Shorts):",
    placeholder="https://www.youtube.com/watch?v=... or https://youtube.com/shorts/...",
    key="video_input"
)

# --- Process Video ---
if st.button("⚡ Summarize Video", use_container_width=True):
    if not video_url.strip():
        st.warning("⚠️ Please enter a valid YouTube video URL.")
    else:
        try:
            start_time = time.time()
            video_id = extract_video_id(video_url)
            details = fetch_video_details(video_url)

            st.video(f"https://www.youtube.com/watch?v={video_id}")

            col1, col2, col3 = st.columns([3, 5, 3])
            with col1:
                if details["thumbnail"]:
                    st.image(details["thumbnail"], caption="Video Thumbnail", use_container_width=True)
            with col2:
                st.markdown(f"### {details['title']}")
                st.markdown(f"**Uploader:** {details['uploader']}")
                st.markdown(f"**Upload Date:** {format_date(details['upload_date'])}")
                st.markdown(f"**Views:** {details['view_count']:,}")
                duration_min, duration_sec = divmod(details['duration'], 60)
                st.markdown(f"**Duration:** {duration_min}m {duration_sec}s")
            with col3:
                st.markdown("### 📄 Description")
                desc = details['description'] if details['description'] else "No description available."
                st.markdown(f"<div class='justified-text'>{desc[:400] + ('...' if len(desc) > 400 else '')}</div>", unsafe_allow_html=True)

            st.markdown("---")

            transcript = fetch_transcript_yt_dlp(video_url)
            if not transcript:
                st.error("❌ Transcript unavailable; summarization not possible.")
                st.stop()

            st.info(f"📝 Transcript length: {len(transcript.split()):,} words")
            chunks = chunk_text(transcript, max_words=150)
            st.info(f"✂️ Transcript split into **{len(chunks)}** chunks for processing.")

            with st.spinner("🤖 Summarizing the transcript..."):
                summary = summarize_using_agent(chunks)

            st.success("✅ Summary ready!")
            st.markdown("## 📜 Final Summary")
            st.markdown(f"<div class='scrollable-summary'>{summary}</div>", unsafe_allow_html=True)

            st.download_button(
                label="💾 Download Summary as TXT",
                data=summary,
                file_name=f"{details['title'][:50]}_summary.txt",
                mime="text/plain",
            )

            st.markdown(f"⏳ Process completed in **{time.time() - start_time:.2f} seconds**")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

st.markdown("<div class='footer'>Made with ❤️ by VidNexus</div>", unsafe_allow_html=True)
