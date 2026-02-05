from youtube_transcript_api import YouTubeTranscriptApi
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
import re
import yt_dlp
TS_RE = re.compile(
    r"(?m)^\s*(?P<ts>(?:\d{1,2}:)?\d{1,2}:\d{2})\s*[-–—]?\s*(?P<title>.+?)\s*$"
)

def ts_to_seconds(ts: str) -> int:
    parts = [int(x) for x in ts.split(":")]
    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    h, m, s = parts
    return h * 3600 + m * 60 + s

def seconds_to_ts(sec: int) -> str:
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

@tool
def get_youtube_chapters(video_id: str) -> list[dict]:
    """
    Get YouTube chapters (if present) or parse timestamp chapters from the video description.

    Input: video_id like 'b1Fo_M_tj6w'
    Output: list of {title, start_time, end_time} timestamps as strings.
    """
    video_id = video_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("video_id must be 11 characters like 'b1Fo_M_tj6w'")

    url = f"https://www.youtube.com/watch?v={video_id}"

    with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)

    duration = info.get("duration")  # seconds or None

    # 1) Real chapters from metadata
    if info.get("chapters"):
        out = []
        for c in info["chapters"]:
            st = int(c.get("start_time", 0) or 0)
            et = c.get("end_time")
            out.append({
                "title": (c.get("title") or "").strip(),
                "start_time": seconds_to_ts(st),
                "end_time": None if et is None else seconds_to_ts(int(et)),
            })
        return out

    # 2) Fallback: parse timestamps from description
    desc = info.get("description") or ""
    starts = []
    for m in TS_RE.finditer(desc):
        st_sec = ts_to_seconds(m.group("ts"))
        starts.append({"title": m.group("title").strip(), "start_sec": st_sec})

    starts.sort(key=lambda x: x["start_sec"])

    out = []
    for i, item in enumerate(starts):
        st = item["start_sec"]
        nxt = starts[i + 1]["start_sec"] if i + 1 < len(starts) else duration
        out.append({
            "title": item["title"],
            "start_time": seconds_to_ts(st),
            "end_time": None if nxt is None else seconds_to_ts(int(nxt)),
        })

    return out

@tool
def get_youtube_transcript(video_id: str) -> str:
    """Fetch the transcript of a YouTube video by video ID."""
    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id).to_raw_data()
    return " ".join([entry['text'] for entry in transcript])

def get_youtube_transcript_summary(transcript: str) -> str:
    """Summarize the transcript of a YouTube video in a structured format."""
    prompt = f"""
        Help me summarize this transcript.

        Output format (use headings):
        ## Summary (bullets)
        - ...

        ## Key quotes
        - "..."

        ## Key concepts
        - ...

        Transcript:
        {transcript}
        """
    llm = ChatOpenAI(model="gpt-4o-mini")
    summary = llm.invoke(prompt)
    return summary

# result =get_youtube_transcript("dQw4w9WgXcQ")
# summary = get_youtube_transcript_summary(result)
# print(summary.content) 
# Create agent with tool
llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(llm, [get_youtube_transcript, get_youtube_chapters])

# Test it
result = agent.invoke({
    "messages": [("user", "Get the transcript for video dQw4w9WgXcQ and summarize it")]
})

# print(result)