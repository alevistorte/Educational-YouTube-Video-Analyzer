import json
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent


def extract_video_id(video_id_or_url: str) -> str:
    """Extract the video ID from a YouTube URL or return the ID as-is."""
    parsed = urlparse(video_id_or_url)
    if parsed.hostname in ("www.youtube.com", "youtube.com"):
        return parse_qs(parsed.query)["v"][0]
    if parsed.hostname == "youtu.be":
        return parsed.path.lstrip("/")
    return video_id_or_url


@tool
def get_youtube_transcript(video_id_or_url: str) -> str:
    """Fetch the transcript of a YouTube video by video ID or URL."""
    video_id = extract_video_id(video_id_or_url)
    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id)
    return " ".join([snippet.text for snippet in transcript.snippets])


# Create agent with tool
llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_agent(llm, [get_youtube_transcript])


def main():
    # video_id = input("Enter the YouTube video ID: ")
    # video_id = "dQw4w9WgXcQ"  # Example video ID
    video_id = "aircAruvnKk"  # Example video ID with transcript
    prompt = (
        f"Get the transcript for video {video_id}, then:\n"
        "1. Summarize the key points of the video.\n"
        "2. Generate a quiz with 5 multiple-choice questions to test understanding "
        "of the video content. Each question should have 4 options (A, B, C, D) "
        "with one correct answer.\n\n"
        "Return the result as valid JSON with this structure:\n"
        '{"summary": "...", "quiz": [{"question": "...", "options": {"A": "...", '
        '"B": "...", "C": "...", "D": "..."}, "answer": "A"}, ...]}'
    )
    result = agent.invoke({
        "messages": [("user", prompt)]
    })
    # Extract the final assistant message
    final_message = result["messages"][-1].content

    # Try to parse structured JSON from the response
    try:
        output = json.loads(final_message)
    except json.JSONDecodeError:
        # If the model wrapped the JSON in markdown code fences, extract it
        import re
        match = re.search(r"```(?:json)?\s*(.*?)```", final_message, re.DOTALL)
        if match:
            output = json.loads(match.group(1).strip())
        else:
            output = {"raw_response": final_message}

    with open(f'questions_summary_{video_id}.json', "w") as f:
        json.dump(output, f, indent=4)
    print(
        f"Transcript summary and quiz saved to questions_summary_{video_id}.json")


if __name__ == "__main__":
    main()
