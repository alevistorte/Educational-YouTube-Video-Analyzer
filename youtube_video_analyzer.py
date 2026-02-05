import json
import os
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
import scrapetube


def extract_video_id(video_id_or_url: str) -> str:
    """Extract the video ID from a YouTube URL or return the ID as-is."""
    parsed = urlparse(video_id_or_url)
    if parsed.hostname in ("www.youtube.com", "youtube.com"):
        return parse_qs(parsed.query)["v"][0]
    if parsed.hostname == "youtu.be":
        return parsed.path.lstrip("/")
    return video_id_or_url


def search_youtube(query: str, limit: int = 5) -> list[dict]:
    """Search YouTube and return a list of video results."""
    videos = []
    for video in scrapetube.get_search(query, limit=limit):
        videos.append({
            "video_id": video["videoId"],
            "title": video["title"]["runs"][0]["text"],
            "channel": video.get("ownerText", {}).get("runs", [{}])[0].get("text", "Unknown"),
            "duration": video.get("lengthText", {}).get("simpleText", "N/A"),
        })
    return videos


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
    query = input("What topic do you want to learn about? ")
    print(f"\nSearching YouTube for: {query}\n")
    results = search_youtube(query)

    if not results:
        print("No videos found. Try a different search.")
        return

    for i, video in enumerate(results, 1):
        print(f"  {i}) {video['title']}")
        print(f"     Channel: {video['channel']}  |  Duration: {video['duration']}")

    choice = input(f"\nPick a video (1-{len(results)}): ").strip()
    if not choice.isdigit() or not 1 <= int(choice) <= len(results):
        print("Invalid choice.")
        return

    video_id = results[int(choice) - 1]["video_id"]
    print(f"\nSelected: {results[int(choice) - 1]['title']}")
    os.makedirs("files", exist_ok=True)
    cache_file = os.path.join("files", f'questions_summary_{video_id}.json')

    if os.path.exists(cache_file):
        print(f"Loading cached results from {cache_file}")
        with open(cache_file, "r") as f:
            output = json.load(f)
    else:
        print(
            f"Fetching transcript and generating summary/quiz for video ID: {video_id}")
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
            match = re.search(r"```(?:json)?\s*(.*?)```",
                              final_message, re.DOTALL)
            if match:
                output = json.loads(match.group(1).strip())
            else:
                output = {"raw_response": final_message}

        with open(cache_file, "w") as f:
            json.dump(output, f, indent=4)

    # Show summary
    print("\n" + "=" * 60)
    print("VIDEO SUMMARY")
    print("=" * 60)
    print(output.get("summary", "No summary available."))
    print("=" * 60)

    # Run interactive quiz
    quiz = output.get("quiz", [])
    if not quiz:
        print("No quiz questions were generated.")
        return

    print(f"\nQUIZ TIME! {len(quiz)} questions\n")
    score = 0
    for i, q in enumerate(quiz, 1):
        print(f"Question {i}: {q['question']}")
        for letter, text in q["options"].items():
            print(f"  {letter}) {text}")
        answer = input("Your answer (A/B/C/D): ").strip().upper()
        correct = q["answer"].upper()
        if answer == correct:
            print("Correct!\n")
            score += 1
        else:
            print(
                f"Wrong! The correct answer was {correct}) {q['options'][correct]}\n")

    print("=" * 60)
    print(f"FINAL SCORE: {score}/{len(quiz)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
