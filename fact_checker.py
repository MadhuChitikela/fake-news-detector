import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from tavily import TavilyClient

load_dotenv()

# ── Initialize clients ───────────────────────────────────────────
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

ALL_MODELS = [
    {"provider": "groq",   "model": "gemma2-9b-it"},
    {"provider": "groq",   "model": "llama-3.1-8b-instant"},
    {"provider": "groq",   "model": "mixtral-8x7b-32768"},
    {"provider": "gemini", "model": "gemini-2.0-flash"},
    {"provider": "gemini", "model": "gemini-1.5-flash"},
]


def get_llm():
    """Get first working LLM"""
    for entry in ALL_MODELS:
        try:
            if entry["provider"] == "groq":
                llm = ChatGroq(
                    model=entry["model"],
                    groq_api_key=os.getenv("GROQ_API_KEY"),
                    temperature=0
                )
            else:
                llm = ChatGoogleGenerativeAI(
                    model=entry["model"],
                    google_api_key=os.getenv("GEMINI_API_KEY"),
                    temperature=0
                )
            llm.invoke("OK")
            print(f"✅ Using [{entry['provider']}] {entry['model']}")
            return llm
        except Exception as e:
            if "429" in str(e) or "quota" in str(e):
                continue
    return None


def extract_claims(text: str, llm) -> list:
    """Extract top 3 key claims from article"""
    system = SystemMessage(content="""You are a fact-checking assistant.
Extract exactly 3 key factual claims from the given text.
Return ONLY a numbered list like:
1. claim one
2. claim two
3. claim three
Nothing else.""")

    user = HumanMessage(content=f"Extract 3 key claims from:\n\n{text[:1000]}")

    try:
        response = llm.invoke([system, user])
        lines = response.content.strip().split("\n")
        claims = []
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                claim = line.lstrip("0123456789.-) ").strip()
                if claim:
                    claims.append(claim)
        return claims[:3]
    except:
        return ["Unable to extract claims"]


def search_claim(claim: str) -> dict:
    """Search web for evidence about a claim"""
    try:
        results = tavily.search(
            query=claim,
            max_results=3
        )
        sources = []
        for r in results.get("results", []):
            sources.append({
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "content": r.get("content", "")[:300]
            })
        return {"claim": claim, "sources": sources, "found": len(sources) > 0}
    except Exception as e:
        return {"claim": claim, "sources": [], "found": False, "error": str(e)}


def verify_claim(claim: str, sources: list, llm) -> dict:
    """Use LLM to verify claim against found sources"""
    if not sources:
        return {"verdict": "UNVERIFIED", "explanation": "No sources found"}

    sources_text = "\n".join([
        f"Source {i+1}: {s['title']}\n{s['content']}"
        for i, s in enumerate(sources)
    ])

    system = SystemMessage(content="""You are a fact checker.
Given a claim and sources, respond with ONLY:
VERDICT: SUPPORTED or CONTRADICTED or UNVERIFIED
REASON: one sentence explanation""")

    user = HumanMessage(content=f"""
CLAIM: {claim}

SOURCES:
{sources_text}
""")

    try:
        response = llm.invoke([system, user])
        lines = response.content.strip().split("\n")
        verdict = "UNVERIFIED"
        reason = "Could not determine"

        for line in lines:
            if line.startswith("VERDICT:"):
                verdict = line.replace("VERDICT:", "").strip()
            if line.startswith("REASON:"):
                reason = line.replace("REASON:", "").strip()

        return {"verdict": verdict, "explanation": reason}
    except:
        return {"verdict": "UNVERIFIED", "explanation": "Verification failed"}


def fact_check_article(text: str) -> dict:
    """
    Full fact checking pipeline:
    1. Extract claims
    2. Search web for each claim
    3. Verify each claim with LLM
    4. Return overall verdict
    """
    print("🔍 Starting fact check...")
    llm = get_llm()

    if llm is None:
        return {
            "claims":          [],
            "verified_count":  0,
            "supported_count": 0,
            "overall_verdict": "UNAVAILABLE",
            "explanation":     "LLM rate limited. Try again later."
        }

    # Step 1: Extract claims
    print("📋 Extracting claims...")
    claims = extract_claims(text, llm)

    # Step 2 + 3: Search and verify each claim
    results = []
    supported = 0

    for claim in claims:
        print(f"🔍 Checking: {claim[:50]}...")
        search_result = search_claim(claim)
        verification = verify_claim(claim, search_result["sources"], llm)

        if "SUPPORTED" in verification["verdict"]:
            supported += 1

        results.append({
            "claim":       claim,
            "sources":     search_result["sources"],
            "verdict":     verification["verdict"],
            "explanation": verification["explanation"]
        })

    # Overall verdict
    total = len(claims)
    support_ratio = supported / total if total > 0 else 0

    if support_ratio >= 0.7:
        overall = "LIKELY REAL"
    elif support_ratio >= 0.4:
        overall = "UNCERTAIN"
    else:
        overall = "LIKELY FAKE"

    return {
        "claims":          results,
        "verified_count":  total,
        "supported_count": supported,
        "overall_verdict": overall,
        "support_ratio":   round(support_ratio * 100, 1)
    }


# ── Test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    test = """
    NASA announces plans to send humans to Mars by 2035.
    The mission will cost approximately $500 billion.
    Scientists say the journey will take 7 months.
    """
    result = fact_check_article(test)
    print("\n📊 Fact Check Results:")
    print(f"Overall: {result['overall_verdict']}")
    print(f"Supported: {result['supported_count']}/{result['verified_count']} claims")
    for r in result['claims']:
        print(f"\n• {r['claim'][:60]}")
        print(f"  Verdict: {r['verdict']}")
        print(f"  Reason: {r['explanation']}")
