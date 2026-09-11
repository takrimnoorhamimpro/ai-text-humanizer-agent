import time
from google.adk.agents import Agent, LoopAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.genai import types
from .tools import check_sapling_ai_score, load_prompt_guidelines, exit_loop

# ==========================================
# PROACTIVE PACING (avoids 429s instead of just recovering from them)
# ==========================================
# Gemini's free tier limits by RPM (requests/minute), not TPM like Groq.
# Public docs generally show ~15 RPM for Flash models, but this account
# showed as low as 5 RPM on a similar Gemini model earlier in this project
# (see the "gemini-3.6-flash ... limit: 5" error from before). Pacing at
# 5 RPM is the safe assumption -- if the account actually has more headroom,
# this just makes the run a bit slower than strictly necessary, which is
# fine since a slower run was explicitly okay.
_MIN_SECONDS_BETWEEN_CALLS = 13  # ~4.6 calls/min, safely under a 5 RPM ceiling
_last_call_at = {"t": 0.0}


def pace_llm_calls(callback_context, llm_request):
    """before_model_callback: sleeps just long enough before each LLM call
    to keep the pipeline under Gemini's RPM ceiling. Returning None lets
    the call proceed normally afterward (no response is injected)."""
    now = time.monotonic()
    elapsed = now - _last_call_at["t"]
    if elapsed < _MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(_MIN_SECONDS_BETWEEN_CALLS - elapsed)
    _last_call_at["t"] = time.monotonic()
    return None


# A factory (not a single shared instance) so each agent gets its own Gemini
# adapter -- avoids any shared-state surprises between agents that call
# concurrently. Retries on 429 (rate limit), 500/503/504 (transient server
# errors like the "high demand" 503 seen from this project) with exponential
# backoff, instead of crashing the whole run on a momentary Google-side
# hiccup. Requires GOOGLE_API_KEY (or GEMINI_API_KEY) in .env, already set
# up earlier in this project.
def make_model():
    return Gemini(
        model="gemini-3.6-flash",
        retry_options=types.HttpRetryOptions(
            attempts=5,
            initial_delay=2,
            exp_base=2,
            http_status_codes=[429, 500, 503, 504],
        ),
    )


# ==========================================
# 1. BASELINE ANALYZER SUB-AGENT
# ==========================================
# Runs once, so it's fine for this one to see full history (there isn't any
# yet). It loads BOTH the AI score AND the ~300-line rewrite_rules.md
# guidelines into session state here, a single time, so nothing downstream
# has to re-load or re-send that file on every loop iteration.
baseline_agent = Agent(
    name="baseline_analyzer",
    model=make_model(),
    description="Analyzes the original user text and records initial AI score + guidelines in state.",
    instruction="""
    You are the Baseline Analyzer.
    1. Read the user's input text.
    2. Call `check_sapling_ai_score` with the exact input text.
       This stores the result in both `current_score` and `initial_score`,
       and caches the input itself as `original_text` in state.
    3. Call `load_prompt_guidelines` once. This caches the rewriting
       guidelines as `guidelines` in state so later agents can reference
       them directly without re-loading the file.
    4. State clearly: the initial AI detection score. Do not repeat the
       full guidelines or the full input text back -- they're now in state.
    """,
    tools=[check_sapling_ai_score, load_prompt_guidelines],
    before_model_callback=pace_llm_calls,
    output_key="initial_analysis"
)

# ==========================================
# 2. REWRITER SUB-AGENT (Inside Loop)
# ==========================================
# include_contents='none': this agent does NOT receive the accumulating
# conversation history (which would otherwise grow every loop iteration --
# every past draft, every past tool call/result). Instead it reads only
# the compact state placeholders below, keeping token usage roughly flat
# across iterations instead of growing linearly.
rewriter_agent = Agent(
    name="academic_rewriter",
    model=make_model(),
    description="Rewrites text using cached prompt guidelines to humanize tone.",
    include_contents="none",
    instruction="""
    You are an expert Academic Rewriter.

    Guidelines to follow:
    {guidelines}

    Original input text:
    {original_text}

    Current draft (if this is a later iteration, improve THIS instead of
    the original text; if empty, this is the first iteration -- rewrite
    the original text instead):
    {current_draft?}

    Feedback from the last review (if any -- you MUST address every single
    point raised here. Do not just swap synonyms -- restructure the actual
    sentence or cut the offending phrase entirely):
    {critic_feedback?}

    Extra rule: avoid generic "AI creative writing" imagery and phrasing --
    things like "neon," "rain-slick streets," "whispered its secret," "the
    unknown," or any sentence built as a poetic abstract metaphor. If the
    text leans this way, rewrite it into something plainer and more
    concrete instead of just rephrasing the same metaphor.

    Controlled imperfection (apply this even if the draft already looks
    clean -- overly smooth, evenly-paced prose is itself a signal that gets
    flagged, even with no cliché phrases left in it):
    - Let at least one sentence run on a bit too long, or cut one short
      into a fragment on purpose.
    - Use a comma splice or a dash instead of a "proper" conjunction at
      least once.
    - Drop in one small aside or self-correction, like "actually, more
      like—" or "or something close to that."
    - Don't fix every awkward transition. One slightly rough handoff
      between sentences is fine.
    - Vary capitalization/punctuation slightly where a real person
      typing quickly would (e.g. a missing comma before "and" once).
    Do this sparingly -- one or two spots per paragraph, not every
    sentence. The text should still read clearly, just not machine-smooth.

    Task:
    - Alter sentence structures (burstiness).
    - Eliminate repetitive robotic patterns (perplexity).
    - STRICTLY PRESERVE all facts, citations, dates, and core semantic meaning.
    - Output ONLY the newly rewritten text, nothing else.
    """,
    tools=[],
    before_model_callback=pace_llm_calls,
    output_key="current_draft"
)

# ==========================================
# 3. CRITIC / AUDITOR SUB-AGENT (Inside Loop)
# ==========================================
# Also include_contents='none' for the same reason -- it only needs the
# current draft, which is already available via the {current_draft} state
# placeholder, not the full running history.
critic_agent = Agent(
    name="quality_critic",
    model=make_model(),
    description="Evaluates AI detection score and breaks the loop if score <= 20.0%.",
    include_contents="none",
    instruction="""
    You are the Rigorous Quality Auditor.

    Text to evaluate:
    {current_draft}

    1. Call `check_sapling_ai_score` on this text.
    2. CHECK THE CRITERIA:
       - IF the score is <= 20.0%: Immediately call the `exit_loop` tool to terminate the loop!
       - IF the score is > 20.0%: Output brief, specific feedback (a few
         bullet points max) on which phrases feel AI-generated so the next
         iteration can improve. Keep it short -- this becomes the next
         rewriter's input.
    """,
    tools=[check_sapling_ai_score, exit_loop],
    before_model_callback=pace_llm_calls,
    output_key="critic_feedback"
)

# ==========================================
# 4. REFINEMENT LOOP AGENT (ADK Loop Primitive)
# ==========================================
# Executes Rewriter -> Critic repeatedly until exit_loop is triggered or max_iterations is reached
refinement_loop = LoopAgent(
    name="refinement_loop",
    sub_agents=[rewriter_agent, critic_agent],
    max_iterations=5
)

# ==========================================
# 5. FINAL REPORTER SUB-AGENT
# ==========================================
# Also stateless w.r.t. history -- it only needs the final aggregated
# state values, all available as placeholders below.
reporter_agent = Agent(
    name="final_reporter",
    model=make_model(),
    description="Formats the final research comparison and final humanized text.",
    include_contents="none",
    instruction="""
    You are the Final Research Reporter.
    Use the following state values directly:
    - Initial score: {initial_score}
    - Final score: {current_score}
    - Target reached flag: {target_reached?}
    - Final text: {current_draft}

    If `target_reached` is true, Status is "Target Achieved (<20%)".
    Otherwise, Status is "Max Iterations Reached".

    Format your final response strictly as follows:

    # 📝 AI DETECTION OPTIMIZATION REPORT
    ### 📊 Benchmark Metrics:
    - **Initial AI Detection Score:** {initial_score}%
    - **Final AI Detection Score:** {current_score}%
    - **Status:** [Target Achieved (<20%) OR Max Iterations Reached]

    ---
    # 📄 MODIFIED HUMANIZED TEXT
    [Display the final rewritten text cleanly, from {current_draft}]
    """,
    before_model_callback=pace_llm_calls,
)

# ==========================================
# 6. ROOT SEQUENTIAL WORKFLOW AGENT
# ==========================================
# Executes: Baseline Analysis -> Refinement Loop -> Final Report
root_agent = SequentialAgent(
    name="ai_mitigation_orchestrator",
    description="End-to-end multi-agent system for iterative AI text mitigation.",
    sub_agents=[baseline_agent, refinement_loop, reporter_agent]
)