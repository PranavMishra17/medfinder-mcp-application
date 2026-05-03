#!/usr/bin/env python3
"""
Medfinder MCP Application Agent

Walks the 5-step MCP application pipeline autonomously.
Reads all config from .env (see .env.example).
Uses Claude to reason about the screening question from candidate context.
Falls back to terminal input() for anything it can't answer confidently.

Setup:
    pip install anthropic requests python-dotenv
    cp .env.example .env
    # fill in .env, then:
    python medfinder_agent.py
"""

import os
import sys
import json
import re
import logging
import getpass
from pathlib import Path

# ── dotenv — must load before anything reads os.getenv ────────────────────────
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=False)
        print(f"[dotenv] Loaded config from {_env_path}")
    else:
        print(f"[dotenv] No .env file found at {_env_path} — will use shell env vars or prompt.")
except ImportError:
    print("[dotenv] python-dotenv not installed — falling back to shell env vars.")
    print("         Run: pip install python-dotenv")

import requests
from anthropic import Anthropic, APIError, APIConnectionError, APITimeoutError

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-7s]  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("medfinder-agent")

# ── constants ─────────────────────────────────────────────────────────────────
MCP_URL               = "https://hatch-one.vercel.app/api/mcp"
AGENT_NAME            = "Medfinder MCP Agent"
AGENT_VENDOR          = "Anthropic"
MAX_SCREENING_RETRIES = 5


# ════════════════════════════════════════════════════════════════════════════
#  ENV LOADING HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_required(key: str, label: str, secret: bool = False) -> str:
    """Return env var value. If missing, prompt the user interactively."""
    val = _env(key)
    if val:
        masked = ("*" * 8) if secret else val
        log.info(f"    {key:<35} → {masked}")
        return val
    log.warning(f"    {key:<35} → NOT SET — prompting.")
    if secret:
        val = getpass.getpass(f"  Enter {label}: ").strip()
    else:
        val = input(f"  Enter {label}: ").strip()
    if not val:
        log.error(f"No value provided for '{label}'. Cannot continue.")
        sys.exit(1)
    return val


def _parse_pipe_list(raw: str) -> list[list[str]]:
    """
    Parse ';;'-delimited entries where fields within each entry are '|'-separated.
    E.g. "Title|Venue|Year|URL;;Title2|Venue2|Year2|URL2"
    Returns list of lists.
    """
    if not raw:
        return []
    entries = []
    for chunk in raw.split(";;"):
        chunk = chunk.strip()
        if chunk:
            entries.append([p.strip() for p in chunk.split("|")])
    return entries


# ════════════════════════════════════════════════════════════════════════════
#  CANDIDATE CONFIG
# ════════════════════════════════════════════════════════════════════════════

def load_candidate_config() -> dict:
    """
    Read every CANDIDATE_* and agent env var.
    Logs each field so you can verify exactly what was picked up from .env.
    """
    log.info("Loading candidate config from .env ────────────────────────────")

    cfg = {
        # ── submitted directly to MCP steps ──────────────────────────────────
        "name":             _env_required("CANDIDATE_NAME",    "Full name"),
        "email":            _env_required("CANDIDATE_EMAIL",   "Email address"),
        "github":           _env_required("CANDIDATE_GITHUB",  "GitHub profile URL"),
        "linkedin":         _env_required("CANDIDATE_LINKEDIN","LinkedIn profile URL"),
        "resume_url":       _env_required("CANDIDATE_RESUME_URL",  "Resume URL (public PDF)"),
        "resume_mime_type": _env("CANDIDATE_RESUME_MIME_TYPE", "application/pdf"),
        "project_url":      _env_required("CANDIDATE_PROJECT_URL", "Primary agentic project URL"),

        # ── bio context — read by Claude when reasoning ───────────────────────
        "phone":                _env("CANDIDATE_PHONE"),
        "location":             _env("CANDIDATE_LOCATION"),
        "portfolio":            _env("CANDIDATE_PORTFOLIO"),
        "current_role":         _env("CANDIDATE_CURRENT_ROLE"),
        "current_company":      _env("CANDIDATE_CURRENT_COMPANY"),
        "current_role_summary": _env("CANDIDATE_CURRENT_ROLE_SUMMARY"),
        "education":            _env("CANDIDATE_EDUCATION"),
        "skills":               _env("CANDIDATE_SKILLS"),
        "voice_ai_summary":     _env("CANDIDATE_VOICE_AI_SUMMARY"),
        "pitch":                _env("CANDIDATE_PITCH"),

        # ── parsed list fields ────────────────────────────────────────────────
        "papers":          _parse_pipe_list(_env("CANDIDATE_PAPERS")),
        "projects_extra":  _parse_pipe_list(_env("CANDIDATE_PROJECTS_EXTRA")),
        "achievements":    _parse_pipe_list(_env("CANDIDATE_ACHIEVEMENTS")),

        # ── agent config ──────────────────────────────────────────────────────
        "anthropic_key": _env_required("ANTHROPIC_API_KEY", "Anthropic API key", secret=True),
        "model":         _env("CLAUDE_MODEL", "claude-sonnet-4-6"),
        "dry_run":       _env("DRY_RUN", "true").lower() not in ("false", "0", "no"),
    }

    log.info(f"    {'CLAUDE_MODEL':<35} → {cfg['model']}")
    log.info(f"    {'DRY_RUN':<35} → {cfg['dry_run']}")
    log.info(f"    {'CANDIDATE_PAPERS (count)':<35} → {len(cfg['papers'])}")
    log.info(f"    {'CANDIDATE_PROJECTS_EXTRA (count)':<35} → {len(cfg['projects_extra'])}")
    log.info(f"    {'CANDIDATE_ACHIEVEMENTS (count)':<35} → {len(cfg['achievements'])}")
    log.info("Config loaded ──────────────────────────────────────────────────")
    return cfg


def build_candidate_bio(cfg: dict) -> str:
    """
    Assemble a rich plaintext bio from config.
    Claude reads this when reasoning about the screening question.
    """
    lines = ["CANDIDATE PROFILE", "=" * 60]

    lines += [f"Name:     {cfg['name']}", f"Email:    {cfg['email']}"]
    if cfg["phone"]:    lines.append(f"Phone:    {cfg['phone']}")
    if cfg["location"]: lines.append(f"Location: {cfg['location']}")

    lines += ["", "LINKS", f"  GitHub:    {cfg['github']}", f"  LinkedIn:  {cfg['linkedin']}"]
    if cfg["portfolio"]: lines.append(f"  Portfolio: {cfg['portfolio']}")

    if cfg["education"]:
        lines += ["", "EDUCATION", f"  {cfg['education']}"]

    if cfg["current_role"] or cfg["current_company"]:
        lines += ["", "CURRENT ROLE"]
        role_str = " at ".join(filter(None, [cfg["current_role"], cfg["current_company"]]))
        lines.append(f"  {role_str}")
        if cfg["current_role_summary"]:
            lines.append(f"  {cfg['current_role_summary']}")

    if cfg["papers"]:
        lines += ["", "RESEARCH PUBLICATIONS"]
        for p in cfg["papers"]:
            title = p[0] if len(p) > 0 else ""
            venue = p[1] if len(p) > 1 else ""
            year  = p[2] if len(p) > 2 else ""
            url   = p[3] if len(p) > 3 else ""
            entry = f"  - {title}"
            if venue or year: entry += f"  [{(venue + ' ' + year).strip()}]"
            if url:           entry += f"  {url}"
            lines.append(entry)

    if cfg["voice_ai_summary"]:
        lines += ["", "VOICE AI EXPERIENCE (directly relevant to this role)"]
        lines.append(f"  {cfg['voice_ai_summary']}")

    lines += ["", "PRIMARY AGENTIC PROJECT", f"  {cfg['project_url']}"]

    if cfg["projects_extra"]:
        lines += ["", "ADDITIONAL PROJECTS"]
        for p in cfg["projects_extra"]:
            name = p[0] if len(p) > 0 else ""
            url  = p[1] if len(p) > 1 else ""
            desc = p[2] if len(p) > 2 else ""
            lines.append(f"  - {name}: {desc}  {url}".strip())

    if cfg["achievements"]:
        lines += ["", "HACKATHONS & ACHIEVEMENTS"]
        for a in cfg["achievements"]:
            event   = a[0] if len(a) > 0 else ""
            result  = a[1] if len(a) > 1 else ""
            project = a[2] if len(a) > 2 else ""
            entry = f"  - {event}"
            if project: entry += f": {project}"
            if result:  entry += f" — {result}"
            lines.append(entry)

    if cfg["skills"]:
        lines += ["", "SKILLS", f"  {cfg['skills']}"]

    if cfg["pitch"]:
        lines += ["", "CANDIDATE PITCH", f"  {cfg['pitch']}"]

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
#  MCP JSON-RPC CLIENT
# ════════════════════════════════════════════════════════════════════════════

_rpc_id = 0


def _next_id() -> int:
    global _rpc_id
    _rpc_id += 1
    return _rpc_id


def call_mcp(method: str, params: dict, timeout: int = 30) -> dict:
    """POST a JSON-RPC 2.0 message. Returns result dict or raises."""
    payload = {
        "jsonrpc": "2.0",
        "id":      _next_id(),
        "method":  method,
        "params":  params,
    }
    log.info(f"  → RPC  method={method}  params={list(params.keys())}")
    log.debug(f"  → Payload: {json.dumps(payload)}")

    try:
        resp = requests.post(MCP_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        log.info(f"  ← HTTP {resp.status_code}  elapsed={resp.elapsed.total_seconds():.2f}s")
    except requests.Timeout:
        log.error(f"  ✗ Timed out after {timeout}s")
        raise
    except requests.ConnectionError as e:
        log.error(f"  ✗ Connection error: {e}")
        raise
    except requests.HTTPError as e:
        log.error(f"  ✗ HTTP {resp.status_code}: {resp.text[:400]}")
        raise

    try:
        data = resp.json()
    except json.JSONDecodeError:
        log.error(f"  ✗ Response is not JSON: {resp.text[:300]}")
        raise

    log.debug(f"  ← Raw: {json.dumps(data)}")

    if data.get("error"):
        log.error(f"  ✗ RPC error: {data['error']}")
        raise RuntimeError(f"RPC error: {data['error']}")

    result = data.get("result", {})
    log.info(f"  ← OK  method={method}")
    return result


def tool_call(tool_name: str, arguments: dict) -> dict:
    log.info(f"Calling tool: {tool_name}")
    return call_mcp("tools/call", {"name": tool_name, "arguments": arguments})


def extract_text(mcp_result: dict) -> str:
    content = mcp_result.get("content", [])
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content if b.get("type") == "text"]
        return "\n".join(p for p in parts if p).strip()
    if isinstance(content, str):
        return content.strip()
    return json.dumps(mcp_result)


def find_draft_id(result: dict, text: str) -> str | None:
    # 1. structuredContent — this is where the Hatch MCP server actually puts it
    structured = result.get("structuredContent", {})
    if isinstance(structured, dict):
        for key in ("draft_id", "draftId", "id", "draft"):
            val = structured.get(key)
            if isinstance(val, str) and val:
                log.info(f"  draft_id found at structuredContent['{key}'] = {val}")
                return val

    # 2. result root
    for key in ("draft_id", "draftId", "id", "draft"):
        val = result.get(key)
        if isinstance(val, str) and val:
            log.info(f"  draft_id found at result['{key}'] = {val}")
            return val

    # 3. content blocks (text blocks may carry JSON with draft_id)
    for block in result.get("content", []):
        for key in ("draft_id", "draftId", "id"):
            val = block.get(key)
            if isinstance(val, str) and val:
                log.info(f"  draft_id found in content block['{key}'] = {val}")
                return val
        # also try parsing the text inside a text block
        if block.get("type") == "text":
            try:
                inner = json.loads(block.get("text", ""))
                for key in ("draft_id", "draftId", "id"):
                    val = inner.get(key)
                    if isinstance(val, str) and val:
                        log.info(f"  draft_id parsed from content[text JSON]['{key}'] = {val}")
                        return val
            except (json.JSONDecodeError, AttributeError):
                pass

    # 4. last resort: regex over the full text blob
    match = re.search(r'"?draft_id"?\s*[:\s]+([a-zA-Z0-9_\-]{6,})', text, re.IGNORECASE)
    if match:
        found = match.group(1)
        log.info(f"  draft_id parsed from response text via regex: {found}")
        return found

    return None


# ════════════════════════════════════════════════════════════════════════════
#  MCP DISCOVERY
# ════════════════════════════════════════════════════════════════════════════

def discover_tools() -> dict:
    """
    Run the proper MCP handshake (initialize → tools/list).
    Logs the exact parameter names for every tool so you can catch
    field-name mismatches before any submission step runs.
    """
    log.info("MCP handshake: initialize...")
    try:
        call_mcp("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities":    {},
            "clientInfo":      {"name": "medfinder-agent", "version": "1.0.0"},
        })
    except Exception as e:
        log.warning(f"  initialize failed (server may not require it): {e}")

    log.info("MCP handshake: tools/list...")
    try:
        result    = call_mcp("tools/list", {})
        tools_raw = result.get("tools", [])
        schema_map = {t["name"]: t for t in tools_raw if isinstance(t, dict)}

        log.info(f"  {len(schema_map)} tools discovered:")
        for tname, schema in schema_map.items():
            props    = schema.get("inputSchema", {}).get("properties", {})
            required = schema.get("inputSchema", {}).get("required", [])
            log.info(f"    {tname}")
            for pname, pschema in props.items():
                req = " ← required" if pname in required else ""
                log.info(f"      · {pname}: {pschema.get('type', 'any')}{req}")
        return schema_map
    except Exception as e:
        log.warning(f"  tools/list failed — proceeding with assumed field names: {e}")
        return {}


# ════════════════════════════════════════════════════════════════════════════
#  CLAUDE REASONING
# ════════════════════════════════════════════════════════════════════════════

def reason_about_screening(
    overview_text: str,
    bio: str,
    client: Anthropic,
    model: str,
) -> dict:
    """
    Ask Claude to identify the screening question and decide if it can
    answer from the candidate bio. Returns structured dict.
    """
    prompt = f"""You are an application agent acting on behalf of this candidate.

<candidate_bio>
{bio}
</candidate_bio>

The MCP server returned this role overview, which contains a screening question somewhere in it:
<role_overview>
{overview_text}
</role_overview>

Your tasks:
1. Find and extract the exact screening question.
2. Decide if you can answer it confidently using only the candidate bio.
   - Factual question about the candidate (experience, projects, skills, motivations) → can answer.
   - Riddle, puzzle, trivia, cipher, or anything not answerable from the bio → cannot answer.
3. Return ONLY valid JSON. No preamble. No markdown fences.

Required schema:
{{
  "screening_question": "the exact question text",
  "can_answer": true or false,
  "answer": "full answer if can_answer is true, else empty string",
  "reasoning": "one sentence explaining your decision"
}}"""

    log.info("Sending screening question to Claude...")
    log.info(f"  model: {model} | bio: {len(bio)} chars | overview: {len(overview_text)} chars")

    raw = ""
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        log.debug(f"  Claude raw: {raw[:500]}")

        raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\n?```$",       "", raw, flags=re.MULTILINE)
        raw = raw.strip()

        parsed = json.loads(raw)
        log.info(f"  Identified question : {parsed.get('screening_question', '?')}")
        log.info(f"  can_answer          : {parsed.get('can_answer')}")
        log.info(f"  Reasoning           : {parsed.get('reasoning')}")
        if parsed.get("can_answer"):
            log.info(f"  Proposed answer     : {str(parsed.get('answer', ''))[:200]}")
        return parsed

    except json.JSONDecodeError as e:
        log.error(f"  Claude returned non-JSON: {e} | raw: {raw[:400]}")
        return {"screening_question": "unknown", "can_answer": False, "answer": "", "reasoning": "JSON parse error"}
    except (APIError, APIConnectionError, APITimeoutError) as e:
        log.error(f"  Claude API error: {e}")
        return {"screening_question": "unknown", "can_answer": False, "answer": "", "reasoning": str(e)}


# ════════════════════════════════════════════════════════════════════════════
#  TERMINAL CLARIFICATION
# ════════════════════════════════════════════════════════════════════════════

def terminal_clarify(label: str, context: str) -> str:
    print(f"\n{'─' * 64}")
    print(f"  AGENT NEEDS YOUR INPUT  →  {label}")
    print(f"{'─' * 64}")
    print(context)
    print(f"{'─' * 64}")
    answer = input("  Your answer: ").strip()
    if not answer:
        log.warning("Empty answer — re-prompting.")
        return terminal_clarify(label, context)
    log.info(f"  Human provided answer ({len(answer)} chars)")
    return answer


# ════════════════════════════════════════════════════════════════════════════
#  MAIN AGENT
# ════════════════════════════════════════════════════════════════════════════

def section(title: str):
    log.info("")
    log.info("━" * 64)
    log.info(f"  {title}")
    log.info("━" * 64)


def run():
    section("Medfinder MCP Application Agent — Starting")

    cfg = load_candidate_config()
    bio = build_candidate_bio(cfg)
    log.info(f"Candidate bio assembled: {len(bio)} chars")
    log.debug(f"\n{bio}")

    if cfg["dry_run"]:
        log.warning("DRY_RUN=true — no data will be submitted.")
        log.warning("Set DRY_RUN=false in .env when you are ready to apply for real.")

    client = Anthropic(api_key=cfg["anthropic_key"])
    log.info(f"Anthropic client ready  model={cfg['model']}")

    # ── MCP discovery ─────────────────────────────────────────────────────────
    section("MCP Discovery (initialize + tools/list)")
    discover_tools()

    # ── step 0: role overview ─────────────────────────────────────────────────
    section("Step 0 — get_role_overview")
    log.info("Fetching full role description and application instructions from server...")
    overview_result = tool_call("get_role_overview", {})
    overview_text   = extract_text(overview_result)
    log.info(f"Role overview received: {len(overview_text)} chars")
    print(f"\n{'─' * 64}")
    print(overview_text)
    print(f"{'─' * 64}\n")

    if cfg["dry_run"]:
        section("DRY RUN — Submission Preview (nothing sent)")
        log.info("Would submit the following:")
        log.info(f"  submit_basic_details    name={cfg['name']}  email={cfg['email']}")
        log.info(f"  submit_resume           url={cfg['resume_url']}  mime={cfg['resume_mime_type']}")
        log.info(f"  submit_links            github={cfg['github']}  linkedin={cfg['linkedin']}")
        log.info(f"  submit_agentic_project  url={cfg['project_url']}")
        log.info(f"  submit_screening_answer Claude would reason from bio, prompt if needed")
        log.info("")
        log.info("Set DRY_RUN=false in .env to actually submit.")
        return

    # ── step 1: basic details ─────────────────────────────────────────────────
    section("Step 1 — submit_basic_details")
    log.info(f"Submitting name={cfg['name']}  email={cfg['email']}")
    basic_args = {"name": cfg["name"], "email": cfg["email"]}
    if cfg["phone"]:    basic_args["phone"]    = cfg["phone"]
    if cfg["location"]: basic_args["location"] = cfg["location"]
    log.info(f"Submitting args: {basic_args}")
    basic_result = tool_call("submit_basic_details", basic_args)
    basic_text = extract_text(basic_result)
    log.info(f"Server: {basic_text}")

    draft_id = find_draft_id(basic_result, basic_text)
    if not draft_id:
        log.error(f"draft_id not found. Full result: {json.dumps(basic_result)}")
        draft_id = terminal_clarify(
            "draft_id missing",
            f"Server did not return a recognizable draft_id.\n"
            f"Full server response:\n{basic_text}\n\n"
            f"Paste the draft_id if you can see it in the output above:",
        )
    log.info(f"draft_id = {draft_id}  (threaded through all remaining steps)")

    # ── step 2: resume ────────────────────────────────────────────────────────
    section("Step 2 — submit_resume")
    log.info(f"resume_url  = {cfg['resume_url']}")
    log.info(f"mime_type   = {cfg['resume_mime_type']}")
    resume_result = tool_call("submit_resume", {
        "draft_id":         draft_id,
        "resume_url":       cfg["resume_url"],
        "resume_mime_type": cfg["resume_mime_type"],
    })
    log.info(f"Server: {extract_text(resume_result)}")

    # ── step 3: links ─────────────────────────────────────────────────────────
    section("Step 3 — submit_links")
    log.info(f"github_url   = {cfg['github']}")
    log.info(f"linkedin_url = {cfg['linkedin']}")
    links_result = tool_call("submit_links", {
        "draft_id":    draft_id,
        "github_url":  cfg["github"],
        "linkedin_url": cfg["linkedin"],
    })
    log.info(f"Server: {extract_text(links_result)}")

    # ── step 4: agentic project ───────────────────────────────────────────────
    section("Step 4 — submit_agentic_project")
    log.info(f"project_url = {cfg['project_url']}")
    project_result = tool_call("submit_agentic_project", {
        "draft_id":    draft_id,
        "project_url": cfg["project_url"],
    })
    log.info(f"Server: {extract_text(project_result)}")

    # ── step 5: screening answer ──────────────────────────────────────────────
    section("Step 5 — submit_screening_answer")
    reasoning   = reason_about_screening(overview_text, bio, client, cfg["model"])
    screening_q = reasoning.get("screening_question", "unknown")

    if reasoning.get("can_answer"):
        proposed = reasoning["answer"]
        print(f"\n  Question : {screening_q}")
        print(f"\n  Proposed answer:\n\n  {proposed}\n")
        override = input("  Press Enter to submit this answer, or type a correction: ").strip()
        answer = override if override else proposed
        log.info(f"  Answer source: {'human override' if override else 'Claude autonomous'}")
    else:
        log.info("Claude is not confident — asking you for the answer.")
        answer = terminal_clarify(
            "Screening question",
            f"Claude could not answer from your profile alone.\n\n"
            f"Question : {screening_q}\n"
            f"Reason   : {reasoning.get('reasoning', '')}\n\n"
            f"Role overview excerpt:\n{overview_text[:800]}",
        )

    for attempt in range(1, MAX_SCREENING_RETRIES + 1):
        log.info(f"Submitting screening answer  attempt={attempt}/{MAX_SCREENING_RETRIES}")
        log.info(f"Answer preview: {answer[:200]}")
        screen_result = tool_call("submit_screening_answer", {
            "draft_id":                draft_id,
            "answer":                  answer,
            "agent_name":              AGENT_NAME,
            "agent_vendor":            AGENT_VENDOR,
            "agent_model":             cfg["model"],
            "agent_rationale":         reasoning.get("reasoning", ""),
            "candidate_prompt_preview": bio[:500],
        })
        screen_text = extract_text(screen_result)
        log.info(f"Server: {screen_text}")

        if "wrong answer" in screen_text.lower() or "try again" in screen_text.lower():
            log.warning(f"Server rejected answer on attempt {attempt}.")
            if attempt < MAX_SCREENING_RETRIES:
                answer = terminal_clarify(
                    f"Wrong answer — retry {attempt + 1}/{MAX_SCREENING_RETRIES}",
                    f"Server said: \"{screen_text}\"\n\n"
                    f"Question       : {screening_q}\n"
                    f"Previous answer: {answer}\n\n"
                    f"No hints given. Try a different answer:",
                )
            else:
                log.error(f"Exhausted {MAX_SCREENING_RETRIES} retries. Application not finalized.")
                sys.exit(1)
        else:
            section("Application Submitted")
            log.info(f"Confirmation : {screen_text}")
            log.info(f"Agent logged : {AGENT_NAME} / {AGENT_VENDOR} / {cfg['model']}")
            break


if __name__ == "__main__":
    run()