# app.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from flask import Flask, request, render_template_string

from google import genai

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. Load API key from apikey.env
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
load_dotenv("apikey.env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment. Check apikey.env")

client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-2.5-flash"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2. Core LLM helper
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def call_gemini(prompt: str) -> str:
    """
    Call Gemini 2.5 Flash with a text prompt and return response text.
    """
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text or ""


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3. Agent prompt definitions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def cardiologist_agent(medical_report: str) -> str:
    prompt = f"""
You are an experienced cardiologist.

You are reviewing the following medical report. Analyze ONLY from a cardiology perspective.

Report:
\"\"\"{medical_report}\"\"\"

Tasks:
1. Identify any possible cardiovascular issues, risk factors, and red flags.
2. Suggest possible differential diagnoses (cardiology-focused), clearly marked as *not confirmed*.
3. Recommend further tests or evaluations that a real cardiologist might order.
4. Provide a short explanation understandable to a layperson.

IMPORTANT:
- Do NOT provide actual medical diagnosis or treatment.
- Clearly state that this is for educational purposes only and not medical advice.
"""
    return call_gemini(prompt)


def psychologist_agent(medical_report: str) -> str:
    prompt = f"""
You are an experienced clinical psychologist.

You are reviewing the following medical report. Analyze ONLY from a psychological / mental health perspective.

Report:
\"\"\"{medical_report}\"\"\"

Tasks:
1. Identify possible psychological patterns, symptoms, or concerns.
2. Suggest possible differential psychological explanations, clearly marked as *not confirmed*.
3. Suggest what kind of real-world psychological assessments or referrals might be appropriate.
4. Provide a short explanation understandable to a layperson.

IMPORTANT:
- Do NOT provide actual diagnosis or treatment.
- Clearly state that this is for educational purposes only and not medical advice.
"""
    return call_gemini(prompt)


def pulmonologist_agent(medical_report: str) -> str:
    prompt = f"""
You are an experienced pulmonologist.

You are reviewing the following medical report. Analyze ONLY from a respiratory / pulmonology perspective.

Report:
\"\"\"{medical_report}\"\"\"

Tasks:
1. Identify possible respiratory issues, patterns, or risk factors.
2. Suggest possible differential diagnoses (pulmonology-focused), clearly marked as *not confirmed*.
3. Recommend further tests or evaluations a pulmonologist might consider.
4. Provide a short explanation understandable to a layperson.

IMPORTANT:
- Do NOT provide actual medical diagnosis or treatment.
- Clearly state that this is for educational purposes only and not medical advice.
"""
    return call_gemini(prompt)


def multidisciplinary_team_agent(cardiologist_report: str,
                                 psychologist_report: str,
                                 pulmonologist_report: str) -> str:
    prompt = f"""
You are a multidisciplinary medical team composed of:
- Cardiologist
- Psychologist
- Pulmonologist

Each specialist has provided a preliminary (non-diagnostic) report.

Cardiologist report:
\"\"\"{cardiologist_report}\"\"\"

Psychologist report:
\"\"\"{psychologist_report}\"\"\"

Pulmonologist report:
\"\"\"{pulmonologist_report}\"\"\"

Your task:
1. Integrate these three perspectives into a single, coherent narrative.
2. Highlight possible connections between cardiovascular, psychological, and respiratory aspects.
3. Suggest a list of questions a patient should ask their real doctor.
4. Suggest what types of real specialists/tests they might want to consult in real life.
5. Summarize in a patient-friendly way.

CRITICAL:
- Do NOT provide a definitive diagnosis or treatment plan.
- Repeatedly remind that this is NOT medical advice and NOT a substitute for a real doctor.
- Label the output clearly as an AI-generated educational summary.
"""
    return call_gemini(prompt)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 4. Function to run all agents concurrently
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def analyze_medical_report(medical_report: str):
    agents = {
        "Cardiologist": cardiologist_agent,
        "Psychologist": psychologist_agent,
        "Pulmonologist": pulmonologist_agent,
    }

    responses = {}

    # Run agents concurrently
    with ThreadPoolExecutor() as executor:
        future_to_name = {
            executor.submit(func, medical_report): name
            for name, func in agents.items()
        }

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            responses[name] = future.result()

    # Team-level integration
    final_diagnosis = multidisciplinary_team_agent(
        cardiologist_report=responses["Cardiologist"],
        psychologist_report=responses["Psychologist"],
        pulmonologist_report=responses["Pulmonologist"],
    )

    # Prepare final text & save to file
    final_diagnosis_text = "### Final AI-Generated Educational Summary (NOT Medical Advice):\n\n" + final_diagnosis

    txt_output_path = "results/final_diagnosis.txt"
    os.makedirs(os.path.dirname(txt_output_path), exist_ok=True)

    with open(txt_output_path, "w", encoding="utf-8") as f:
        f.write(final_diagnosis_text)

    return responses, final_diagnosis_text, txt_output_path


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 5. Flask app with dark-mode UI
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
app = Flask(__name__)


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>AI Medical Multi-Agent (Gemini 2.5 Flash)</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <!-- Simple font -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

    <style>
        :root {
            --bg-main: #020617;
            --bg-card: #020617;
            --bg-card-soft: #020617;
            --accent: #38bdf8;
            --accent-soft: rgba(56, 189, 248, 0.15);
            --text-main: #e5e7eb;
            --text-muted: #9ca3af;
            --border-subtle: rgba(148, 163, 184, 0.3);
            --danger: #f97373;
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            padding: 0;
            font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
            background: radial-gradient(circle at top, #0f172a, #020617 60%);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: flex-start;
        }

        .wrapper {
            width: 100%;
            max-width: 1100px;
            padding: 24px 16px 40px;
        }

        .glass-card {
            background: radial-gradient(circle at top left, rgba(56, 189, 248, 0.06), transparent 40%),
                        radial-gradient(circle at bottom right, rgba(168, 85, 247, 0.05), transparent 50%),
                        rgba(15, 23, 42, 0.96);
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.4);
            box-shadow:
                0 24px 80px rgba(15, 23, 42, 0.9),
                0 0 0 1px rgba(15, 23, 42, 0.9);
            padding: 24px 22px 22px;
            backdrop-filter: blur(24px);
        }

        .header {
            display: flex;
            flex-direction: column;
            gap: 6px;
            margin-bottom: 20px;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(56, 189, 248, 0.35);
            font-size: 12px;
            color: var(--accent);
            width: fit-content;
        }

        .pulse-dot {
            width: 6px;
            height: 6px;
            border-radius: 999px;
            background: var(--accent);
            box-shadow: 0 0 0 4px rgba(56, 189, 248, 0.4);
        }

        h1 {
            font-size: 22px;
            font-weight: 600;
            margin: 0;
            letter-spacing: 0.02em;
            color: #f9fafb;
        }

        .subtitle {
            margin: 0;
            font-size: 13px;
            color: var(--text-muted);
        }

        .warning {
            margin-top: 10px;
            padding: 8px 10px;
            border-radius: 10px;
            border: 1px solid rgba(248, 113, 113, 0.4);
            background: rgba(24, 24, 27, 0.9);
            font-size: 12px;
            color: #fecaca;
            display: flex;
            align-items: flex-start;
            gap: 8px;
        }

        .warning-icon {
            font-size: 14px;
            margin-top: 2px;
        }

        form {
            margin-top: 18px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        label {
            font-size: 13px;
            font-weight: 500;
            color: #e5e7eb;
        }

        textarea {
            width: 100%;
            min-height: 200px;
            resize: vertical;
            padding: 10px 11px;
            border-radius: 12px;
            border: 1px solid var(--border-subtle);
            background: rgba(15, 23, 42, 0.9);
            color: var(--text-main);
            font-size: 13px;
            font-family: "JetBrains Mono", "Consolas", ui-monospace, SFMono-Regular, Menlo, Monaco, monospace;
        }

        textarea:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.4);
        }

        .controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 6px;
            gap: 12px;
            flex-wrap: wrap;
        }

        .hint {
            font-size: 11px;
            color: var(--text-muted);
        }

        .btn-primary {
            border: none;
            padding: 8px 16px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            background: linear-gradient(135deg, #38bdf8, #6366f1);
            color: #0b1020;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            box-shadow: 0 10px 30px rgba(56, 189, 248, 0.4);
            transition: transform 0.08s ease, box-shadow 0.08s ease, filter 0.08s ease;
        }

        .btn-primary span.icon {
            font-size: 15px;
        }

        .btn-primary:hover {
            transform: translateY(-1px);
            filter: brightness(1.05);
            box-shadow: 0 14px 40px rgba(56, 189, 248, 0.55);
        }

        .btn-primary:active {
            transform: translateY(0);
            box-shadow: 0 8px 24px rgba(56, 189, 248, 0.3);
        }

        .error {
            margin-top: 10px;
            font-size: 13px;
            color: #fecaca;
        }

        .results-wrapper {
            margin-top: 20px;
            display: grid;
            grid-template-columns: minmax(0, 1.15fr) minmax(0, 1.15fr);
            gap: 14px;
        }

        @media (max-width: 900px) {
            .results-wrapper {
                grid-template-columns: minmax(0, 1fr);
            }
        }

        .result-card {
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.4);
            background: radial-gradient(circle at top, rgba(15, 23, 42, 0.9), rgba(15, 23, 42, 0.98));
            padding: 14px 12px;
        }

        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }

        .result-title {
            font-size: 14px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .role-pill {
            padding: 3px 9px;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.5);
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-muted);
        }

        .role-icon {
            width: 18px;
            height: 18px;
            border-radius: 999px;
            background: rgba(248, 250, 252, 0.06);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
        }

        pre {
            margin: 0;
            font-size: 12px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
            background: #020617;
            border-radius: 10px;
            padding: 8px 10px;
            border: 1px solid rgba(31, 41, 55, 0.9);
            font-family: "JetBrains Mono", "Consolas", ui-monospace, SFMono-Regular, Menlo, Monaco, monospace;
            max-height: 420px;
            overflow: auto;
        }

        .summary-card {
            grid-column: span 2;
        }

        @media (max-width: 900px) {
            .summary-card {
                grid-column: span 1;
            }
        }

        .summary-tagline {
            font-size: 11px;
            color: var(--text-muted);
            margin: 2px 0 0;
        }

        .file-info {
            margin-top: 6px;
            font-size: 11px;
            color: var(--text-muted);
        }

        code {
            font-family: "JetBrains Mono", "Consolas", ui-monospace, SFMono-Regular, Menlo, Monaco, monospace;
            font-size: 11px;
            background: rgba(15, 23, 42, 0.9);
            padding: 1px 6px;
            border-radius: 999px;
            border: 1px solid rgba(75, 85, 99, 0.8);
        }
    </style>
</head>
<body>
    <div class="wrapper">
        <div class="glass-card">
            <div class="header">
                <div class="badge">
                    <span class="pulse-dot"></span>
                    <span>Gemini 2.5 Flash · Multi-Agent Medical Demo</span>
                </div>
                <h1>AI-Assisted Medical Report Analysis</h1>
                <p class="subtitle">
                    Parallel specialist-style AI agents (Cardiology · Psychology · Pulmonology) plus a final multidisciplinary summary.
                </p>

                <div class="warning">
                    <span class="warning-icon">⚠</span>
                    <span>
                        This tool is for <strong>educational and experimental</strong> purposes only.
                        It is <strong>NOT</strong> medical advice, diagnosis, or treatment.
                        Always consult a licensed medical professional for real health concerns.
                    </span>
                </div>
            </div>

            <form method="POST">
                <label for="report">Paste Medical Report Text</label>
                <textarea id="report" name="report" placeholder="Paste or type the clinical / medical note you want the AI agents to analyze...">{{ report or "" }}</textarea>

                <div class="controls">
                    <p class="hint">
                        Tip: Remove any personal identifying information before pasting. Short notes also work — the agents will still collaborate.
                    </p>
                    <button type="submit" class="btn-primary">
                        <span class="icon">⚡</span>
                        <span>Analyze with AI Agents</span>
                    </button>
                </div>
            </form>

            {% if error %}
                <div class="error">{{ error }}</div>
            {% endif %}

            {% if final_diagnosis %}
            <div class="results-wrapper">
                <div class="result-card">
                    <div class="result-header">
                        <div class="result-title">
                            <div class="role-icon">♥</div>
                            <span>Cardiologist</span>
                        </div>
                        <span class="role-pill">Specialist View</span>
                    </div>
                    <pre>{{ cardiologist_report }}</pre>
                </div>

                <div class="result-card">
                    <div class="result-header">
                        <div class="result-title">
                            <div class="role-icon">Ψ</div>
                            <span>Psychologist</span>
                        </div>
                        <span class="role-pill">Specialist View</span>
                    </div>
                    <pre>{{ psychologist_report }}</pre>
                </div>

                <div class="result-card">
                    <div class="result-header">
                        <div class="result-title">
                            <div class="role-icon">☁</div>
                            <span>Pulmonologist</span>
                        </div>
                        <span class="role-pill">Specialist View</span>
                    </div>
                    <pre>{{ pulmonologist_report }}</pre>
                </div>

                <div class="result-card summary-card">
                    <div class="result-header">
                        <div class="result-title">
                            <div class="role-icon">◎</div>
                            <span>Multidisciplinary Summary</span>
                        </div>
                        <span class="role-pill">AI-Generated · Not Medical Advice</span>
                    </div>
                    <p class="summary-tagline">
                        Integrated view combining cardiology, psychology, and pulmonology perspectives into one educational summary.
                    </p>
                    <pre>{{ final_diagnosis }}</pre>
                    <p class="file-info">
                        Saved locally as <code>{{ output_path }}</code>
                    </p>
                </div>
            </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        report = request.form.get("report", "").strip()
        if not report:
            return render_template_string(
                INDEX_HTML,
                error="Please paste a medical report first.",
                report="",
            )

        specialist_reports, final_diagnosis_text, output_path = analyze_medical_report(report)

        return render_template_string(
            INDEX_HTML,
            report=report,
            cardiologist_report=specialist_reports.get("Cardiologist", ""),
            psychologist_report=specialist_reports.get("Psychologist", ""),
            pulmonologist_report=specialist_reports.get("Pulmonologist", ""),
            final_diagnosis=final_diagnosis_text,
            output_path=output_path,
            error=None,
        )

    # GET
    return render_template_string(INDEX_HTML, report="", error=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
