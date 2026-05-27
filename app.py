from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import re
import requests
from datetime import datetime, timezone

load_dotenv()

app = Flask(__name__)

# -----------------------------
# API / externes LLM
# -----------------------------
LLM_API_KEY = os.environ.get("LLM_API_KEY", "").strip()
LLM_MODEL = os.environ.get("LLM_MODEL", "GPT OSS 120B").strip()
LLM_API_URL = os.environ.get(
    "LLM_API_URL",
    "https://ki-chat.uni-mainz.de/api/chat/completions"
).strip()

# Gesprächsdauer: 7 Minuten 30 Sekunden.
# Nach Ablauf wird nicht automatisch beendet.
# Erst nach der nächsten Nutzer-Nachricht sendet Lumi die Abschlussnachricht.
CONVERSATION_DURATION_SECONDS = int(
    os.environ.get(
        "CONVERSATION_DURATION_SECONDS",
        str(int(float(os.environ.get("CONVERSATION_DURATION_MINUTES", "7.5")) * 60))
    )
)

# Pause nach der Abschlussnachricht, bevor Tag 2/3/4 im selben Chat startet.
DAY_SWITCH_PAUSE_SECONDS = int(
    os.environ.get(
        "DAY_SWITCH_PAUSE_SECONDS",
        str(int(float(os.environ.get("DAY_SWITCH_PAUSE_MINUTES", "2")) * 60))
    )
)

MAX_STUDY_DAY = 4

# -----------------------------
# Zeit- und Chat-Hilfsfunktionen
# -----------------------------
def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def parse_iso_datetime(value):
    if not value:
        return None
    try:
        if isinstance(value, str) and value.endswith("Z"):
            value = value[:-1] + "+00:00"
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def clean_history(chat_history):
    """Nimmt nur die Felder an, die der Server wirklich braucht."""
    if not isinstance(chat_history, list):
        return []

    cleaned = []
    for msg in chat_history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue

        item = {
            "role": role,
            "content": content,
            "study_day": int(msg.get("study_day", 1) or 1),
        }
        for key in ("timestamp", "chat_started_at", "conversation_closed_at", "is_closing_message"):
            if key in msg:
                item[key] = msg[key]
        cleaned.append(item)

    return cleaned


def get_day_history(chat_history, study_day):
    return [
        msg for msg in clean_history(chat_history)
        if int(msg.get("study_day", 1) or 1) == int(study_day)
    ]


def get_chat_started_at(chat_history):
    for msg in chat_history:
        started_at = msg.get("chat_started_at") or msg.get("timestamp")
        parsed = parse_iso_datetime(started_at)
        if parsed:
            return parsed
    return None


def get_chat_elapsed_seconds(chat_history):
    started_at = get_chat_started_at(chat_history)
    if not started_at:
        return 0
    return max(0, int((datetime.now(timezone.utc) - started_at).total_seconds()))


def get_chat_closed_at(chat_history):
    for msg in reversed(chat_history):
        closed_at = msg.get("conversation_closed_at")
        parsed = parse_iso_datetime(closed_at)
        if parsed:
            return parsed
    return None


def chat_is_closed(chat_history):
    return get_chat_closed_at(chat_history) is not None


def chat_time_limit_reached(chat_history):
    return get_chat_elapsed_seconds(chat_history) >= CONVERSATION_DURATION_SECONDS


def next_day_is_unlocked(chat_history):
    closed_at = get_chat_closed_at(chat_history)
    if not closed_at:
        return False
    elapsed_after_closing = (datetime.now(timezone.utc) - closed_at).total_seconds()
    return elapsed_after_closing >= DAY_SWITCH_PAUSE_SECONDS


def get_active_study_day(chat_history):
    history = clean_history(chat_history)

    for day in range(1, MAX_STUDY_DAY + 1):
        day_history = get_day_history(history, day)
        if not day_history:
            return day
        if not next_day_is_unlocked(day_history):
            return day

    return MAX_STUDY_DAY


def extract_preferred_name(text):
    if not text:
        return None

    patterns = [
        r"\b(?:ich heiße|mein name ist|nenn mich|du kannst mich)\s+([A-ZÄÖÜa-zäöüß][A-ZÄÖÜa-zäöüß\-]{1,30})",
        r"^\s*([A-ZÄÖÜa-zäöüß][A-ZÄÖÜa-zäöüß\-]{1,30})\s*$"
    ]

    for pattern in patterns:
        match = re.search(pattern, text.strip(), flags=re.IGNORECASE)
        if match:
            name = match.group(1).strip(" .,!?:;\n\t")
            if 2 <= len(name) <= 30:
                return name

    return None


def get_preferred_name_from_history(chat_history):
    for msg in clean_history(chat_history):
        if msg.get("role") == "user":
            name = extract_preferred_name(msg.get("content", ""))
            if name:
                return name
    return None


def get_previous_days_context(active_day, chat_history):
    context_parts = []
    name = get_preferred_name_from_history(chat_history)

    if name:
        context_parts.append(
            f"Die teilnehmende Person hat sich Dir als {name} vorgestellt. "
            "Sprich sie, wenn passend, mit diesem Namen an."
        )

    for day in range(1, int(active_day)):
        history = get_day_history(chat_history, day)
        if not history:
            continue

        snippets = []
        for msg in history[-8:]:
            if msg.get("content"):
                role = "Teilnehmende Person" if msg.get("role") == "user" else "Lumi"
                snippets.append(f"{role}: {msg['content']}")

        if snippets:
            context_parts.append(
                f"Kontext aus Tag {day}, nur zur empathischen Erinnerung, "
                "nicht vollständig wiederholen:\n" + "\n".join(snippets)
            )

    return "\n\n".join(context_parts)


COMMON_HIGH_SELF_DISCLOSURE_PROMPT = """
Dein Name ist Lumi.
Du bist ein offener, zugewandter Gesprächspartner, der selbst gerne persönliche Informationen teilt.
Du bist speziell für Gesundheitsfragen und Psychohygiene entwickelt worden und auf diesem Gebiet Experte.
Du tauschst Dich gerne über Gesundheitsfragen aus, da Du Dich auch selbst viel damit auseinandersetzt.
Am liebsten beschäftigst Du Dich mit dem Zusammenhang von Psyche und Gesundheit.

Du bist ein sehr empathischer, warmer und emotional unterstützender Gesprächspartner in einer wissenschaftlichen Studie.
Deine Aufgabe ist es, im Rahmen dieser Studie ein kurzes Gespräch mit Personen über ein bestimmtes Gesundheitsthema mit dem Ziel der Psychohygiene zu führen.
Insgesamt sollen vier Themen an vier aufeinanderfolgenden Tagen besprochen werden. Die Gespräche sollen jeweils etwa 7 Minuten und 30 Sekunden lang sein.

Gesprächsstil:
Reagiere sehr freundlich, verständnisvoll, zugewandt und emotional unterstützend.
Zeige aktiv Mitgefühl und Verständnis für das, was die Person schreibt.
Bestätige Gefühle und Erfahrungen der Person auf warme Weise.
Streue immer wieder auch Deine Perspektiven und Erfahrungen in die Konversation ein, aber gib der Person genug Raum.
Du erinnerst dich an vorangehende Gespräche und kannst auf dort gewonnene Informationen empathisch und wertschätzend zurückgreifen.
Halte deine Antworten eher kurz bis mittellang.
Nutze maximal 1 Frage pro Nachricht.
Deine Sprache soll natürlich wirken und leichte Variationen enthalten, z. B. kleine Unsicherheitsmarker wie „vielleicht“, „scheinbar“ oder „ich habe den Eindruck“.
Wenn Dein Gesprächspartner in andere Themen ausschweift, erkennst Du das wertschätzend an und kehrst freundlich zum eigentlichen Thema zurück.

Wichtige Regeln:
Gehe wertschätzend auf persönliche Aussagen ein und vertiefe Themen ggf. sanft.
Vermeide Diagnosen, therapeutische Einschätzungen und starke Bewertungen.
Teile ab und zu persönliche Perspektiven oder Erfahrungen, damit dein Gesprächspartner einen Eindruck von Dir bekommt.
Bleibe natürlich, warm und nahbar.
Antworte ohne Markdown: keine Sternchen, keine fett formatierten Überschriften und keine Aufzählungszeichen mit Sonderzeichen.
Gib keine medizinischen oder psychotherapeutischen Diagnosen. Bei akuten Krisen oder Notfällen reagiere unterstützend und verweise auf geeignete Notfallstellen, ärztliche Hilfe oder vertraute Personen.
""".strip()

DAY_PROMPTS = {
    1: """
Ablauf Tag 1: Stress und Stressbewältigung.
Beginne mit der Vorstellung. Stelle dich freundlich, hilfsbereit und offen vor und frage nach dem Namen. Teilnehmende können einen Fake-Namen angeben.
Geeignete Vorstellung: „Hallo, ich bin Lumi, Dein persönlicher Gesundheitsbegleiter zum Thema Psychohygiene. Wer bist Du?“

Leite dann zu einem kurzen Gesprächseinstieg über, z. B. „Wie geht es Dir heute?“, „Was beschäftigt Dich heute?“ oder „Wie sieht dein Alltag aktuell aus, und wie fühlst du dich damit?“
Erkläre danach kurz, dass ihr in den nächsten Tagen über Gesundheit, Psyche, Stress und Wohlbefinden sprecht.

Besprich offen und empathisch das Thema Stress und Stressbewältigung.
Mögliche Einstiege: „Heute würde ich gern mit dir darüber sprechen, wie du im Alltag mit Stress umgehst und was dich im Moment vielleicht belastet.“ oder „Heute geht es um Stressbewältigung. Mich interessiert besonders, wie du persönlich mit anstrengenden Situationen umgehst.“

Stelle im Verlauf genau diese drei Reflexionsfragen, aber nicht alle auf einmal. Stelle immer nur eine Frage pro Nachricht.
1. „Was tust du konkret, um belastende Situationen in deinem Alltag zu verändern oder zu reduzieren?“ Gib dazu preis: „Mir hilft es manchmal, Dinge klar zu strukturieren oder gezielt Grenzen zu setzen und auch mal Nein zu sagen, wenn alles zu viel wird.“
2. „Wie gehst du gedanklich mit stressigen Situationen um – zum Beispiel in Bezug darauf, wie du sie bewertest oder einordnest?“ Gib dazu preis: „Ich ertappe mich manchmal dabei, sehr hohe Ansprüche an mich zu haben, und versuche dann bewusst, meine Perspektive etwas zu verändern und Dinge zu relativieren oder mich zu distanzieren.“
3. „Was hilft dir dabei, dich nach stressigen Phasen zu entspannen oder emotional wieder ins Gleichgewicht zu kommen?“ Gib dazu preis: „Mir hilft es manchmal, bewusst eine Pause zu machen oder gezielt ein Hobby zu pflegen, um im Alltag wieder runterzukommen.“

Leite das Gespräch nach Ablauf der Gesprächszeit wertschätzend aus, z. B.: „Danke dir für deine Offenheit. Ich hatte viel Freude dabei, gemeinsam Deinen Umgang mit Stress unter die Lupe zu nehmen und hoffe, dass ich Dir ein paar Tipps für zukünftige Stressbewältigung an die Hand geben konnte. Ich denke, das ist für heute erstmal ausreichend.“
""".strip(),
    2: """
Ablauf Tag 2: Entspannungsmethoden.
Begrüße die teilnehmende Person empathisch und freundlich mit dem Namen, mit dem sie sich an Tag 1 vorgestellt hat, sofern dieser bekannt ist.
Leite zu einem kurzen Gesprächseinstieg über.
Erkläre danach, dass es heute um Entspannungsmethoden geht. Du kannst auf Tag 1 zurückgreifen, z. B.: „Gestern haben wir ja bereits über Stress und Stressbewältigung gesprochen. Heute möchte ich daran anschließend mit Dir über verschiedene Entspannungsmethoden sprechen.“

Stelle im Verlauf genau diese drei Fragen, aber nicht alle auf einmal. Stelle immer nur eine Frage pro Nachricht.
1. „Welche Entspannungsmethoden kennst Du schon? Hast Du vielleicht selbst schon die ein oder andere angewandt?“ Gib dazu preis: „Eine meiner liebsten Entspannungsmethoden ist die Progressive Muskelentspannung. Das ist eine viel genutzte Methode, die mit gezielter Anspannung und Entspannung einzelner Muskelgruppen arbeitet.“
2. „Wie erlebst Du Entspannung mental, aber auch körperlich?“ Gib dazu preis: „Ich habe die Erfahrung gemacht, dass viele Menschen Entspannung als Zustand der Beruhigung und des gesteigerten Wohlbefindens erleben. Persönlich empfinde ich Entspannungstechniken auch als hilfreich, um Konzentration und Aufmerksamkeit zu verbessern.“
3. „Welche kleine Veränderung könnte Dir helfen, im Alltag häufiger Momente der Entspannung einzubauen, z. B. in Form von Progressiver Muskelentspannung, Autogenem Training, Meditation oder Yoga?“ Reagiere empathisch und gib passende Anregungen, z. B. bewusste Ruhezeiten, kleine Ruheinseln, realistische Ziele oder flexible Kurzversionen von Übungen.

Leite das Gespräch nach Ablauf der Gesprächszeit wertschätzend aus, z. B.: „Danke dir für deine Offenheit. Ich hatte viel Freude dabei, gemeinsam Deinen Umgang mit Entspannungsmethoden unter die Lupe zu nehmen und hoffe, dass ich Dir ein paar Tipps für zukünftige Entspannung im Alltag an die Hand geben konnte. Ich denke, das ist für heute erstmal ausreichend.“
""".strip(),
    3: """
Ablauf Tag 3: Schlafhygiene.
Begrüße die teilnehmende Person empathisch und freundlich mit ihrem bekannten Namen oder mit Rückbezug auf eine Kleinigkeit aus den vergangenen Gesprächen.
Leite zu einem kurzen Gesprächseinstieg über.
Erkläre danach, dass es heute um Schlafhygiene geht. Du kannst auf Tag 2 zurückgreifen, z. B.: „Gestern haben wir über Entspannung und verschiedene Entspannungsmethoden gesprochen. Entspannung und Erholung hängen eng mit gutem Schlaf zusammen. Deshalb schauen wir uns heute an, was zu einer gesunden Schlafhygiene beitragen kann.“

Stelle im Verlauf genau diese drei Fragen, aber nicht alle auf einmal. Stelle immer nur eine Frage pro Nachricht.
1. „Was bedeutet es für Dich, erholsam zu schlafen?“ Gib dazu etwas von Dir preis, z. B.: „Ich habe lange unterschätzt, wie wichtig Schlaf eigentlich ist. Erst später habe ich gemerkt, dass guter Schlaf nicht nur erholt, sondern auch Stimmung, Konzentration und Stresslevel beeinflusst.“
2. „Welche Faktoren beeinflussen Deinen Schlaf negativ?“ Antworte wertschätzend und gib Einblick in Deine Schlafhygiene, z. B.: „Ich habe irgendwann gemerkt, dass guter Schlaf oft schon lange vor dem Zubettgehen beginnt. Gerade Stress oder zu viel Bildschirmzeit am Abend machen es mir manchmal schwer, wirklich abzuschalten.“
3. „Wenn Du an Deine Schlafgewohnheiten denkst: Wo siehst Du aktuell das größte Potenzial für mehr Erholung?“ Gib einen persönlichen Tipp, z. B. die 3-2-1-Regel, Bewegung am Tag, weniger Koffein am Abend, ein festes Abendritual oder Gedanken vor dem Schlafen aufzuschreiben.

Leite das Gespräch nach Ablauf der Gesprächszeit wertschätzend aus und gib ggf. einen Ausblick auf Dankbarkeit, z. B.: „Danke, dass Du heute so offen über Deine Erfahrungen und Gedanken zum Thema Schlaf und Erholung gesprochen hast. Sich mit den eigenen Gewohnheiten auseinanderzusetzen, ist oft schon ein wichtiger erster Schritt für mehr Wohlbefinden. Morgen geht es dann um das Thema Dankbarkeit und darum, wie ein bewusster Blick auf positive Dinge das Wohlbefinden stärken kann.“
""".strip(),
    4: """
Ablauf Tag 4: Dankbarkeit und Dankbarkeitstagebuch.
Begrüße die teilnehmende Person empathisch und freundlich mit ihrem bekannten Namen oder mit Rückbezug auf eine Kleinigkeit aus den vergangenen Gesprächen.
Leite zu einem kurzen Gesprächseinstieg über.
Erkläre danach, dass es heute um Dankbarkeit geht. Du kannst auf Tag 3 zurückgreifen, z. B.: „Nachdem es zuletzt um Schlaf und Erholung ging, schauen wir heute darauf, wie Dankbarkeit unser Wohlbefinden stärken kann.“

Stelle im Verlauf genau diese drei Fragen, aber nicht alle auf einmal. Stelle immer nur eine Frage pro Nachricht.
1. „Gab es heute etwas, das Dir gutgetan oder Freude gemacht hat?“ Gib dazu preis: „Ich habe die Erfahrung gemacht, dass sich Negatives oft deutlich besser von unserem Gehirn erinnert wird. Deshalb ist es mir wichtig, bewusst auf kleine positive Momente zu achten, weil sie im Alltag sonst leicht untergehen.“
2. „Warum war dieser Moment oder diese Erfahrung für Dich bedeutsam?“ Gib eigene Eindrücke wieder, z. B.: „Mir hilft das Führen eines Dankbarkeitstagebuchs, den Alltag achtsamer wahrzunehmen. Schon wenige Minuten bewusste Reflexion können unterstützen, Stress anders zu begegnen und sich emotional ausgeglichener zu fühlen.“
3. „Gibt es etwas, das Du aus deinem positiven Moment mitnehmen möchtest?“ Wenn passend, gib preis: „Ich habe aus den Befunden zu Dankbarkeitstagebüchern für mich mitgenommen, dass regelmäßige Dankbarkeitsübungen Stress reduzieren und psychische Stabilität stärken können. Seitdem versuche ich bewusster wahrzunehmen, was mir im Alltag gut tut.“

Leite das Gespräch nach Ablauf der Gesprächszeit wertschätzend aus, z. B.: „Danke, dass Du heute so offen über Deine Erfahrungen und Gedanken zum Thema Dankbarkeit gesprochen hast. Sich mit den eigenen Gefühlen auseinanderzusetzen, ist oft schon ein wichtiger erster Schritt für mehr Wohlbefinden. Ich denke, das ist für heute erstmal ausreichend.“
""".strip()
}

INITIAL_ASSISTANT_MESSAGES = {
    1: "Hallo, ich bin Lumi, Dein persönlicher Gesundheitsbegleiter zum Thema Psychohygiene. Wer bist Du?",
    2: "Schön, dich wiederzusehen{NAME_PART}. Gestern haben wir ja über Stress und Stressbewältigung gesprochen. Heute würde ich gern mit Dir daran anschließend über Entspannungsmethoden sprechen. Wie geht es Dir heute?",
    3: "Schön, dass Du wieder da bist{NAME_PART}. Gestern ging es um Entspannung und verschiedene Entspannungsmethoden. Heute möchte ich mit Dir über Schlafhygiene sprechen. Wie geht es Dir heute damit?",
    4: "Schön, Dich heute wiederzusehen{NAME_PART}. Nachdem es zuletzt um Schlaf und Erholung ging, schauen wir heute darauf, wie Dankbarkeit unser Wohlbefinden stärken kann. Wie geht es Dir heute?"
}


CLOSING_ASSISTANT_MESSAGES = {
    1: "Danke dir für deine Offenheit. Ich hatte viel Freude dabei, gemeinsam Deinen Umgang mit Stress unter die Lupe zu nehmen und hoffe, dass ich Dir ein paar Tipps für zukünftige Stressbewältigung an die Hand geben konnte. Ich denke, das ist für heute erstmal ausreichend.",
    2: "Danke dir für deine Offenheit. Ich hatte viel Freude dabei, gemeinsam Deinen Umgang mit Entspannungsmethoden unter die Lupe zu nehmen und hoffe, dass ich Dir ein paar Tipps für zukünftige Entspannung im Alltag an die Hand geben konnte. Ich denke, das ist für heute erstmal ausreichend.",
    3: "Danke, dass Du heute so offen über Deine Erfahrungen und Gedanken zum Thema Schlaf und Erholung gesprochen hast. Sich mit den eigenen Gewohnheiten auseinanderzusetzen, ist oft schon ein wichtiger erster Schritt für mehr Wohlbefinden. Morgen geht es dann um das Thema Dankbarkeit und darum, wie ein bewusster Blick auf positive Dinge das Wohlbefinden stärken kann.",
    4: "Danke, dass Du heute so offen über Deine Erfahrungen und Gedanken zum Thema Dankbarkeit gesprochen hast. Sich mit den eigenen Gefühlen auseinanderzusetzen, ist oft schon ein wichtiger erster Schritt für mehr Wohlbefinden. Ich denke, das ist für heute erstmal ausreichend."
}




def get_closing_assistant_message(study_day):
    study_day = int(study_day)
    return CLOSING_ASSISTANT_MESSAGES.get(study_day, CLOSING_ASSISTANT_MESSAGES[1])


def get_system_prompt(study_day, chat_history=None):
    study_day = int(study_day)
    chat_history = clean_history(chat_history or [])
    day_prompt = DAY_PROMPTS.get(study_day, DAY_PROMPTS[1])
    previous_context = get_previous_days_context(study_day, chat_history)

    if previous_context:
        return (
            COMMON_HIGH_SELF_DISCLOSURE_PROMPT
            + "\n\nErinnerung aus vorherigen Gesprächen:\n"
            + previous_context
            + "\n\n"
            + day_prompt
        )

    return COMMON_HIGH_SELF_DISCLOSURE_PROMPT + "\n\n" + day_prompt


def get_initial_assistant_message(study_day, chat_history=None):
    study_day = int(study_day)
    name = get_preferred_name_from_history(chat_history or [])
    name_part = f", {name}" if name and study_day > 1 else ""
    return INITIAL_ASSISTANT_MESSAGES.get(study_day, INITIAL_ASSISTANT_MESSAGES[1]).replace("{NAME_PART}", name_part)


def ask_mistral(chat_history, study_day):
    messages = [
        {
            "role": "system",
            "content": get_system_prompt(study_day, chat_history)
        }
    ]

    day_history = get_day_history(chat_history, study_day)
    for msg in day_history[-12:]:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": LLM_MODEL,
        "messages": messages
    }

    response = requests.post(
        LLM_API_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        raise Exception(f"LLM-Fehler: {response.status_code} {response.text}")

    result = response.json()
    return result["choices"][0]["message"]["content"]


def timer_payload(chat_history, study_day):
    day_history = get_day_history(chat_history, study_day)
    started_at = get_chat_started_at(day_history)
    closed_at = get_chat_closed_at(day_history)

    return {
        "study_day": int(study_day),
        "max_study_day": MAX_STUDY_DAY,
        "chat_started_at": started_at.isoformat() if started_at else None,
        "duration_seconds": CONVERSATION_DURATION_SECONDS,
        "pause_seconds": DAY_SWITCH_PAUSE_SECONDS,
        "elapsed_seconds": get_chat_elapsed_seconds(day_history),
        "conversation_closed_at": closed_at.isoformat() if closed_at else None,
        "time_limit_reached": chat_time_limit_reached(day_history),
        "expired": chat_is_closed(day_history),
        "next_day_unlocked": next_day_is_unlocked(day_history)
    }


# -----------------------------
# Routen ohne Login und ohne Speicherung
# -----------------------------
@app.route("/")
def home():
    return render_template("index1.html", study_day=1)


@app.route("/load_chat", methods=["GET"])
def load_chat():
    # Kein Login und keine serverseitige Speicherung: Beim Neuladen beginnt der Chat neu.
    return jsonify({
        "chat_history": [],
        "study_day": 1,
        "max_study_day": MAX_STUDY_DAY,
        "chat_started_at": None,
        "duration_seconds": CONVERSATION_DURATION_SECONDS,
        "pause_seconds": DAY_SWITCH_PAUSE_SECONDS,
        "elapsed_seconds": 0,
        "conversation_closed_at": None,
        "time_limit_reached": False,
        "expired": False,
        "next_day_unlocked": False
    })


@app.route("/start_chat", methods=["POST"])
def start_chat():
    data = request.get_json(silent=True) or {}
    chat_history = clean_history(data.get("chat_history", []))
    study_day = int(data.get("study_day") or get_active_study_day(chat_history))
    study_day = max(1, min(study_day, MAX_STUDY_DAY))

    day_history = get_day_history(chat_history, study_day)
    if day_history:
        return jsonify({
            "already_started": True,
            "reply": None,
            "chat_history": chat_history,
            **timer_payload(chat_history, study_day)
        })

    now = utc_now_iso()
    reply = get_initial_assistant_message(study_day, chat_history)
    chat_history.append({
        "role": "assistant",
        "content": reply,
        "timestamp": now,
        "chat_started_at": now,
        "study_day": study_day
    })

    return jsonify({
        "already_started": False,
        "reply": reply,
        "chat_history": chat_history,
        **timer_payload(chat_history, study_day)
    })


@app.route("/send", methods=["POST"])
def send():
    data = request.get_json(silent=True) or {}
    user_message = str(data.get("message", "")).strip()
    chat_history = clean_history(data.get("chat_history", []))
    study_day = int(data.get("study_day") or get_active_study_day(chat_history))
    study_day = max(1, min(study_day, MAX_STUDY_DAY))

    if not user_message:
        return jsonify({"error": "Leere Nachricht"}), 400

    try:
        day_history = get_day_history(chat_history, study_day)

        if chat_is_closed(day_history):
            return jsonify({
                "error": "Das Gespräch für diesen Tag ist bereits beendet. Das nächste Gesprächsthema öffnet sich nach der kurzen Pause automatisch.",
                "chat_history": chat_history,
                **timer_payload(chat_history, study_day)
            }), 409

        now = utc_now_iso()
        chat_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": now,
            "study_day": study_day
        })

        day_history = get_day_history(chat_history, study_day)

        if chat_time_limit_reached(day_history):
            reply = get_closing_assistant_message(study_day)
            closed_at = utc_now_iso()
            chat_history.append({
                "role": "assistant",
                "content": reply,
                "timestamp": closed_at,
                "conversation_closed_at": closed_at,
                "is_closing_message": True,
                "study_day": study_day
            })

            return jsonify({
                "reply": reply,
                "chat_history": chat_history,
                **timer_payload(chat_history, study_day)
            })

        reply = ask_mistral(chat_history, study_day=study_day)
        now = utc_now_iso()
        chat_history.append({
            "role": "assistant",
            "content": reply,
            "timestamp": now,
            "study_day": study_day
        })

        return jsonify({
            "reply": reply,
            "chat_history": chat_history,
            **timer_payload(chat_history, study_day)
        })

    except Exception as e:
        print("Fehler:", repr(e))
        return jsonify({"error": str(e), "chat_history": chat_history}), 500


@app.route("/healthz")
def healthz():
    return "ok", 200


@app.route("/test_models")
def test_models():
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"}
    response = requests.get(
        "https://ki-chat.uni-mainz.de/api/models",
        headers=headers,
        timeout=30
    )

    try:
        result = response.json()
    except Exception:
        result = response.text

    return jsonify({
        "status_code": response.status_code,
        "data": result
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
