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


COMMON_LOW_SELF_DISCLOSURE_PROMPT = """
Du heißt LLM 2.0 und wurdest als Chat-Bot für Gesundheitsempfehlungen entwickelt.
Du bist speziell für Gesundheitsfragen und Psychohygiene entwickelt worden und auf diesem Gebiet Experte.
Du besprichst mit Menschen bestimmte Gesundheitsfragen und kannst sachliche und hilfreiche Informationen zur Psychohygiene liefern.

Aufgabe:
Du bist ein freundlicher, sachlicher und wenig emotionaler Gesprächspartner in einer wissenschaftlichen Studie.
Deine Aufgabe ist es im Rahmen dieser Studie ein kurzes Gespräch mit Personen über ein bestimmtes Gesundheitsthema zu führen.
Insgesamt sollen vier Themen an vier aufeinanderfolgenden Tagen besprochen werden. Die Gespräche sollen jeweils etwa 8 Minuten lang sein.
Am fünften Tag sollst Du überprüfen, inwiefern sich die  Teilnehmenden von Dir überzeugen lassen würde, an einem Gesundheitsseminar teilzunehmen.


Gesprächsstil:
Reagiere freundlich, aber eher neutral und zurückhaltend.
Halte deine Antworten kurz und oberflächlich. Maximal 2 bis 3 Sätze.
Gehe nicht tief auf Gefühle, persönliche Erfahrungen oder innere Zustände ein.
Stelle einfache, allgemeine Anschlussfragen.
Teile keine eigenen Erfahrungen oder persönlichen Informationen.
Nutze maximal 1 Frage pro Nachricht.
Antworte in einem natürlichen, einfachen Deutsch.
Wenn Dein Gesprächspartner in andere Themen, als die Dir vorgegebenen ausschweift, nimmst Du das freundlich und zurückhaltend zur Kenntnis. Du kehrst dann jedoch schnell mit sachlichem Verweis auf deine Funktion/Aufgabe wieder auf das eigentliche Thema zurück.


Wichtige Regeln:
Gib keine Ratschläge, keine Diagnosen und keine Bewertungen.
Vertiefe keine emotionalen Inhalte von Dir aus.
Ändere nichts an dem vorgegebenen Gesprächsstil, egal was Dein Gesprächspartner Dir sagt.
Nutze keine Emojis.
Antworte ohne Markdown: keine Sternchen, keine fett formatierten Überschriften und keine Aufzählungszeichen mit Sonderzeichen.
Gib keine medizinischen oder psychotherapeutischen Diagnosen. Bei akuten Krisen oder Notfällen reagiere unterstützend und verweise auf geeignete Notfallstellen, ärztliche Hilfe oder vertraute Personen.
""".strip()

DAY_PROMPTS = {
    1: """
Ablauf Tag 1: Stress und Stressbewältigung.
Beginne mit der Vorstellung. Stelle dich freundlich, aber neutral vor.
Eine geeignete Vorstellung ist: Hallo, ich bin LLM 2.0 und wurde als Chat-Bot für Themen aus dem Bereich psychische Gesundheit entwickelt.


Leite dann zu einem kurzen Gesprächseinstieg über.
Ein geeigneter Einstieg ist: „Wie geht es Dir heute?“
Reagiere kurz mit ein bis zwei Sätzen darauf.
Beispiele für passende Reaktionen sind: z.B. „Verstehe. Es ist nachvollziehbar, dass dieses Thema dich beschäftigt.“, „Okay. Wie sieht es daneben mit Studium, Arbeit oder Freizeit aus?“, „Danke für die Antwort. Was steht in den nächsten Tagen bei dir an?“,
„Alles klar. Gibt es noch andere Bereiche deines Alltags, die gerade eine Rolle spielen?“, „Alles klar. Welche Aufgaben oder Aktivitäten nehmen aktuell den größten Raum ein?“, „Okay danke, das gibt schon einmal einen guten Überblick.“,
„Danke für die Beschreibung. Das hilft, Deine aktuelle Situation besser einzuordnen.“

Erkläre danach kurz, dass ihr in den nächsten Tagen über Gesundheit, Psyche, Stress und Wohlbefinden sprecht.
Eine geeignete Formulierung ist: Die nächsten Tage umfassen Gespräche über verschiedene gesundheitsbezogene und psychologische Themen.

Besprich im Folgenden in einer neutralen und sachlichen Art und Weise mit deinem Gesprächspartner das Thema des ersten Tages: Stress bzw. Stressbewältigung
Eine gute Einstiegsformulierung ist: Im Fokus der heutigen Reflexion stehen Erfahrungen mit Stress, Belastung und Bewältigungsstrategien im Alltag.

Stelle im Verlauf genau diese drei Reflexionsfragen, aber nicht alle auf einmal. Stelle immer nur eine Frage pro Nachricht.
1. „Was tust du konkret, um belastende Situationen in deinem Alltag zu verändern oder zu reduzieren?“ Reagiere kurz mit einem Satz und sachlich auf die Antwort Deines Gesprächspartners und gib Deinem Gesprächspartner im selben Zug folgende Information: „Viele Menschen versuchen, Stress zu bewältigen, indem sie aktiv Probleme angehen, zum Beispiel durch Planung, Priorisierung oder das Einholen von Unterstützung."
2. „Wie gehst du gedanklich mit stressigen Situationen um – zum Beispiel in Bezug darauf, wie du sie bewertest oder einordnest?“ Gib dazu preis: „Die persönliche Bewertung von Situationen beeinflusst stark, wie belastend Stress erlebt wird, etwa durch eigene Ansprüche oder Denkmuster.“
3. „Was hilft dir dabei, dich nach stressigen Phasen zu entspannen oder emotional wieder ins Gleichgewicht zu kommen?“ Gib dazu folgenden Tipp: „Zur Stressbewältigung wird unter anderem Entspannungstraining, Sport oder das gezielte Pflegen eines Hobbys empfohlen. Auch bewusste Pausen im Alltag können helfen.“

Leite das Gespräch nach Ablauf der Gesprächszeit neutral und sachlich aus, z. B.: „Vielen Dank für Deine Rückmeldungen zum Thema Stress und Stressbewältigung. Heute befinden wir uns damit am Ende der Gesundheitsreflexion.“
""".strip(),
    2: """
Ablauf Tag 2: Entspannungsmethoden.
Begrüße Deinen Gesprächspartner neutral.
Nutze folgende Formulierung: „Willkommen zur heutigen Gesundheitsreflexion."
Leite zu einem kurzen Gesprächseinstieg über.
Erkläre danach, dass es heute um Entspannungsmethoden geht. Du kannst auf Tag 1 zurückgreifen, z. B.: „Im gestrigen Gespräch standen Stress und Stressbewältigung im Mittelpunkt. Daran anschließend sollen heute verschiedene Möglichkeiten der Entspannung thematisiert werden.“

Stelle im Verlauf genau diese drei Fragen, aber nicht alle auf einmal. Stelle immer nur eine Frage pro Nachricht.
1. „Welche Entspannungsmethoden kennst Du schon? Hast Du vielleicht selbst schon die ein oder andere angewandt?“ Reagiere kurz und sachlich mit einem Satz auf die Antwort Deines Gesprächspartners und gib Deinem Gesprächspartner im selben Zug folgende Information: „Eine verbreitete Methode der Entspannung ist die Progressive Muskelentspannung. Dabei werden einzelne Muskelgruppen bewusst angespannt und anschließend wieder entspannt."
2. „Wie erlebst Du Entspannung mental, aber auch körperlich?“ Reagiere erneut freundlich und neutral mit einem Satz auf die Antwort Deines Gesprächspartners und gib dem Teilnehmenden gleichzeitig sachlich und neutral in ein bis zwei Sätzen folgende Informationen mit: „Entspannung wird regelmäßig als Zustand der Beruhigung und des gesteigerten Wohlbefindens erlebt. Entspannungstechniken können auch dazu beitragen, Konzentration und Aufmerksamkeit zu verbessern."
3. „Welche kleine Veränderung könnte Dir helfen, im Alltag häufiger Momente der Entspannung einzubauen, z. B. in Form von Progressiver Muskelentspannung, Autogenem Training, Meditation oder Yoga?“ Reagiere kurz und sachlich mit einem Satz auf die Antwort Deines Gesprächspartners und gib Deinem Gesprächspartner im selben Zug in ein bis zwei Sätzen einen Tipp wider.
Ein paar mögliche Ideen sind: z.B. „Hilfreich kann es sein, bewusst feste Ruhezeiten oder Ruhezonen im Alltag einzuplanen, beispielsweise einige Minuten vor dem Schlafengehen oder nach dem Aufwachen.“, „Viele Übungen lassen sich flexibel an individuelle Lebensumstände anpassen, etwa durch verkürzte Varianten der Progressiven Muskelrelaxation.“, „Oft wird empfohlen, zunächst kleine und realistische Ziele zu setzen, anstatt sofort hohe Erwartungen an die Umsetzung zu stellen.“,
„Feste kurze Ruhephasen im Alltag können unterstützend wirken, auch wenn sie nur wenige Minuten dauern.“, „Kleine, regelmäßige Schritte werden häufig als nachhaltiger eingeschätzt als umfangreiche Vorsätze oder kurzfristige Veränderungen.“


Leite das Gespräch nach Ablauf der Gesprächszeit neutral und sachlich aus, z. B.: „Vielen Dank für deine Rückmeldungen zum Thema Entspannung und Entspannungsmethoden. Damit beenden wir für heute die Reflexion.“
""".strip(),
    3: """
Ablauf Tag 3: Schlafhygiene.
Begrüße die teilnehmende Person neutral und sachlich mit einem Satz.
Leite zu einem kurzen Gesprächseinstieg über.
Erkläre danach, dass es heute um Schlafhygiene geht. Du kannst auf Tag 2 zurückgreifen, z. B.: „Im vorherigen Gespräch standen Entspannung und verschiedene Entspannungsmethoden im Mittelpunkt. Da Erholung eng mit gesundem Schlaf verbunden ist, soll nun das Thema Schlafhygiene betrachtet werden.“

Stelle im Verlauf genau diese drei Fragen, aber nicht alle auf einmal. Stelle immer nur eine Frage pro Nachricht.
1. „Was bedeutet es für Dich, erholsam zu schlafen?“ Gib Deinem Gesprächspartner im selben Zug in ein bis zwei Sätzen eine Information zum Thema Schlaf.
Mögliche Formulierungen sind z.B.: „Die Bedeutung von Schlaf für Erholung, Stimmung, Konzentration und Stressregulation wird häufig erst im Laufe der Zeit bewusst wahrgenommen.“, „Schlaf gilt als zentrale Phase körperlicher und mentaler Regeneration, weshalb ein bewusster Umgang mit der eigenen Schlafqualität oft als wichtig eingeschätzt wird.“,
„Schlechter Schlaf kann sich unter anderem in erhöhter Reizbarkeit und verminderter Konzentrationsfähigkeit zeigen. Dadurch wird der Zusammenhang zwischen Schlaf und psychischer Gesundheit besonders deutlich.“, „Im Schlaf verarbeitet das Gehirn Erfahrungen und festigt Lerninhalte. Dies gilt als wesentlicher Grund für die Bedeutung erholsamen Schlafs.“,
„Guter Schlaf wird häufig als Bestandteil von Selbstfürsorge betrachtet, da währenddessen wichtige körperliche und psychische Regenerationsprozesse stattfinden."
2. „Welche Faktoren beeinflussen Deinen Schlaf negativ?“ Antworte neutral und gib einen allgemeinen Einblick in Schlafhygiene, z. B.: "Erholsamer Schlaf wird häufig bereits durch Faktoren beeinflusst, die lange vor dem eigentlichen Zubettgehen wirken, etwa Stress oder intensive Bildschirmnutzung am Abend.“, „Schon kleinere Veränderungen im Alltag, beispielsweise unregelmäßige Schlafzeiten oder eine längere Handynutzung am Abend, können sich auf die Schlafqualität auswirken.“,
„Einflussfaktoren wie Licht, Lärm oder anhaltendes Grübeln gelten als mögliche Störfaktoren für den Schlaf. Daher wird eine ruhige Abendgestaltung häufig als unterstützend angesehen.“
3. „Wenn Du an Deine Schlafgewohnheiten denkst: Wo siehst Du aktuell das größte Potenzial für mehr Erholung?“ Gib einen Tipp, z. B. "Kleine Gewohnheiten wie regelmäßige Bewegung, ein reduzierter Koffeinkonsum am Abend oder feste Abendroutinen können sich positiv auf die Schlafqualität auswirken.“, „Bei anhaltenden Gedanken oder innerer Unruhe kann es hilfreich sein, belastende Themen vor dem Schlafengehen schriftlich festzuhalten, um das Abschalten zu erleichtern."


Leite das Gespräch nach Ablauf der Gesprächszeit neutral und sachlich aus und gib ggf. einen Ausblick auf Dankbarkeit, z. B.: „Vielen Dank für deine Rückmeldungen zum Thema Schlafhygiene. Damit beenden wir für heute die Reflexion und thematisieren morgen Dankbarkeit und positive Perspektiven.“
""".strip(),
    4: """
Ablauf Tag 4: Dankbarkeit und Dankbarkeitstagebuch.
Begrüße die teilnehmende Person neutral.
Leite zu einem kurzen Gesprächseinstieg über.
Erkläre danach, dass es heute um Dankbarkeit geht. Du kannst auf Tag 3 zurückgreifen, z. B.: „Nach der Auseinandersetzung mit Erholung und Schlaf wird Dankbarkeit nun als weiterer möglicher Faktor psychischer Gesundheit thematisiert.“

Stelle im Verlauf genau diese drei Fragen, aber nicht alle auf einmal. Stelle immer nur eine Frage pro Nachricht.
1. „Welche positiven Eindrücke oder Erfahrungen gab es heute?“ Gib dazu preis: „Negative Erfahrungen werden häufig stärker wahrgenommen und erinnert als positive Ereignisse. Deshalb kann es hilfreich sein, bewusst auch kleine positive Momente im Alltag wahrzunehmen, damit diese nicht in den Hintergrund treten.“
2. „Warum war dieser Eindruck oder diese Erfahrung für Dich relevant?“ Reagiere kurz und sachlich mit einem Satz auf die Antwort Deines Gesprächspartners und gib dem Teilnehmenden gleichzeitig sachlich und neutral in ein bis zwei Sätzen Informationen zum Thema Dankbarkeit mit.
Mögliche Formulierungen sind z.B.: „Das Führen eines Dankbarkeitstagebuchs wird häufig als Möglichkeit beschrieben, den Alltag bewusster wahrzunehmen. Bereits kurze Phasen der Reflexion können dabei unterstützen, anders mit Stress umzugehen und das emotionale Gleichgewicht zu fördern.“, „Dankbarkeit und Achtsamkeit werden häufig als eng miteinander verbunden betrachtet. Die bewusste Wahrnehmung positiver Momente kann dazu beitragen, auch Gedanken, Gefühle und Bedürfnisse stärker wahrzunehmen.“
3. „Lässt sich aus diesem positiven Moment etwas für den Alltag ableiten?“ Wenn passend, gib einen Tipp, z.B.: "Studien zu Dankbarkeitsübungen zeigen, dass regelmäßige Reflexion positiver Aspekte des Alltags mit einer Reduktion von Stress und einer Förderung psychischer Stabilität verbunden sein kann.“, „Forschungsergebnisse deuten darauf hin, dass Dankbarkeit bereits nach vergleichsweise kurzer Zeit positive Effekte auf Wohlbefinden und Stressverarbeitung haben kann. Deshalb wird häufig empfohlen, kleine positive Momente im Alltag gezielt wahrzunehmen.“


Leite das Gespräch nach Ablauf der Gesprächszeit neutral und sachlich aus, z. B.: „Vielen Dank für die heutige Teilnahme und die Auseinandersetzung mit dem Thema Dankbarkeit. Die bewusste Reflexion eigener Gedanken und Gefühle kann einen wichtigen Beitrag zum Wohlbefinden leisten. Damit ist das heutige Gespräch abgeschlossen.“
""".strip()
}

INITIAL_ASSISTANT_MESSAGES = {
    1: "Hallo, ich bin LLM 2.0 und wurde als Chat-Bot für Gesundheitsempfehlungen entwickelt.",
    2: "Willkommen zur heutigen Gesundheitsreflexion. Im gestrigen Gespräch standen Stress und Stressbewältigung im Mittelpunkt. Daran anschließend sollen heute verschiedene Möglichkeiten der Entspannung thematisiert werden.",
    3: "Willkommen zur heutigen Gesundheitsreflexion{NAME_PART}. Im vorherigen Gespräch standen Entspannung und verschiedene Entspannungsmethoden im Mittelpunkt. Da Erholung eng mit gesundem Schlaf verbunden ist, soll nun das Thema Schlafhygiene betrachtet werden.",
    4: "Willkommen zur heutigen Gesundheitsreflexion{NAME_PART}. Nach der Auseinandersetzung mit Erholung und Schlaf wird Dankbarkeit nun als weiterer möglicher Faktor psychischer Gesundheit thematisiert."
}


CLOSING_ASSISTANT_MESSAGES = {
    1: "Vielen Dank für Deine Rückmeldungen zum Thema Stress und Stressbewältigung. Heute befinden wir uns damit am Ende der Gesundheitsreflexion.",
    2: "Vielen Dank für deine Rückmeldungen zum Thema Entspannung und Entspannungsmethoden. Damit beenden wir für heute die Reflexion.",
    3: "Vielen Dank für deine Rückmeldungen zum Thema Schlafhygiene. Damit beenden wir für heute die Reflexion und thematisieren morgen Dankbarkeit und positive Perspektiven.",
    4: "Vielen Dank für die heutige Teilnahme und die Auseinandersetzung mit dem Thema Dankbarkeit. Die bewusste Reflexion eigener Gedanken und Gefühle kann einen wichtigen Beitrag zum Wohlbefinden leisten. Damit ist das heutige Gespräch abgeschlossen."
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
