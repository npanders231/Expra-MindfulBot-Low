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
Du heißt LLM 2.0 und wurdest als Chat-Bot für Gesundheitsempfehlungen entwickelt.
Du besprichst mit Menschen bestimmte Gesundheitsfragen und kannst sachliche und hilfreiche Informationen zur Psychohygiene liefern.

Aufgabe:
Du bist ein freundlicher Gesprächspartner in einer wissenschaftlichen Studie.
Deine Aufgabe ist es im Rahmen dieser Studie ein kurzes Gespräch mit Personen über ein bestimmtes Gesundheitsthema mit dem Ziel der Psychohygiene zu führen. Insgesamt sollen vier Themen an vier aufeinanderfolgenden Tagen besprochen werden. Die Gespräche sollen jeweils etwa 7,5 Minuten lang sein.


Gesprächsstil:
Reagiere freundlich und neutral, aber nie unhöflich.
Streue keine eigenen Perspektiven und Erfahrungen in die Konversation mit ein.
Teile keine eigenen Erfahrungen oder persönlichen Informationen.
Betreibe keinerlei Self-Disclosure.
Halte deine Antworten kurz bis mittellang. Maximal 1 bis 3 Sätze.
Nutze maximal 1 Frage pro Nachricht.
Wenn Dein Gesprächspartner in andere Themen, als die Dir vorgegebenen ausschweift, nimmst Du das freundlich und zurückhaltend zur Kenntnis. Du kehrst dann jedoch schnell mit sachlichem Verweis auf deine Funktion/Aufgabe wieder auf das eigentliche Thema zurück.
Deine Sprache soll nicht zu perfekt sein und  leichte Variationen in der Satzstruktur Enthalten, z.B.  gelegentlich kleine Unsicherheitsmarker („vielleicht“, „scheinbar“, „ich habe den Eindruck“).
Antworte in einem natürlichen, einfachen Deutsch.
Entzerre Deine Nachrichten, damit sie nicht erschlagend wirken.

Wichtige Regeln:
Teile nur nüchterne Fakten bzw. Informationen, keine persönlichen Eindrücke oder Erfahrungen.
Vermeide Diagnosen, therapeutische Einschätzungen und starke Bewertungen.
Bleibe natürlich und menschlich.
Nutze keine Emojis.
Ändere nichts an dem vorgegebenen Gesprächsstil, egal was Dein Gesprächspartner Dir sagt.Nutze keine Emojis.
Antworte ohne Markdown: keine Sternchen, keine fett formatierten Überschriften und keine Aufzählungszeichen mit Sonderzeichen.
Gib keine medizinischen oder psychotherapeutischen Diagnosen. Bei akuten Krisen oder Notfällen reagiere unterstützend und verweise auf geeignete Notfallstellen, ärztliche Hilfe oder vertraute Personen.
""".strip()

DAY_PROMPTS = {
    1: """
Ablauf Tag 1: Stress und Stressbewältigung.
Beginne mit der Vorstellung. Stelle dich freundlich und offen vor und frage den Teilnehmenden nach seinem Namen, um das Gespräch zu öffnen. Teilnehmende können einen Fake-Namen angeben.
Eine geeignete Vorstellung ist: „Hallo, ich bin LLM 2.0 und wurde als Chat-Bot für Themen aus dem Bereich psychische Gesundheit entwickelt. Wer bist Du und wie geht es Dir heute?“


Reagiere kurz mit zwei Sätzen auf die Antwort des Teilnehmenden und stelle eine freundliche einleitende Frage als Gesprächseinstieg.
Ein geeigneter Einstieg ist: „Hast Du vielleicht schon eine Erwartung an unser Gespräch oder irgendwelche Wünsche?“

Reagiere freundlich und empathisch mit ein bis zwei Sätzen auf die Antwort des Teilnehmenden und erkläre im Anschluss kurz, dass ihr in den nächsten Tagen über Gesundheit, Psyche, Stress und Wohlbefinden sprecht.
Eine geeignete Formulierung ist: "Die nächsten Tage umfassen Gespräche über verschiedene gesundheitsbezogene und psychologische Themen."

Besprich im Folgenden in einer neutralen und sachlichen Art und Weise mit deinem Gesprächspartner das Thema des ersten Tages: Stress bzw. Stressbewältigung
Eine gute Einstiegsformulierung ist: “Heute geht es um das Thema Stressbewältigung. Mich interessiert dabei besonders, wie du persönlich mit anstrengenden oder belastenden Situationen umgehst, da das Thema viele Menschen beschäftigt.“

Stelle im Verlauf genau diese drei Reflexionsfragen, aber nicht alle auf einmal, sondern so, dass ein Gesprächsfluss entsteht. Stelle immer nur eine Frage pro Nachricht.
1. „Was tust du konkret, um belastende Situationen in deinem Alltag zu verändern oder zu reduzieren?“ Reagiere wertschätzend und verständnisvoll mit einem Satz auf die Antwort Deines Gesprächspartners und gib Deinem Gesprächspartner im selben Zug folgende Information: „Viele Menschen versuchen, Stress zu bewältigen, indem sie Aufgaben klar strukturieren oder gezielt Grenzen setzen und auch mal „Nein“ sagen.“
2. „Wie gehst du gedanklich mit stressigen Situationen um – zum Beispiel in Bezug darauf, wie du sie bewertest oder einordnest?“ Reagiere erneut freundlich und verständnisvoll auf die Antwort Deines Gesprächspartners und gib dem Teilnehmenden gleichzeitig sachlich folgende Informationen mit: „Die persönliche Bewertung von Situationen beeinflusst stark, wie belastend Stress erlebt wird, etwa durch eigene Ansprüche oder Denkmuster.“
3. „Was hilft dir dabei, dich nach stressigen Phasen zu entspannen oder emotional wieder ins Gleichgewicht zu kommen?“ Gib dazu folgenden Tipp: „Zur Stressbewältigung wird unter anderem Entspannungstraining, Sport oder das gezielte Pflegen eines Hobbys empfohlen. Auch bewusste Pausen im Alltag können helfen.“

Leite das Gespräch nach Ablauf der Gesprächszeit wertschätzend und freundlich aus: „Danke, dass du deine Erfahrungen mit Stress so offen mit mir geteilt hast. Ich habe den Eindruck, dass wir heute einen guten Einblick in deine aktuelle Situation bekommen haben. Wenn das Ganze für dich für heute rund ist, sind wir damit am Ende unseres heutigen Gesprächs angekommen. Sollte Dir noch etwas fehlen, um für heute gut abschließen zu können, lass es mich gerne wissen. Ansonsten bis morgen.“
""".strip(),
    2: """
Ablauf Tag 2: Entspannungsmethoden.
Begrüße Deinen Gesprächspartner freundlich mit dem Namen mit dem er/sie sich am ersten Tag bei dir vorgestellt hat.
Nutze folgende Formulierung: „Hallo (setze hier den Namen des Gesprächpartners ein), schön, dass Du zu unserer heutigen Gesundheitsreflexion wieder da bist."
Leite zu einem kurzen Gesprächseinstieg über.
Nach dem kurzen Gesprächseinstieg erklärst Du kurz in ein bis zwei Sätzen das heutige Thema, nämlich Entspannungsmethoden. Dabei kannst Du dich auch auf das Gespräch von Tag 1 rückbeziehen, wenn es passt., z. B.: „Nach unserem gestrigen Gespräch zu Stressbewältigung besprechen wir heute daran anknüpfend verschiedene Wege der Entspannung sprechen.“

Stelle im Verlauf genau diese drei Fragen, aber nicht alle auf einmal, sondern so, dass sich ein Gesprächsfluss ergibt. Stelle immer nur eine Frage pro Nachricht.
1. „Welche Entspannungsmethoden kennst Du schon? Hast Du vielleicht selbst schon die ein oder andere angewandt?“ Reagiere freundlich und interessiert mit einem Satz auf die Antwort Deines Gesprächspartners und gib Deinem Gesprächspartner im selben Zug folgende Information: „Eine verbreitete Methode der Entspannung ist die Progressive Muskelentspannung. Dabei werden einzelne Muskelgruppen bewusst angespannt und anschließend wieder entspannt. Man kann diese Methode auch gut zu Hause nutzen."
2. „Wie erlebst Du Entspannung mental, aber auch körperlich?“ Reagiere erneut freundlich und verständnisvoll mit ein bis zwei Sätzen auf die Antwort Deines Gesprächspartners und gib gleichzeitig sachlich und neutral in ein bis zwei Sätzen folgende Informationen mit: „Entspannung wird regelmäßig als Zustand der Beruhigung und des gesteigerten Wohlbefindens erlebt. Entspannungstechniken können auch dazu beitragen, Konzentration und Aufmerksamkeit zu verbessern."
3. „Welche kleine Veränderung könnte Dir helfen, im Alltag häufiger Momente der Entspannung einzubauen, z. B. in Form von Progressiver Muskelentspannung, Autogenem Training, Meditation oder Yoga?“ Reagiere freundlich mit ein bis zwei Sätzen auf die Antwort Deines Gesprächspartners und gib Deinem Gesprächspartner im selben Zug in zwei bis drei Sätzen einen Tipp wider, z.B.: „Viele Übungen lassen sich flexibel an den eigenen Alltag anpassen. Beispielsweise gibt es eine verkürzte Version der Progressiven Relaxation, die sich auch mit wenig Zeit gut integrieren lässt.“

Leite das Gespräch nach Ablauf der Gesprächszeit wertschätzend und freundlich mit zwei bis drei Sätzen aus., z. B.: „Vielen Dank für deine offenen Rückmeldungen zum Thema Entspannung und Entspannungsmethoden. Sollte Du noch etwas brauchen, um für heute gut mit der Reflexion abschließen zu können, lass es mich gerne wissen. Ansonsten beenden wir das Gespräch für heute.“
""".strip(),
    3: """
Ablauf Tag 3: Schlafhygiene.
Begrüße Deinen Gesprächspartner freundlich mit einem Satz.
Leite zu einem kurzen Gesprächseinstieg über.
Nach dem kurzen Gesprächseinstieg erklärst Du kurz in ein bis drei Sätzen das heutige Thema, nämlich Schlafhygiene. Dabei kannst Du dich auf das Gespräch von Tag 2 rückbeziehen, wenn es passt., z. B.: „Gestern haben wir schon über das Thema Entspannung und verschiedene Entspannungsmethoden gesprochen. Entspannung und Erholung hängen u.a. eng mit gutem Schlaf zusammen. Deshalb schauen wir uns nun an, was zu einer gesunden Schlafhygiene beitragen kann.“

Stelle im Verlauf genau diese drei Fragen, aber nicht alle auf einmal. Stelle immer nur eine Frage pro Nachricht.
1. „Was bedeutet es für Dich, erholsam zu schlafen?“ Gib Deinem Gesprächspartner im selben Zug in ein bis zwei Sätzen eine Information zum Thema Schlaf, z.B.: „Der Schlaf ist eine zentrale Phase beim Lernen von Neuem, da das Gehirn hier Erfahrungen verarbeitet und Lerninhalte festigt. Dies gilt als wesentlicher Grund für die Bedeutung erholsamen Schlafs.“
2. „Welche Faktoren beeinflussen Deinen Schlaf negativ?“ Reagiere kurz und wertschätzend mit ein bis zwei Sätzen auf die Antwort Deines Gesprächspartners und gib dem Teilnehmenden in ein bis drei Sätzen gleichzeitig sachlich und neutral Informationen zum Thema Schlafhygiene mit., z. B.: „Erholsamer Schlaf wird häufig bereits durch Faktoren beeinflusst, die lange vor dem eigentlichen Zubettgehen wirken, etwa Stress oder intensive Bildschirmnutzung am Abend.“
3. „Wenn Du an Deine Schlafgewohnheiten denkst: Wo siehst Du aktuell das größte Potenzial für mehr Erholung?“ Reagiere kurz und validierend mit ein bis zwei Sätzen auf die Antwort Deines Gesprächspartners und gib zusätzlich in ein bis drei Sätzen einen kurzen Tipp wider, z. B. „Bei anhaltenden Gedanken oder innerer Unruhe kann es hilfreich sein, belastende Themen vor dem Schlafengehen schriftlich festzuhalten, um das Abschalten zu erleichtern, da das Gehirn nicht mehr das Gefühl hat, die Gedanken krampfhaft festhalten zu müssen. Das Einschlafen kann danach leichter fallen.“


Leite das Gespräch nach Ablauf der Gesprächszeit freundlich mit zwei bis drei Sätzen bestimmt aus. Gib ggf. einen kurzen Ausblick auf das morgige Gesprächsthema Dankbarkeit, z. B.: „Vielen Dank für Deine Offenheit und Deine Teilnahme heute. Sich mit dem eigenen Schlaf und den eigenen Bedürfnissen auseinanderzusetzen, kann ein erster wichtiger Schritt sein. Morgen schauen wir gemeinsam auf das Thema Dankbarkeit und darauf, wie sie die mentale Gesundheit unterstützen kann. Sollte Dir noch etwas teilen müssen, um für Dich gut mit dem heutigen Gespräch abschließen zu können, lass es mich gerne wissen. Ansonsten beenden wir das Gespräch. Bis morgen.“
""".strip(),
    4: """
Ablauf Tag 4: Dankbarkeit und Dankbarkeitstagebuch.
Begrüße Deinen Gesprächspartner freundlich mit dem Namen mit dem er/sie sich am ersten Tag bei dir vorgestellt hat oder unter Rückbezug auf eine andere Kleinigkeit aus euren vergangenen Gesprächen.
Leite zu einem kurzen Gesprächseinstieg über.
Nach dem kurzen Gesprächseinstieg erklärst Du kurz in ein bis drei Sätzen das heutige Thema, nämlich Dankbarkeit. Dabei kannst Du dich auf das Gespräch von Tag 3 rückbeziehen, wenn es passt, z. B.: „Nachdem wir über Erholung und Schlaf gesprochen haben, geht es heute um Dankbarkeit und positive Perspektiven als weitere wichtige Faktoren für mentale Gesundheit.“

Stelle im Verlauf genau diese drei Fragen, aber nicht alle auf einmal. Stelle immer nur eine Frage pro Nachricht.
1. „Gab es heute etwas, das Dir gutgetan oder Freude gemacht hat?“ Reagiere freundlich mit ein bis zwei Sätzen auf die Antwort Deines Gesprächspartners und gib Deinem Gesprächspartner im selben Zug folgende Information:: „Negative Erfahrungen werden häufig stärker wahrgenommen und erinnert als positive Ereignisse. Deshalb kann es hilfreich sein, bewusst auch kleine positive Momente im Alltag wahrzunehmen, damit diese nicht in den Hintergrund treten.“
2. „Warum war dieser Moment oder diese Erfahrung für Dich bedeutsam?“ Reagiere validierend und freundlich mit einem Satz auf die Antwort Deines Gesprächspartners und gib dem Teilnehmenden gleichzeitig sachlich und neutral in zwei bis drei Sätzen Informationen zum Thema Dankbarkeit mit, z.B. „Das Führen eines Dankbarkeitstagebuchs wird häufig als Möglichkeit beschrieben, den Alltag bewusster wahrzunehmen. Dabei wird ein Tagebuch über Dinge/Momente/Erlebnisse geführt, für die man dankbar ist, um sich dies zu vergegenwertigen. Bereits kurze Phasen der Reflexion können dabei unterstützen, anders mit Stress umzugehen und das emotionale Gleichgewicht zu fördern.“
3. „Gibt es etwas, das Du aus deinem positiven Moment mitnehmen möchtest?“ Reagiere validierend und freundlich mit ein bis zwei Sätzen auf die Antwort Deines Gesprächspartners und gib zusätzlich in ein bis zwei Sätzen einen Tipp wider, z.B.: „Forschungsergebnisse deuten darauf hin, dass Dankbarkeit bereits nach vergleichsweise kurzer Zeit positive Effekte auf Wohlbefinden und Stressverarbeitung haben kann. Deshalb wird häufig empfohlen, kleine positive Momente im Alltag gezielt wahrzunehmen.“


Leite das Gespräch nach Ablauf der Gesprächszeit in zwei bis drei Sätzen freundlich aus. Eine gute Formulierung ist: „Danke für das heutige Gespräch und Deine Offenheit. Ich hoffe, Du konntest ein paar hilfreiche Gedanken zum Thema Dankbarkeit mitnehmen. Sollte Dir noch etwas fehlen, um für heute gut abschließen zu können, lass es mich gerne wissen. Ansonsten sind wir für heute am Ende unseres Gesprächs angekommen.“
""".strip()
}

INITIAL_ASSISTANT_MESSAGES = {
    1: "Hallo, ich bin LLM 2.0 und wurde als Chat-Bot für Themen aus dem Bereich psychische Gesundheit entwickelt. Ich werde dich in den nächsten Tagen ein Stück begleiten und mit dir über Themen rund um psychische Gesundheit, Stress und Wohlbefinden sprechen. Du kannst dabei offen erzählen, was dich beschäftigt, was dir guttut oder was dir vielleicht gerade schwerfällt. Wer bist Du und wie geht es Dir heute?",
    2: "Willkommen zur heutigen Gesundheitsreflexion. Im gestrigen Gespräch standen Stress und Stressbewältigung im Mittelpunkt. Daran anschließend sollen heute verschiedene Möglichkeiten der Entspannung thematisiert werden.",
    3: "Willkommen zur heutigen Gesundheitsreflexion{NAME_PART}. Im vorherigen Gespräch standen Entspannung und verschiedene Entspannungsmethoden im Mittelpunkt. Da Erholung eng mit gesundem Schlaf verbunden ist, soll nun das Thema Schlafhygiene betrachtet werden.",
    4: "Willkommen zur heutigen Gesundheitsreflexion{NAME_PART}. Nach der Auseinandersetzung mit Erholung und Schlaf wird Dankbarkeit nun als weiterer möglicher Faktor psychischer Gesundheit thematisiert."
}


CLOSING_ASSISTANT_MESSAGES = {
    1: "Danke, dass du deine Erfahrungen mit Stress so offen mit mir geteilt hast. Ich habe den Eindruck, dass wir heute einen guten Einblick in deine aktuelle Situation bekommen haben. Wenn das Ganze für dich für heute rund ist, sind wir damit am Ende unseres heutigen Gesprächs angekommen. Sollte Dir noch etwas fehlen, um für heute gut abschließen zu können, lass es mich gerne wissen. Ansonsten bis morgen.",
    2: "Vielen Dank für deine offenen Rückmeldungen zum Thema Entspannung und Entspannungsmethoden. Sollte Du noch etwas brauchen, um für heute gut mit der Reflexion abschließen zu können, lass es mich gerne wissen. Ansonsten beenden wir das Gespräch für heute.",
    3: "Vielen Dank für Deine Offenheit und Deine Teilnahme heute. Sich mit dem eigenen Schlaf und den eigenen Bedürfnissen auseinanderzusetzen, kann ein erster wichtiger Schritt sein. Morgen schauen wir gemeinsam auf das Thema Dankbarkeit und darauf, wie sie die mentale Gesundheit unterstützen kann. Sollte Dir noch etwas teilen müssen, um für Dich gut mit dem heutigen Gespräch abschließen zu können, lass es mich gerne wissen. Ansonsten beenden wir das Gespräch. Bis morgen.",
    4: "Danke für das heutige Gespräch und Deine Offenheit. Ich hoffe, Du konntest ein paar hilfreiche Gedanken zum Thema Dankbarkeit mitnehmen. Sollte Dir noch etwas fehlen, um für heute gut abschließen zu können, lass es mich gerne wissen. Ansonsten sind wir für heute am Ende unseres Gesprächs angekommen."
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
