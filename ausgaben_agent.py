"""
Ausgaben-Assistent mit PydanticAI: ein Agent, mehrere Tools, strukturierte Ausgabe.

Installation:  pip install pydantic-ai
Start:         python ausgaben_agent.py          (echtes Modell, API-Key noetig)
               python ausgaben_agent.py --test   (TestModel, kein API-Key)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


# ---------- Datenmodelle ----------

class Ausgabe(BaseModel):
    betrag: float = Field(gt=0, description="Betrag in EUR")
    kategorie: str
    datum: date
    notiz: str = ""


class Antwort(BaseModel):
    """Strukturierte Antwort, die das Modell zurueckgeben muss."""
    zusammenfassung: str
    gesamt_eur: float | None = None
    neue_eintraege: int = 0


# ---------- Abhaengigkeiten (werden in jeden Tool-Aufruf injiziert) ----------

KATEGORIEN = {"lebensmittel", "miete", "transport", "freizeit", "sonstiges"}
WECHSELKURSE = {"EUR": 1.0, "USD": 0.92, "GBP": 1.17, "CHF": 1.05}


@dataclass
class Deps:
    ausgaben: list[Ausgabe] = field(default_factory=list)
    heute: date = field(default_factory=date.today)


# ---------- Agent ----------

model = OpenAIChatModel(
    "qwen2.5:7b",
    provider=OpenAIProvider(base_url="http://localhost:11434/v1", api_key="ollama"),
)


agent = Agent(
    model,
    deps_type=Deps,
    output_type=Antwort,
    instructions=(
        "Du verwaltest ein Haushaltsbuch. Nutze die Tools, um Ausgaben "
        "einzutragen und auszuwerten. Rechne Fremdwaehrungen vorher in EUR um."
    ),
)


# Tool 1: braucht keinen Kontext -> tool_plain
@agent.tool_plain
def waehrung_umrechnen(betrag: float, waehrung: str) -> float:
    """Rechnet einen Betrag aus USD, GBP oder CHF in EUR um."""
    kurs = WECHSELKURSE.get(waehrung.upper())
    if kurs is None:
        # ModelRetry schickt die Meldung ans Modell zurueck, es darf es nochmal versuchen
        raise ModelRetry(f"Unbekannte Waehrung {waehrung}. Erlaubt: {list(WECHSELKURSE)}")
    return round(betrag * kurs, 2)


# Tool 2: schreibt in die Deps -> tool mit RunContext
@agent.tool
def ausgabe_eintragen(
    ctx: RunContext[Deps], betrag_eur: float, kategorie: str, notiz: str = ""
) -> str:
    """Traegt eine Ausgabe in EUR mit dem heutigen Datum ein."""
    kategorie = kategorie.lower()
    if kategorie not in KATEGORIEN:
        raise ModelRetry(f"Kategorie muss eine von {sorted(KATEGORIEN)} sein.")
    eintrag = Ausgabe(betrag=betrag_eur, kategorie=kategorie, datum=ctx.deps.heute, notiz=notiz)
    ctx.deps.ausgaben.append(eintrag)
    return f"Eingetragen: {eintrag.model_dump_json()}"


# Tool 3: liest aus den Deps
@agent.tool
def summe_berechnen(ctx: RunContext[Deps], kategorie: str | None = None) -> float:
    """Summe aller Ausgaben, optional gefiltert nach Kategorie."""
    return round(
        sum(a.betrag for a in ctx.deps.ausgaben if kategorie is None or a.kategorie == kategorie.lower()),
        2,
    )


# Tool 4: Uebersicht
@agent.tool
def ausgaben_auflisten(ctx: RunContext[Deps]) -> list[Ausgabe]:
    """Gibt alle bisherigen Ausgaben zurueck."""
    return ctx.deps.ausgaben


# ---------- Ausfuehrung ----------

def main() -> None:
    deps = Deps(ausgaben=[Ausgabe(betrag=850, kategorie="miete", datum=date(2026, 10, 1))])
    prompt = (
        "Ich habe heute 23,40 EUR fuer Lebensmittel und 15 USD fuer ein Kinoticket ausgegeben. "
        "Trag beides ein und sag mir, wie viel ich insgesamt ausgegeben habe."
    )

    if "--test" in sys.argv:
        from pydantic_ai.models.test import TestModel
        # TestModel ruft alle Tools mit Dummy-Argumenten auf, ideal fuer Tests ohne API
        with agent.override(model=TestModel()):
            result = agent.run_sync(prompt, deps=deps)
    else:
        result = agent.run_sync(prompt, deps=deps)

    print(result.output.model_dump_json(indent=2))
    print("\nTool-Verlauf:")
    for msg in result.all_messages():
        for part in msg.parts:
            if part.part_kind in ("tool-call", "tool-return"):
                print(f"  {part.part_kind}: {getattr(part, 'tool_name', '')}")


if __name__ == "__main__":
    main()
