import asyncio
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

import jev

_ = load_dotenv()


class Triage(BaseModel):
    department: Literal["billing", "technical", "sales"] = Field(
        description="Which team should handle this ticket"
    )
    is_urgent: bool = Field(
        description="The ticket conveys urgency or time-sensitivity"
    )
    frustration: int = Field(
        ge=0,
        le=2,
        description="How frustrated the customer appears",
        json_schema_extra={
            "levels": [
                "Calm, just stating facts",
                "Frustrated but civil",
                "Very angry, strong language",
            ]
        },
    )


@jev.fn
def triage(ticket: str) -> Triage:
    """A customer support ticket:

    {{ ticket }}
    """
    return triage.state()


@jev.fn
async def atriage(ticket: str) -> Triage:
    """A customer support ticket:

    {{ ticket }}
    """
    return atriage.state()


def main() -> None:
    ticket = (
        "Hi, I've been trying to connect my Stripe account for 3 days and it "
        "keeps failing. I'm losing sales. Please help ASAP."
    )
    print("sync: ", triage(ticket))
    print("async:", asyncio.run(atriage(ticket)))


if __name__ == "__main__":
    main()
