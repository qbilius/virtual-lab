"""Runs a meeting with LLM agents using local HuggingFace models."""

import time
from pathlib import Path
from typing import Literal, Optional

from transformers import pipeline, AutoTokenizer
from tqdm import trange, tqdm

from virtual_lab.agent import Agent
from virtual_lab.constants import CONSISTENT_TEMPERATURE
from virtual_lab.prompts import (
    individual_meeting_agent_prompt,
    individual_meeting_critic_prompt,
    individual_meeting_start_prompt,
    SCIENTIFIC_CRITIC,
    team_meeting_start_prompt,
    team_meeting_team_lead_initial_prompt,
    team_meeting_team_lead_intermediate_prompt,
    team_meeting_team_lead_final_prompt,
    team_meeting_team_member_prompt,
)
from virtual_lab.utils import count_tokens, save_meeting


# @dataclass
# class ModelConfig:
#     """Configuration for the local model."""
#     model_name: str
#     device: str = "cuda" if torch.cuda.is_available() else "cpu"
#     max_new_tokens: int = 1024
#     do_sample: bool = True
#     top_p: float = 0.9
#     top_k: int = 50


class LocalAssistant:
    """Local implementation of assistant functionality using HuggingFace models."""

    def __init__(
        self,
        name: str,
        instructions: str,
        model: str,
        temperature: float
    ):
        self.name = name
        self.instructions = instructions
        self.temperature = temperature

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.pipeline = pipeline(
            'text-generation',
            model=model,
            model_kwargs={'torch_dtype': 'auto'},
            tokenizer=self.tokenizer,
            device_map='auto',
            # low_cpu_mem_usage=False
        )

    def generate_response(self, messages: list[dict[str, str]]) -> str:
        """Generate a response based on the conversation history.

        :param messages: List of message dictionaries with 'role' and 'content' keys
        :return: Generated response text
        """
        # Format conversation history into prompt
        prompt = self._format_prompt(messages)

        # Generate response
        response = self.pipeline(
            prompt,
            max_new_tokens=1024 * 10,
            temperature=self.temperature,
            do_sample=True,
            top_p=.9,
            top_k=50
        )[0]["generated_text"]

        # Extract just the new generated text
        return response[len(prompt):].strip()

    def _format_prompt(self, messages: list[dict[str, str]]) -> str:
        """Format messages into a prompt the model can understand.

        :param messages: List of message dictionaries
        :return: Formatted prompt string
        """
        # Start with system instructions
        prompt = f"System: {self.instructions}\n\n"

        # Add message history
        for msg in messages:
            role = "Assistant" if msg["role"] == "assistant" else msg["role"].capitalize()
            prompt += f"{role}: {msg['content']}\n\n"

        # Add assistant prefix for next response
        prompt += "Assistant: "

        return prompt


class LocalThread:
    """Local implementation of conversation thread functionality."""

    def __init__(self):
        self.messages: list[dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the thread.

        :param role: Role of the message sender ("user" or "assistant")
        :param content: Content of the message
        """
        self.messages.append({
            "role": role,
            "content": content
        })

    def get_messages(self) -> list[dict[str, str]]:
        """Get all messages in the thread.

        :return: List of message dictionaries
        """
        return self.messages


def run_meeting(
    meeting_type: Literal["team", "individual"],
    agenda: str,
    save_dir: Path,
    save_name: str = "discussion",
    team_lead: Optional[Agent] = None,
    team_members: Optional[tuple[Agent, ...]] = None,
    team_member: Optional[Agent] = None,
    agenda_questions: tuple[str, ...] = (),
    agenda_rules: tuple[str, ...] = (),
    summaries: tuple[str, ...] = (),
    contexts: tuple[str, ...] = (),
    num_rounds: int = 0,
    temperature: float = CONSISTENT_TEMPERATURE,
    model: str = "microsoft/phi-4",
    return_summary: bool = False,
) -> Optional[str]:
    """Runs a meeting with LLM agents using local HuggingFace models.

    :param meeting_type: The type of meeting ("team" or "individual")
    :param agenda: The agenda for the meeting
    :param save_dir: The directory to save the discussion
    :param save_name: The name of the discussion file that will be saved
    :param team_lead: The team lead for a team meeting (None for individual meeting)
    :param team_members: The team members for a team meeting (None for individual meeting)
    :param team_member: The team member for an individual meeting (None for team meeting)
    :param agenda_questions: The agenda questions to answer
    :param agenda_rules: The rules for the meeting
    :param summaries: The summaries of previous meetings
    :param contexts: The contexts for the meeting
    :param num_rounds: The number of rounds of discussion
    :param temperature: The sampling temperature
    :param model: huggingface model to use
    :param return_summary: Whether to return the summary of the meeting
    :return: The summary of the meeting if return_summary is True, else None
    """
    # Validate meeting type
    if meeting_type == "team":
        if team_lead is None or team_members is None or len(team_members) == 0:
            raise ValueError("Team meeting requires team lead and team members")
        if team_member is not None:
            raise ValueError("Team meeting does not require individual team member")
        if team_lead in team_members:
            raise ValueError("Team lead must be separate from team members")
        if len(set(team_members)) != len(team_members):
            raise ValueError("Team members must be unique")
    elif meeting_type == "individual":
        if team_member is None:
            raise ValueError("Individual meeting requires individual team member")
        if team_lead is not None or team_members is not None:
            raise ValueError("Individual meeting does not require team lead or team members")
    else:
        raise ValueError(f"Invalid meeting type: {meeting_type}")

    # Start timing the meeting
    start_time = time.time()

    # Set up team
    if meeting_type == "team":
        team = [team_lead] + list(team_members)
    else:
        team = [team_member] + [SCIENTIFIC_CRITIC]

    # Set up the assistants
    agent_to_assistant = {
        agent: LocalAssistant(
            name=agent.title,
            instructions=agent.prompt,
            model=model,
            temperature=temperature
        )
        for agent in team
    }

    # Set up the thread
    thread = LocalThread()
    discussion = []

    # Initial prompt for team meeting
    if meeting_type == "team":
        initial_prompt = team_meeting_start_prompt(
            team_lead=team_lead,
            team_members=team_members,
            agenda=agenda,
            agenda_questions=agenda_questions,
            agenda_rules=agenda_rules,
            summaries=summaries,
            contexts=contexts,
            num_rounds=num_rounds,
        )
        thread.add_message(role="user", content=initial_prompt)
        discussion.append({"agent": "User", "message": initial_prompt})

    # Loop through rounds
    for round_index in trange(num_rounds + 1, desc="Rounds (+ Final Round)"):
        round_num = round_index + 1

        # Loop through team and elicit responses
        for agent in tqdm(team, desc="Team"):
            # Generate appropriate prompt based on agent and round number
            if meeting_type == "team":
                if agent == team_lead:
                    if round_index == 0:
                        prompt = team_meeting_team_lead_initial_prompt(team_lead=team_lead)
                    elif round_index == num_rounds:
                        prompt = team_meeting_team_lead_final_prompt(
                            team_lead=team_lead,
                            agenda=agenda,
                            agenda_questions=agenda_questions,
                            agenda_rules=agenda_rules,
                        )
                    else:
                        prompt = team_meeting_team_lead_intermediate_prompt(
                            team_lead=team_lead,
                            round_num=round_num - 1,
                            num_rounds=num_rounds,
                        )
                else:
                    prompt = team_meeting_team_member_prompt(
                        team_member=agent,
                        round_num=round_num,
                        num_rounds=num_rounds
                    )
            else:
                if agent == SCIENTIFIC_CRITIC:
                    prompt = individual_meeting_critic_prompt(
                        critic=SCIENTIFIC_CRITIC,
                        agent=team_member
                    )
                else:
                    if round_index == 0:
                        prompt = individual_meeting_start_prompt(
                            team_member=team_member,
                            agenda=agenda,
                            agenda_questions=agenda_questions,
                            agenda_rules=agenda_rules,
                            summaries=summaries,
                            contexts=contexts,
                        )
                    else:
                        prompt = individual_meeting_agent_prompt(
                            critic=SCIENTIFIC_CRITIC,
                            agent=team_member
                        )

            # Add user prompt to thread and discussion
            thread.add_message(role="user", content=prompt)
            discussion.append({"agent": "User", "message": prompt})

            # Generate response using appropriate assistant
            response = agent_to_assistant[agent].generate_response(thread.get_messages())

            # Add response to thread and discussion
            thread.add_message(role="assistant", content=response)
            discussion.append({"agent": agent.title, "message": response})

            # If final round, only team lead or team member responds
            if round_index == num_rounds:
                break

    # Calculate approximate token counts
    token_counts = {
        "input": sum(count_tokens(turn["message"]) for turn in discussion if turn["agent"] == "User"),
        "output": sum(count_tokens(turn["message"]) for turn in discussion if turn["agent"] != "User"),
        "tool": 0,  # No tool usage in local version
        "max": max(count_tokens(turn["message"]) for turn in discussion)
    }

    # Print meeting statistics
    print("\nMeeting Statistics:")
    print(f"Input tokens: {token_counts['input']:,}")
    print(f"Output tokens: {token_counts['output']:,}")
    print(f"Max tokens: {token_counts['max']:,}")
    print(f"Time: {int((time.time() - start_time) // 60)}:{int((time.time() - start_time) % 60):02d}")
    print(f"Model: {model}")
    # print(f"Device: {model_config.device}")

    # Save the discussion
    save_meeting(
        save_dir=save_dir,
        save_name=save_name,
        discussion=discussion
    )

    # Return summary if requested
    if return_summary:
        return discussion[-1]["message"]
