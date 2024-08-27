from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import List
from langchain_core.messages import BaseMessage
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from agents.artist.artist_lib import Artist, Impressionist
from agents.critic.critic_lib import Critic

class PromptQualityClassifier:
    def __init__(self, llm=ChatOpenAI(model="gpt-3.5-turbo-0125")):
        self.llm = llm
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert in evaluating text-to-image prompts. Your job is to determine if a given prompt is sufficiently refined and ready to be used for image generation.

            A good prompt should be:
            1. Clear and specific
            2. Descriptive of visual elements
            4. Appropriate in length (neither too short nor excessively long). 50 word max.
            5. Aligned with the artist's style and intention

            Analyze the given prompt and respond with either:
            "REFINED" if the prompt meets the criteria and is ready for use, or
            "NEEDS_WORK" if the prompt still requires improvement.

            Provide a brief explanation for your decision.
            """),
            ("human", "{prompt}")
        ])
        # We can just define the chain at this level. 
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def classify(self, prompt: str) -> tuple[str, str]:
        result = self.chain.invoke({"prompt": prompt})
        classification = "REFINED" if "REFINED" in result else "NEEDS_WORK"
        explanation = result.split("\n", 1)[1] if "\n" in result else ""
        return classification, explanation


class PromptRefinementConversation:
    """
    This class is made to refine the prompt to create an artwork. 
    """
    def __init__(self, critic: Critic, artist: Artist, classifier: PromptQualityClassifier, max_rounds: int = 5):
        self.critic = critic
        self.artist = artist
        self.classifier = classifier
        self.max_rounds = max_rounds
        self.messages: List[BaseMessage] = []

    def initialize_conversation(self, initial_prompt: str):
        self.messages = [
            SystemMessage(content="We are refining a prompt for artwork creation. The conversation will alternate between a critic and an artist."),
            HumanMessage(content=f"Initial prompt: {initial_prompt}")
        ]

    def add_message(self, role: str, content: str):
        if role == "critic":
            self.messages.append(AIMessage(content=f"Critic: {content}"))
        elif role == "artist":
            self.messages.append(HumanMessage(content=f"Artist: {content}"))
        elif role == "classifier":
            self.messages.append(SystemMessage(content=f"Classifier: {content}"))

    # should be able to pass a subclass of the critic. 
    # store it. 
    # The question is still how could we use RAG long-memory style ? 
    def refine_prompt(self, initial_prompt: str) -> str:
        self.initialize_conversation(initial_prompt)
        current_prompt = initial_prompt

        for round in range(self.max_rounds):
            print(f"\nRound {round + 1}:")

            # Critic's turn
            critic_response = self.critic.dicussion_with_artist_prompt_enhance(self.messages)
            self.add_message("critic", critic_response)
            print(f"Critic: {critic_response}")

            # Check prompt quality
            classification, explanation = self.classifier.classify(critic_response)
            self.add_message("classifier", f"{classification} - {explanation}")
            print(f"Classifier: {classification}")
            print(f"Explanation: {explanation}")

            if classification == "REFINED":
                return critic_response

            # Artist's turn
            artist_response = self.artist.generate_artwork_prompt(self.messages)
            self.add_message("artist", artist_response)
            print(f"Artist: {artist_response}")

            # Check prompt quality again
            classification, explanation = self.classifier.classify(artist_response)
            self.add_message("classifier", f"{classification} - {explanation}")
            print(f"Classifier: {classification}")
            print(f"Explanation: {explanation}")

            if classification == "REFINED":
                return artist_response

            current_prompt = artist_response

        print("Max rounds reached. Returning the last prompt.")
        return current_prompt

    

# Usage
if __name__ == "__main__": 
    critic = Critic()
    artist = Impressionist()
    classifier = PromptQualityClassifier()

    conversation = PromptRefinementConversation(critic, artist, classifier)
    refined_prompt = conversation.refine_prompt("Initial artwork prompt: A serene landscape with mountains and a lake")

    print(f"\nFinal refined prompt: {refined_prompt}")