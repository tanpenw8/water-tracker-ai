import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(model="gpt-4o-mini",
 temperature=0.5,
 api_key=OPENAI_API_KEY,
)
class waterIntakeAgent:

    def __init__(self):
        self.history = []
        
    def analyze_intake(self, intake_ml):

        prompt = f"""
        You are a water intake agent.
        You are a hydration assistant.
        The user has consumed {intake_ml} milliliters of water.
        You need to analyze the intake and provide a recommendation for the user.
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content


if __name__ == "__main__":
    agent = waterIntakeAgent()
    intake = 1500
    feedback = agent.analyze_intake(intake)
    print(f"Hydration Analysis: {feedback}")

