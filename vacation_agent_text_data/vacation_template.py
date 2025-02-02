from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.schema import SystemMessage

class PromptTemplateImpl():

     # Step 3: Create message templates and then build your custom chat prompt.
    def generate(self, **kwargs):
        # Build the custom chat prompt using your subclass.
        return VacationManagerPromptTemplate()
    
# Step 2: Create a custom subclass overriding the behavior of ChatPromptTemplate.
class VacationManagerPromptTemplate(ChatPromptTemplate):
    
    def __init__(self):
        system_message_template = SystemMessagePromptTemplate.from_template(
            """
                You are a Vacation Manager and your objective is to know information regarding employees scheduled vacations.\n\n
                Only answer based on information data you get from the documents
                If you don't have information, just answer that you don't have it 
                and don't mention no person information that is not related to the question asked.
            """

        )

        human_message = HumanMessagePromptTemplate.from_template(
            "User Query: {input}\n\nAdditional Information:\n{context}\n"
        )

        super().__init__(messages=[system_message_template, human_message])    