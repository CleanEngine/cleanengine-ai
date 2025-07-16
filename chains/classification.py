"""
Classification chain for determining the type of user query.
"""

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from typing import Dict, Any, Optional


class ClassificationChain:
    """
    Classifies user questions into categories: 최신소식, 전문지식, 리셋, or 기타
    """
    
    def __init__(self, model: str = "gpt-4o-mini", thread_id: Optional[str] = None, memory: Optional[Any] = None):
        """
        Initialize the classification chain.
        
        Args:
            model: OpenAI model to use for classification
            thread_id: Thread ID for memory management
            memory: Shared memory instance
        """
        self.model = model
        self.thread_id = thread_id
        self.memory = memory
        self._setup_chain()
    
    def _setup_chain(self):
        """Set up the classification chain with prompt and model."""
        
        # Classification prompt template
        self.prompt = PromptTemplate.from_template(
            """주어진 사용자 질문과 대화 히스토리를 보고 `최신소식`, `전문지식`, `리셋`, 또는 `기타` 중 하나로 분류하세요. 
리셋 관련 키워드: "리셋", "초기화", "지워", "새로시작", "reset", "clear" 등
한 단어 이상으로 응답하지 마세요.

<chat_history>
{chat_history}
</chat_history>

<question>
{question}
</question>

Classification:"""
        )
        
        # Create the chain
        self.chain = (
            self.prompt
            | ChatOpenAI(model=self.model)
            | StrOutputParser()
        )
    
    def classify(self, question: str, chat_history: str = "") -> str:
        """
        Classify a user question.
        
        Args:
            question: The user's question
            chat_history: Previous conversation history
            
        Returns:
            Classification result as string
        """
        # Use shared memory if available, otherwise use provided chat_history
        if self.memory:
            from modules.memory_manager import memory_manager
            chat_history = memory_manager.get_chat_history_string(self.memory)
        
        return self.chain.invoke({
            "question": question,
            "chat_history": chat_history
        })
    
    def invoke(self, inputs: Dict[str, Any]) -> str:
        """
        Invoke the classification chain with inputs.
        
        Args:
            inputs: Dictionary containing 'question' and 'chat_history'
            
        Returns:
            Classification result as string
        """
        # Use shared memory if available
        if self.memory:
            from modules.memory_manager import memory_manager
            inputs = inputs.copy()
            inputs["chat_history"] = memory_manager.get_chat_history_string(self.memory)
        
        return self.chain.invoke(inputs)
    
    async def ainvoke(self, inputs: Dict[str, Any]) -> str:
        """
        Asynchronously invoke the classification chain.
        
        Args:
            inputs: Dictionary containing 'question' and 'chat_history'
            
        Returns:
            Classification result as string
        """
        # Use shared memory if available
        if self.memory:
            from modules.memory_manager import memory_manager
            inputs = inputs.copy()
            inputs["chat_history"] = memory_manager.get_chat_history_string(self.memory)
        
        return await self.chain.ainvoke(inputs)
    
    def get_chain(self):
        """Get the underlying chain object."""
        return self.chain