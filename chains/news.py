"""
News chain for handling latest news and current events queries.
"""

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.retrievers.tavily_search_api import TavilySearchAPIRetriever
from typing import Dict, Any, Optional


class NewsChain:
    """
    Handles queries about latest news and current events using web search.
    """
    
    def __init__(self, model: str = "gpt-4o-mini", search_k: int = 3, thread_id: Optional[str] = None, memory: Optional[Any] = None):
        """
        Initialize the news chain.
        
        Args:
            model: OpenAI model to use for response generation
            search_k: Number of search results to retrieve
            thread_id: Thread ID for memory management
            memory: Shared memory instance
        """
        self.model = model
        self.search_k = search_k
        self.thread_id = thread_id
        self.memory = memory
        self._setup_chain()
    
    def _setup_chain(self):
        """Set up the news chain with retriever, prompt and model."""
        
        # Web search retriever
        self.retriever = TavilySearchAPIRetriever(k=self.search_k)
        
        # News prompt template
        self.prompt = PromptTemplate.from_template(
            """You are an HUMINT in news. \
Always answer questions starting with "최신 데이터에 따르면..". \
When you work with numbers, be mindful of units.\
If you don't know the answer, just say that you don't know\

Previous conversation:
{chat_history}

Respond to the following question based on the context and previous conversation:
Context: {context}
Question: {question}
Answer:"""
        )
        
        # Create the chain
        self.chain = (
            {
                "question": lambda x: x["question"],
                "context": lambda x: self.retriever.invoke(x["question"]),
                "chat_history": lambda x: x.get("chat_history", "")
            }
            | self.prompt
            | ChatOpenAI(model=self.model)
            | StrOutputParser()
        )
    
    def process(self, question: str, chat_history: str = "") -> str:
        """
        Process a news-related question.
        
        Args:
            question: The user's question
            chat_history: Previous conversation history
            
        Returns:
            Response based on latest news data
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
        Invoke the news chain with inputs.
        
        Args:
            inputs: Dictionary containing 'question' and optional 'chat_history'
            
        Returns:
            News-based response as string
        """
        # Use shared memory if available
        if self.memory:
            from modules.memory_manager import memory_manager
            inputs = inputs.copy()
            inputs["chat_history"] = memory_manager.get_chat_history_string(self.memory)
        
        return self.chain.invoke(inputs)
    
    async def ainvoke(self, inputs: Dict[str, Any]) -> str:
        """
        Asynchronously invoke the news chain.
        
        Args:
            inputs: Dictionary containing 'question' and optional 'chat_history'
            
        Returns:
            News-based response as string
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
    
    def get_retriever(self):
        """Get the web search retriever."""
        return self.retriever