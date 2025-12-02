from typing import Annotated
from autogen import ConversableAgent, LLMConfig, UpdateSystemMessage
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import AutoPattern
from autogen.agentchat.group import ReplyResult, AgentNameTarget
from autogen.agentchat.group import AgentTarget, TerminateTarget
from autogen.agentchat.group import ContextVariables
from autogen.agentchat.group import OnCondition, StringLLMCondition

def _read_prompt_from_file(path):
    with open(path, 'r') as f:
        return f.read()

def review_reply(
    feedback: Annotated[str,"Feedback on improving this reply to be accurate and relavant for the user prompt"], 
    rating: Annotated[int,"The rating of the reply on a scale of 1 to 10"], 
    context_variables: ContextVariables
) -> ReplyResult:
    """Review the reply of the Ai Agent to the user prompt with respect to correctness, clarity and relevance for the user prompt"""
    context_variables["feedback"] = feedback
    context_variables["rating"] = rating
    context_variables["revisions"] += 1

    
    messages = list(st.session_state.agents['qiskit_agent'].chat_messages.values())[0]


    #st.markdown(messages[-2])
    reply = None
    for item in messages:
        if item['name'] == 'qiskit_agent' or item['name'] == 'improve_reply_agent':
            reply = item["content"]
       
    if reply:
        context_variables["last_answer"] = reply
    
    if rating < 8 and context_variables["revisions"] < 3:
        return ReplyResult(
            context_variables=context_variables,
            target=AgentNameTarget("improve_reply_agent"),
            message=f'Please revise the answer considering this feedback {feedback}',
        )

    elif rating >= 8:
        #st.markdown("Formatting final answer...")
        return ReplyResult(
            context_variables=context_variables,
            target=AgentNameTarget("improve_reply_agent_final"),
            message=f'The answer is already of sufficient quality. Focus on formatting the reply',
        )
        
    else:
        return ReplyResult(
            context_variables=context_variables,
            target=AgentNameTarget("improve_reply_agent_final"),
            message=f'Please revise the answer considering this feedback {feedback}',
        )


def deep_thought_mode(user_input: str, context: str, message_history):
    # 1. Load Config
    Initial_Agent_Instructions = _read_prompt_from_file("prompts/qiskit_instructions.txt") # Reuse or adapt qiskit_instructions
    Refine_Agent_Instructions = _read_prompt_from_file("prompts/qiskit_refinement.txt") # Instructions on imporving an answer
    Review_Agent_Instructions = _read_prompt_from_file("prompts/review_instructions.txt") # Adapt rating_instructions
    Formatting_Agent_Instructions = _read_prompt_from_file("prompts/formatting_instructions.txt") # New prompt file
    Code_Execution_Agent_Instructions = _read_prompt_from_file("prompts/codeexecutor_instructions.txt") # New prompt file

    common_config = LLMConfig.from_json(path="config/common_llm_config.json")
    review_config =LLMConfig.from_json(path="config/review_llm_config.json")

    # 2. Define agents
    qiskit_agent = ConversableAgent(
        name="qiskit_agent",
        system_message=Initial_Agent_Instructions,
        description="Initial agent that answers user prompt. Expert in the qiskit code",
        human_input_mode="NEVER",
        llm_config=common_config
    )
    review_agent = ConversableAgent(
        name="review_agent",
        update_agent_state_before_reply=[
            UpdateSystemMessage(Review_Agent_Instructions),
        ],
        human_input_mode="NEVER",
        description="Reviews the AI answer to user prompt",
        llm_config=review_config,
        functions=review_reply,
    )
    refine_agent = ConversableAgent(
        name="improve_reply_agent",
        update_agent_state_before_reply=[
            UpdateSystemMessage(Refine_Agent_Instructions),
        ],
        human_input_mode="NEVER",
        description="Improves the AI reply by taking into account the feedback",
        llm_config=common_config,
    )
    refine_agent_final = ConversableAgent(
        name="improve_reply_agent_final",
        update_agent_state_before_reply=[
            UpdateSystemMessage(Refine_Agent_Instructions),
        ],
        human_input_mode="NEVER",
        description="Improves the AI reply by taking into account the feedback",
        llm_config=common_config,
    )

    # 3. Define initial contexts
    shared_context = ContextVariables(data =  {
        "last_answer": "see chat history",
        "feedback": "see chat history",
        "rating": 0,
        "revisions": 0,
    })

    # 3. Define Patterns
    qiskit_agent.handoffs.set_after_work(AgentTarget(review_agent))
    review_agent.handoffs.set_after_work(AgentTarget(refine_agent))
    refine_agent.handoffs.set_after_work(AgentTarget(review_agent))
    refine_agent_final.handoffs.set_after_work(TerminateTarget())
    refine_agent.handoffs.add_llm_conditions([
        OnCondition(target=AgentTarget(refine_agent_final), condition=StringLLMCondition(prompt="The reply to the latest user question has been reviewd and received a favarable rating (equivalent to 7 or higher)"))
    ])

    pattern = AutoPattern(
        initial_agent=qiskit_agent,
        agents=[qiskit_agent, review_agent, refine_agent, refine_agent_final],
        group_manager_args={"llm_config": common_config},
        context_variables=shared_context,
    )

    # 5. Run group chat
    result, context_variables, last_agent = initiate_group_chat(
        pattern=pattern,
        messages=f"Context from documents: {context}\n\nConversation history:\n{message_history}\n\nUser question: {user_input}",
        max_rounds=10,
    )
    formatted_answer = None  # default to nothing

    # 1. If the formatting agent gave the last reply, use that
    if last_agent == refine_agent_final or last_agent == refine_agent:
        formatted_answer = result.chat_history[-1]["content"]

    # 2. Otherwise, use shared_context["last_answer"] if it's non-empty
    if not formatted_answer and shared_context.get("last_answer"):
        formatted_answer = shared_context["last_answer"]
                
    # 3. Otherwise, fall back to the initial agent's last message

    if not formatted_answer:
        try:
            for item in result.chat_history:
                if item['name'] == 'qiskit_agent' or item['name'] == 'imporve_reply_agent':
                    formatted_answer = item["content"]
        except:
            formatted_answer = 'failed to load chat history'

    return formatted_answer