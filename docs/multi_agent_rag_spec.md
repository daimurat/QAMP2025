# Multi-Agent Iterative RAG System - Technical Specification

## Version 1.0 | January 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [State Model](#3-state-model)
4. [Agent Specifications](#4-agent-specifications)
5. [Message Passing Protocol](#5-message-passing-protocol)
6. [Orchestrator Design](#6-orchestrator-design)
7. [Failure Handling](#7-failure-handling)
8. [Interface Definitions](#8-interface-definitions)
9. [Pseudocode Implementation](#9-pseudocode-implementation)
10. [Configuration](#10-configuration)
11. [Appendix A: File Structure](#appendix-a-file-structure)
12. [Appendix B: Sequence Diagram](#appendix-b-sequence-diagram)
13. [Appendix C: Implementation Checkpoints](#appendix-c-implementation-checkpoints)

---

## 1. Executive Summary

### 1.1 Current State

The existing system implements a single-pass RAG pipeline:
- **Fast Mode**: One RAG retrieval → one LLM response
- **Deep Thought Mode**: Planning with function calls but no structured iteration loop

### 1.2 Target State

A multi-agent iterative system with:
- **Planner Agent**: Decomposes questions, requests RAG retrievals, defines execution plans
- **Code Agent**: Generates executable code based on plans
- **Executor Agent**: Runs code in sandbox, captures outputs
- **Evaluator Agent**: Validates results, provides feedback for retry

### 1.3 Key Design Principles

1. **Bounded Iterations**: Hard limit with clear termination conditions
2. **State Persistence**: Full audit trail across iterations
3. **Graceful Degradation**: Partial results returned on failure
4. **Separation of Concerns**: Each agent has a single responsibility

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              ORCHESTRATOR                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     State Manager                                │    │
│  │  • Iteration counter  • Plan versions  • Execution logs         │    │
│  │  • Retrieved docs     • Code versions  • Evaluation history     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ PLANNER  │───▶│   CODE   │───▶│ EXECUTOR │───▶│EVALUATOR │          │
│  │  AGENT   │◀───│  AGENT   │    │  AGENT   │    │  AGENT   │──────┐   │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘      │   │
│       │              ▲                                               │   │
│       ▼              │                                               ▼   │
│  ┌──────────┐        │                                    ┌──────────┐  │
│  │   RAG    │        │                                    │ DECISION │  │
│  │RETRIEVER │        └────────────────────────────────────│  ROUTER  │  │
│  └──────────┘                                             └──────────┘  │
│                            feedback loop                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| **Orchestrator** | Controls iteration flow, manages state, enforces limits |
| **State Manager** | Persists all artifacts across iterations |
| **Planner Agent** | Decomposes problems, requests RAG, creates execution plans |
| **Code Agent** | Generates executable Python code from plans |
| **Executor Agent** | Sandboxed code execution, output capture |
| **Evaluator Agent** | Validates correctness, decides next action |
| **RAG Retriever** | Semantic search over Qiskit documentation |
| **Decision Router** | Routes evaluator decisions to appropriate handlers |

### 2.3 Data Flow

```
User Question
      │
      ▼
┌─────────────┐
│  PLANNER    │◀──────────────────────────────────┐
│             │                                    │
│ • Analyze   │                                    │
│ • RAG?      │──yes──▶ RAG Retriever             │
│ • Plan      │                                    │
└──────┬──────┘                                    │
       │ plan + context                            │
       ▼                                           │
┌─────────────┐                                    │
│    CODE     │                                    │
│   AGENT     │                                    │
│             │                                    │
│ • Generate  │                                    │ feedback
│   code      │                                    │
└──────┬──────┘                                    │
       │ code                                      │
       ▼                                           │
┌─────────────┐                                    │
│  EXECUTOR   │                                    │
│   AGENT     │                                    │
│             │                                    │
│ • Execute   │                                    │
│ • Capture   │                                    │
└──────┬──────┘                                    │
       │ results                                   │
       ▼                                           │
┌─────────────┐                                    │
│  EVALUATOR  │                                    │
│   AGENT     │                                    │
│             │──retry──▶──────────────────────────┘
│ • Validate  │
│ • Decide    │──success──▶ Final Answer
│             │
│             │──abort──▶ Partial Result / Error
└─────────────┘
```

---

## 3. State Model

### 3.1 Iteration State Schema

```python
@dataclass
class IterationState:
    """Immutable snapshot of a single iteration."""
    iteration_id: int
    timestamp: datetime
    
    # Planner outputs
    plan: Plan
    rag_queries: List[str]
    retrieved_documents: List[RetrievedDocument]
    
    # Code Agent outputs
    generated_code: str
    code_version: int
    
    # Executor outputs
    execution_result: ExecutionResult
    
    # Evaluator outputs
    evaluation: Evaluation
    decision: Decision  # SUCCESS | RETRY | ABORT
    feedback: Optional[str]
```

### 3.2 Session State Schema

```python
@dataclass
class SessionState:
    """Persistent state across all iterations."""
    session_id: str
    user_question: str
    start_time: datetime
    end_time: Optional[datetime]
    
    # Configuration
    max_iterations: int
    current_iteration: int
    
    # Cumulative state
    iterations: List[IterationState]
    all_retrieved_docs: Dict[str, RetrievedDocument]  # Deduplicated by ID
    
    # Final outputs
    final_answer: Optional[str]
    final_code: Optional[str]
    termination_reason: TerminationReason
    
    # Metrics
    total_rag_calls: int
    total_llm_calls: int
    total_code_executions: int
```

### 3.3 Supporting Data Types

```python
@dataclass
class Plan:
    """Structured execution plan from Planner."""
    plan_id: str
    version: int
    
    # Problem decomposition
    sub_tasks: List[SubTask]
    
    # RAG requirements
    rag_needed: bool
    rag_queries: List[str]
    
    # Code requirements
    code_needed: bool
    code_requirements: List[str]
    
    # Success criteria
    acceptance_criteria: List[str]
    
    # Dependencies on previous iterations
    uses_previous_context: bool
    referenced_iterations: List[int]

@dataclass
class SubTask:
    """Individual task within a plan."""
    task_id: str
    description: str
    task_type: TaskType  # RETRIEVAL | COMPUTATION | VALIDATION
    dependencies: List[str]  # Other task_ids

@dataclass
class RetrievedDocument:
    """Document chunk from RAG retrieval."""
    doc_id: str
    text: str
    source: str
    url: str
    relevance_score: float
    retrieved_at: datetime
    query_used: str

@dataclass
class ExecutionResult:
    """Output from code execution."""
    success: bool
    stdout: str
    stderr: str
    return_value: Any
    artifacts: Dict[str, bytes]  # Files, images, etc.
    execution_time_ms: int
    memory_usage_mb: float
    error_type: Optional[str]
    error_traceback: Optional[str]

@dataclass
class Evaluation:
    """Evaluator's assessment of execution results."""
    evaluation_id: str
    
    # Correctness assessment
    answers_question: bool
    code_executes: bool
    output_valid: bool
    criteria_met: Dict[str, bool]  # Maps criterion to pass/fail
    
    # Quality scores (0-1)
    correctness_score: float
    completeness_score: float
    code_quality_score: float
    
    # Reasoning
    reasoning: str
    identified_issues: List[str]
    
    # Recommendations
    suggested_improvements: List[str]

class Decision(Enum):
    SUCCESS = "success"
    RETRY = "retry"
    ABORT = "abort"

class TerminationReason(Enum):
    SUCCESS = "success"
    MAX_ITERATIONS = "max_iterations_reached"
    ABORT_REQUESTED = "evaluator_abort"
    FATAL_ERROR = "fatal_error"
    TIMEOUT = "timeout"
    USER_CANCELLED = "user_cancelled"

class TaskType(Enum):
    RETRIEVAL = "retrieval"
    COMPUTATION = "computation"
    VALIDATION = "validation"
```

---

## 4. Agent Specifications

### 4.1 Planner Agent

#### Role
Decomposes user questions into actionable plans, determines what information needs to be retrieved, and what code needs to be written.

#### System Prompt

```markdown
You are the PLANNER agent in a multi-agent system for answering Qiskit quantum computing questions.

## Your Responsibilities
1. Analyze the user's question to understand what is being asked
2. Decompose complex questions into smaller, actionable sub-tasks
3. Determine what documentation needs to be retrieved via RAG
4. Specify what code needs to be written and executed
5. Define clear acceptance criteria for the answer

## Input You Receive
- User's original question
- Feedback from previous iterations (if any)
- Previously retrieved documents (if any)
- Previous execution results (if any)

## Output Format
You MUST respond with a JSON object following this exact schema:

```json
{
  "analysis": "Your analysis of what the question is asking",
  "sub_tasks": [
    {
      "task_id": "task_1",
      "description": "Description of the task",
      "task_type": "RETRIEVAL | COMPUTATION | VALIDATION",
      "dependencies": []
    }
  ],
  "rag_needed": true,
  "rag_queries": [
    "specific query 1 for documentation search",
    "specific query 2 for documentation search"
  ],
  "code_needed": true,
  "code_requirements": [
    "Requirement 1 for the code",
    "Requirement 2 for the code"
  ],
  "acceptance_criteria": [
    "Criterion 1: The code must compile without errors",
    "Criterion 2: The circuit must have N qubits",
    "Criterion 3: The output must show expected measurement results"
  ],
  "uses_previous_context": false,
  "referenced_iterations": []
}
```

## Guidelines
- Be SPECIFIC with RAG queries - target exact API methods, classes, or concepts
- Code requirements should be concrete and testable
- Acceptance criteria should be objectively verifiable
- If this is a retry, analyze previous feedback and adjust the plan accordingly
- You may request multiple RAG queries to gather comprehensive context
```

#### Input Schema

```python
@dataclass
class PlannerInput:
    user_question: str
    iteration_number: int
    
    # Previous iteration context (if retry)
    previous_feedback: Optional[str]
    previous_plan: Optional[Plan]
    previous_code: Optional[str]
    previous_execution_result: Optional[ExecutionResult]
    
    # Accumulated context
    all_retrieved_docs: List[RetrievedDocument]
```

#### Output Schema

```python
@dataclass
class PlannerOutput:
    plan: Plan
    raw_response: str  # Original LLM response for debugging
```

### 4.2 Code Agent

#### Role
Generates executable Python code based on the Planner's specifications and available context.

#### System Prompt

```markdown
You are the CODE agent in a multi-agent system for answering Qiskit quantum computing questions.

## Your Responsibilities
1. Generate complete, executable Python code based on the provided plan
2. Use information from retrieved documentation accurately
3. Follow Qiskit best practices and current API conventions
4. Ensure code is self-contained and can run independently

## Input You Receive
- The plan from the Planner agent
- Retrieved documentation context
- Code requirements to implement
- Feedback from previous iterations (if any)

## Output Format
You MUST respond with ONLY executable Python code wrapped in a single code block:

```python
# Your complete, executable code here
```

## Code Requirements
1. **Self-contained**: Include all necessary imports
2. **Executable**: Code must run without modification
3. **Output-producing**: Print results or return values that can be validated
4. **Error-handled**: Include try-except for expected failure modes
5. **Documented**: Include comments explaining key steps
6. **Qiskit-compliant**: Use current Qiskit 1.x API patterns

## IMPORTANT
- Do NOT include explanations outside the code block
- Do NOT use placeholder values - use real, working code
- Do NOT assume external state or variables
- DO save any plots to files instead of using plt.show()
- DO print intermediate results for debugging
- DO use the exact APIs mentioned in the retrieved documentation
```

#### Input Schema

```python
@dataclass
class CodeAgentInput:
    plan: Plan
    code_requirements: List[str]
    retrieved_context: str  # Concatenated relevant docs
    
    # Previous iteration context (if retry)
    previous_code: Optional[str]
    previous_error: Optional[str]
    previous_feedback: Optional[str]
```

#### Output Schema

```python
@dataclass
class CodeAgentOutput:
    code: str
    language: str  # Always "python" for this system
    raw_response: str
```

### 4.3 Executor Agent

#### Role
Executes generated code in a sandboxed environment and captures all outputs.

#### Implementation Notes
This agent is primarily infrastructure code, not an LLM agent. It manages:
- Sandboxed Python execution
- Timeout enforcement
- Output capture (stdout, stderr, return values)
- Artifact collection (files, images)
- Resource monitoring

#### Input Schema

```python
@dataclass
class ExecutorInput:
    code: str
    timeout_seconds: int = 60
    max_memory_mb: int = 512
    allowed_imports: List[str] = field(default_factory=lambda: [
        "qiskit", "numpy", "matplotlib", "scipy", "math", "json", "datetime"
    ])
    working_directory: str = "work"
```

#### Output Schema

```python
@dataclass
class ExecutorOutput:
    result: ExecutionResult
    execution_id: str
```

### 4.4 Evaluator Agent

#### Role
Assesses whether execution results satisfy the plan and decides next action.

#### System Prompt

```markdown
You are the EVALUATOR agent in a multi-agent system for answering Qiskit quantum computing questions.

## Your Responsibilities
1. Assess whether the code execution results answer the user's question
2. Verify that acceptance criteria from the plan are met
3. Decide whether to accept the result, retry with feedback, or abort
4. Provide constructive feedback if a retry is needed

## Input You Receive
- Original user question
- The execution plan with acceptance criteria
- Generated code
- Execution results (stdout, stderr, errors, artifacts)
- Iteration history

## Output Format
You MUST respond with a JSON object following this exact schema:

```json
{
  "assessment": {
    "answers_question": true,
    "code_executes": true,
    "output_valid": true,
    "criteria_met": {
      "criterion_1": true,
      "criterion_2": false,
      "criterion_3": true
    }
  },
  "scores": {
    "correctness": 0.85,
    "completeness": 0.90,
    "code_quality": 0.80
  },
  "reasoning": "Detailed explanation of the assessment",
  "issues": [
    "Issue 1 if any",
    "Issue 2 if any"
  ],
  "decision": "SUCCESS | RETRY | ABORT",
  "feedback": "If RETRY: specific, actionable feedback for the Planner. If ABORT: reason for aborting.",
  "final_answer": "If SUCCESS: the final answer to present to the user"
}
```

## Decision Guidelines

### Choose SUCCESS when:
- Code executes without errors
- Output directly answers the user's question
- All or most acceptance criteria are met
- Any remaining issues are minor

### Choose RETRY when:
- Code has fixable errors
- Output is partial or incorrect but salvageable
- Key acceptance criteria are not met
- You can provide specific feedback for improvement
- Maximum iterations not yet reached

### Choose ABORT when:
- Fundamental misunderstanding of the question
- Required APIs or features don't exist
- Repeated failures on the same issue (3+ times)
- Security or safety concerns
- Irrecoverable API/system errors
```

#### Input Schema

```python
@dataclass
class EvaluatorInput:
    user_question: str
    plan: Plan
    code: str
    execution_result: ExecutionResult
    
    # Iteration context
    iteration_number: int
    max_iterations: int
    previous_evaluations: List[Evaluation]
```

#### Output Schema

```python
@dataclass
class EvaluatorOutput:
    evaluation: Evaluation
    decision: Decision
    feedback: Optional[str]
    final_answer: Optional[str]
    raw_response: str
```

---

## 5. Message Passing Protocol

### 5.1 Message Types

```python
class MessageType(Enum):
    # Orchestrator messages
    START_SESSION = "start_session"
    END_SESSION = "end_session"
    START_ITERATION = "start_iteration"
    END_ITERATION = "end_iteration"
    
    # Agent requests
    PLANNER_REQUEST = "planner_request"
    CODE_REQUEST = "code_request"
    EXECUTOR_REQUEST = "executor_request"
    EVALUATOR_REQUEST = "evaluator_request"
    RAG_REQUEST = "rag_request"
    
    # Agent responses
    PLANNER_RESPONSE = "planner_response"
    CODE_RESPONSE = "code_response"
    EXECUTOR_RESPONSE = "executor_response"
    EVALUATOR_RESPONSE = "evaluator_response"
    RAG_RESPONSE = "rag_response"
    
    # Control messages
    FEEDBACK = "feedback"
    ERROR = "error"
    TIMEOUT = "timeout"

@dataclass
class Message:
    """Base message structure for inter-agent communication."""
    message_id: str
    message_type: MessageType
    sender: str
    recipient: str
    timestamp: datetime
    payload: Any
    correlation_id: str  # Links related messages
    iteration_id: int
```

### 5.2 Communication Patterns

#### Sequential Flow (Normal Operation)

```
Orchestrator ──START_ITERATION──▶ Planner
Planner ──────PLANNER_RESPONSE──▶ Orchestrator
Orchestrator ──RAG_REQUEST──────▶ RAG Retriever (if needed)
RAG Retriever ─RAG_RESPONSE─────▶ Orchestrator
Orchestrator ──CODE_REQUEST─────▶ Code Agent
Code Agent ───CODE_RESPONSE─────▶ Orchestrator
Orchestrator ──EXECUTOR_REQUEST─▶ Executor
Executor ─────EXECUTOR_RESPONSE─▶ Orchestrator
Orchestrator ──EVALUATOR_REQUEST▶ Evaluator
Evaluator ────EVALUATOR_RESPONSE▶ Orchestrator
Orchestrator ──END_ITERATION────▶ (next iteration or complete)
```

#### Feedback Loop (Retry)

```
Evaluator ──EVALUATOR_RESPONSE(RETRY)──▶ Orchestrator
                                             │
                                             ▼
Orchestrator ──START_ITERATION─────────────▶ Planner
             + FEEDBACK(from evaluator)
```

### 5.3 Message Serialization

All messages are JSON-serializable for persistence and debugging:

```python
def serialize_message(msg: Message) -> dict:
    return {
        "message_id": msg.message_id,
        "message_type": msg.message_type.value,
        "sender": msg.sender,
        "recipient": msg.recipient,
        "timestamp": msg.timestamp.isoformat(),
        "payload": serialize_payload(msg.payload),
        "correlation_id": msg.correlation_id,
        "iteration_id": msg.iteration_id
    }
```

---

## 6. Orchestrator Design

### 6.1 Orchestrator Responsibilities

1. **Lifecycle Management**: Start/stop sessions, track iterations
2. **State Management**: Persist and update session state
3. **Flow Control**: Route between agents based on decisions
4. **Resource Management**: Enforce timeouts, limits
5. **Error Recovery**: Handle failures gracefully
6. **Metrics Collection**: Track performance and usage

### 6.2 Orchestrator Interface

```python
class Orchestrator(Protocol):
    """Interface for the multi-agent orchestrator."""
    
    async def run(
        self,
        user_question: str,
        config: OrchestratorConfig
    ) -> OrchestratorResult:
        """
        Execute the multi-agent loop for a user question.
        
        Args:
            user_question: The user's input question
            config: Configuration for this run
            
        Returns:
            Final result including answer, code, and metadata
        """
        ...
    
    def get_state(self) -> SessionState:
        """Get current session state."""
        ...
    
    def cancel(self) -> None:
        """Cancel the current run."""
        ...

@dataclass
class OrchestratorConfig:
    """Configuration for orchestrator behavior."""
    max_iterations: int = 5
    iteration_timeout_seconds: int = 120
    code_execution_timeout_seconds: int = 60
    max_rag_queries_per_iteration: int = 3
    max_total_rag_queries: int = 10
    
    # LLM configurations
    planner_model: str = "gpt-4.1"
    code_model: str = "gpt-4.1"
    evaluator_model: str = "gpt-4.1"
    
    # Feature flags
    enable_code_execution: bool = True
    enable_rag: bool = True
    persist_state: bool = True

@dataclass
class OrchestratorResult:
    """Final output from orchestrator."""
    success: bool
    final_answer: Optional[str]
    final_code: Optional[str]
    
    # Metadata
    session_id: str
    iterations_used: int
    termination_reason: TerminationReason
    total_time_seconds: float
    
    # Full state for debugging
    session_state: SessionState
```

### 6.3 Orchestrator Implementation Structure

```python
class MultiAgentOrchestrator:
    """
    Orchestrates the multi-agent iterative RAG workflow.
    """
    
    def __init__(
        self,
        planner: PlannerAgent,
        code_agent: CodeAgent,
        executor: ExecutorAgent,
        evaluator: EvaluatorAgent,
        rag_retriever: RAGRetriever,
        state_manager: StateManager,
        config: OrchestratorConfig
    ):
        self.planner = planner
        self.code_agent = code_agent
        self.executor = executor
        self.evaluator = evaluator
        self.rag = rag_retriever
        self.state_manager = state_manager
        self.config = config
        self._cancelled = False
    
    async def run(
        self,
        user_question: str,
        config: Optional[OrchestratorConfig] = None
    ) -> OrchestratorResult:
        """Main entry point - see Section 9 for full pseudocode."""
        ...
```

---

## 7. Failure Handling

### 7.1 Failure Categories

| Category | Examples | Recovery Strategy |
|----------|----------|-------------------|
| **Transient** | Network timeout, rate limit | Exponential backoff + retry |
| **Agent Failure** | Invalid JSON response, empty output | Re-prompt with error context |
| **Execution Failure** | Code error, import failure | Feed error to Planner for fix |
| **Logical Failure** | Wrong approach, missing info | Evaluator feedback loop |
| **Fatal** | API key invalid, service down | Abort with error message |

### 7.2 Retry Policies

```python
@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    
    retryable_exceptions: List[Type[Exception]] = field(default_factory=lambda: [
        TimeoutError,
        ConnectionError,
        RateLimitError,
    ])

async def with_retry(
    func: Callable,
    policy: RetryPolicy,
    context: str
) -> Any:
    """Execute function with retry logic."""
    last_exception = None
    delay = policy.initial_delay_seconds
    
    for attempt in range(policy.max_retries + 1):
        try:
            return await func()
        except tuple(policy.retryable_exceptions) as e:
            last_exception = e
            if attempt < policy.max_retries:
                if policy.jitter:
                    delay *= (1 + random.uniform(-0.1, 0.1))
                await asyncio.sleep(min(delay, policy.max_delay_seconds))
                delay *= policy.exponential_base
                logger.warning(f"{context}: Attempt {attempt + 1} failed, retrying: {e}")
            else:
                logger.error(f"{context}: All {policy.max_retries + 1} attempts failed")
    
    raise last_exception
```

### 7.3 Graceful Degradation

When failures occur, the system returns partial results:

```python
@dataclass
class PartialResult:
    """Returned when system cannot complete successfully."""
    partial_answer: Optional[str]
    best_code_attempt: Optional[str]
    retrieved_documents: List[RetrievedDocument]
    error_summary: str
    recovery_suggestions: List[str]
    
def create_partial_result(state: SessionState, error: Exception) -> PartialResult:
    """Create best-effort result from current state."""
    
    # Find the best iteration (highest evaluation score)
    best_iteration = max(
        state.iterations,
        key=lambda it: (
            it.evaluation.correctness_score 
            if it.evaluation else 0
        ),
        default=None
    )
    
    return PartialResult(
        partial_answer=_extract_partial_answer(best_iteration),
        best_code_attempt=best_iteration.generated_code if best_iteration else None,
        retrieved_documents=list(state.all_retrieved_docs.values()),
        error_summary=str(error),
        recovery_suggestions=_generate_suggestions(state, error)
    )
```

### 7.4 Timeout Handling

```python
class TimeoutManager:
    """Manages timeouts at different levels."""
    
    def __init__(self, config: OrchestratorConfig):
        self.iteration_timeout = config.iteration_timeout_seconds
        self.execution_timeout = config.code_execution_timeout_seconds
        self.llm_timeout = 60  # Per-LLM-call timeout
    
    @asynccontextmanager
    async def iteration_scope(self):
        """Context manager for iteration-level timeout."""
        try:
            async with asyncio.timeout(self.iteration_timeout):
                yield
        except asyncio.TimeoutError:
            raise IterationTimeoutError(
                f"Iteration exceeded {self.iteration_timeout}s timeout"
            )
    
    @asynccontextmanager
    async def execution_scope(self):
        """Context manager for code execution timeout."""
        try:
            async with asyncio.timeout(self.execution_timeout):
                yield
        except asyncio.TimeoutError:
            raise ExecutionTimeoutError(
                f"Code execution exceeded {self.execution_timeout}s timeout"
            )
```

### 7.5 Error Response Templates

```python
ERROR_TEMPLATES = {
    "api_key_invalid": {
        "message": "API key is invalid or expired",
        "recovery": "Please check your API key configuration",
        "abort": True
    },
    "rate_limited": {
        "message": "API rate limit exceeded",
        "recovery": "Will retry after backoff period",
        "abort": False
    },
    "code_execution_failed": {
        "message": "Code execution failed: {error}",
        "recovery": "Feeding error back to planner for correction",
        "abort": False
    },
    "max_iterations_reached": {
        "message": "Maximum iterations ({max}) reached without success",
        "recovery": "Returning best partial result",
        "abort": True
    },
    "unsafe_code_detected": {
        "message": "Generated code contains unsafe operations",
        "recovery": "Code execution blocked for safety",
        "abort": True
    }
}
```

---

## 8. Interface Definitions

### 8.1 Agent Base Interface

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')

class Agent(ABC, Generic[InputT, OutputT]):
    """Base interface for all agents."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Agent identifier."""
        ...
    
    @abstractmethod
    async def invoke(self, input: InputT) -> OutputT:
        """
        Process input and produce output.
        
        Args:
            input: Agent-specific input dataclass
            
        Returns:
            Agent-specific output dataclass
            
        Raises:
            AgentError: On processing failure
        """
        ...
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the agent's system prompt."""
        ...
```

### 8.2 Concrete Agent Interfaces

```python
class PlannerAgent(Agent[PlannerInput, PlannerOutput]):
    """Interface for the Planner agent."""
    
    @property
    def name(self) -> str:
        return "planner"
    
    async def invoke(self, input: PlannerInput) -> PlannerOutput:
        ...

class CodeAgent(Agent[CodeAgentInput, CodeAgentOutput]):
    """Interface for the Code agent."""
    
    @property
    def name(self) -> str:
        return "code_agent"
    
    async def invoke(self, input: CodeAgentInput) -> CodeAgentOutput:
        ...

class ExecutorAgent(Agent[ExecutorInput, ExecutorOutput]):
    """Interface for the Executor agent."""
    
    @property
    def name(self) -> str:
        return "executor"
    
    async def invoke(self, input: ExecutorInput) -> ExecutorOutput:
        ...

class EvaluatorAgent(Agent[EvaluatorInput, EvaluatorOutput]):
    """Interface for the Evaluator agent."""
    
    @property
    def name(self) -> str:
        return "evaluator"
    
    async def invoke(self, input: EvaluatorInput) -> EvaluatorOutput:
        ...
```

### 8.3 RAG Retriever Interface

```python
class RAGRetrieverInterface(Protocol):
    """Interface for RAG retrieval."""
    
    async def retrieve(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> List[RetrievedDocument]:
        """
        Retrieve relevant documents for multiple queries.
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
            
        Returns:
            Deduplicated list of retrieved documents
        """
        ...
    
    def retrieve_context(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> str:
        """
        Retrieve and format context string for LLM prompts.
        
        Returns:
            Formatted string of retrieved documents
        """
        ...
```

### 8.4 State Manager Interface

```python
class StateManager(Protocol):
    """Interface for state persistence."""
    
    def create_session(
        self,
        user_question: str,
        config: OrchestratorConfig
    ) -> SessionState:
        """Create a new session."""
        ...
    
    def save_iteration(
        self,
        session_id: str,
        iteration: IterationState
    ) -> None:
        """Persist an iteration."""
        ...
    
    def get_session(self, session_id: str) -> SessionState:
        """Retrieve session state."""
        ...
    
    def update_session(
        self,
        session_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """Update session fields."""
        ...
    
    def add_retrieved_docs(
        self,
        session_id: str,
        docs: List[RetrievedDocument]
    ) -> None:
        """Add documents to accumulated context."""
        ...
```

### 8.5 Executor Sandbox Interface

```python
class CodeSandbox(Protocol):
    """Interface for sandboxed code execution."""
    
    async def execute(
        self,
        code: str,
        timeout_seconds: int,
        allowed_imports: List[str],
        working_directory: str
    ) -> ExecutionResult:
        """
        Execute code in isolated environment.
        
        Args:
            code: Python code to execute
            timeout_seconds: Maximum execution time
            allowed_imports: Whitelist of importable modules
            working_directory: Directory for file operations
            
        Returns:
            ExecutionResult with stdout, stderr, artifacts
        """
        ...
    
    def cleanup(self) -> None:
        """Clean up sandbox resources."""
        ...
```

---

## 9. Pseudocode Implementation

### 9.1 Main Orchestrator Loop

```python
class MultiAgentOrchestrator:
    """
    Full implementation of the multi-agent orchestrator.
    """
    
    async def run(
        self,
        user_question: str,
        config: Optional[OrchestratorConfig] = None
    ) -> OrchestratorResult:
        """
        Execute the multi-agent loop.
        
        This is the main entry point that coordinates all agents
        through the iterative refinement process.
        """
        config = config or self.config
        start_time = time.time()
        
        # Initialize session state
        session = self.state_manager.create_session(user_question, config)
        
        try:
            # Main iteration loop
            while session.current_iteration < config.max_iterations:
                if self._cancelled:
                    return self._create_cancelled_result(session, start_time)
                
                # Run one iteration
                iteration_result = await self._run_iteration(
                    session=session,
                    user_question=user_question,
                    config=config
                )
                
                # Save iteration state
                self.state_manager.save_iteration(session.session_id, iteration_result)
                session.iterations.append(iteration_result)
                session.current_iteration += 1
                
                # Check termination condition
                if iteration_result.decision == Decision.SUCCESS:
                    return self._create_success_result(
                        session=session,
                        final_answer=iteration_result.evaluation.final_answer,
                        final_code=iteration_result.generated_code,
                        start_time=start_time
                    )
                
                if iteration_result.decision == Decision.ABORT:
                    return self._create_abort_result(
                        session=session,
                        reason=iteration_result.feedback,
                        start_time=start_time
                    )
                
                # Decision is RETRY - continue loop with feedback
                logger.info(f"Iteration {session.current_iteration}: RETRY requested")
            
            # Max iterations reached
            return self._create_max_iterations_result(session, start_time)
            
        except Exception as e:
            logger.exception(f"Orchestrator error: {e}")
            return self._create_error_result(session, e, start_time)
        
        finally:
            session.end_time = datetime.now()
            self.state_manager.update_session(
                session.session_id,
                {"end_time": session.end_time}
            )
    
    async def _run_iteration(
        self,
        session: SessionState,
        user_question: str,
        config: OrchestratorConfig
    ) -> IterationState:
        """
        Execute a single iteration of the agent loop.
        
        Flow: Planner -> [RAG] -> Code Agent -> Executor -> Evaluator
        """
        iteration_id = session.current_iteration
        
        async with self.timeout_manager.iteration_scope():
            # Get previous iteration context (if any)
            prev_iteration = (
                session.iterations[-1] 
                if session.iterations else None
            )
            
            # ========== STEP 1: PLANNER ==========
            planner_input = PlannerInput(
                user_question=user_question,
                iteration_number=iteration_id,
                previous_feedback=(
                    prev_iteration.feedback 
                    if prev_iteration else None
                ),
                previous_plan=(
                    prev_iteration.plan 
                    if prev_iteration else None
                ),
                previous_code=(
                    prev_iteration.generated_code 
                    if prev_iteration else None
                ),
                previous_execution_result=(
                    prev_iteration.execution_result 
                    if prev_iteration else None
                ),
                all_retrieved_docs=list(session.all_retrieved_docs.values())
            )
            
            planner_output = await with_retry(
                lambda: self.planner.invoke(planner_input),
                self.retry_policy,
                context=f"Planner (iteration {iteration_id})"
            )
            
            plan = planner_output.plan
            
            # ========== STEP 2: RAG RETRIEVAL ==========
            retrieved_docs = []
            if plan.rag_needed and plan.rag_queries:
                # Limit queries per iteration
                queries = plan.rag_queries[:config.max_rag_queries_per_iteration]
                
                retrieved_docs = await with_retry(
                    lambda: self.rag.retrieve(queries, top_k=5),
                    self.retry_policy,
                    context=f"RAG retrieval (iteration {iteration_id})"
                )
                
                # Add to accumulated docs (deduped by doc_id)
                self.state_manager.add_retrieved_docs(
                    session.session_id,
                    retrieved_docs
                )
                for doc in retrieved_docs:
                    session.all_retrieved_docs[doc.doc_id] = doc
                
                session.total_rag_calls += len(queries)
            
            # ========== STEP 3: CODE AGENT ==========
            # Build context from all retrieved docs
            context = self._format_context(
                list(session.all_retrieved_docs.values())
            )
            
            code_input = CodeAgentInput(
                plan=plan,
                code_requirements=plan.code_requirements,
                retrieved_context=context,
                previous_code=(
                    prev_iteration.generated_code 
                    if prev_iteration else None
                ),
                previous_error=(
                    prev_iteration.execution_result.error_traceback
                    if prev_iteration and not prev_iteration.execution_result.success
                    else None
                ),
                previous_feedback=(
                    prev_iteration.feedback 
                    if prev_iteration else None
                )
            )
            
            code_output = await with_retry(
                lambda: self.code_agent.invoke(code_input),
                self.retry_policy,
                context=f"Code Agent (iteration {iteration_id})"
            )
            
            session.total_llm_calls += 2  # Planner + Code Agent
            
            # ========== STEP 4: EXECUTOR ==========
            execution_result = ExecutionResult(
                success=False,
                stdout="",
                stderr="Code execution disabled",
                return_value=None,
                artifacts={},
                execution_time_ms=0,
                memory_usage_mb=0,
                error_type=None,
                error_traceback=None
            )
            
            if config.enable_code_execution and plan.code_needed:
                executor_input = ExecutorInput(
                    code=code_output.code,
                    timeout_seconds=config.code_execution_timeout_seconds,
                    working_directory=f"work/session_{session.session_id}"
                )
                
                async with self.timeout_manager.execution_scope():
                    executor_output = await self.executor.invoke(executor_input)
                    execution_result = executor_output.result
                
                session.total_code_executions += 1
            
            # ========== STEP 5: EVALUATOR ==========
            evaluator_input = EvaluatorInput(
                user_question=user_question,
                plan=plan,
                code=code_output.code,
                execution_result=execution_result,
                iteration_number=iteration_id,
                max_iterations=config.max_iterations,
                previous_evaluations=[
                    it.evaluation for it in session.iterations
                    if it.evaluation is not None
                ]
            )
            
            evaluator_output = await with_retry(
                lambda: self.evaluator.invoke(evaluator_input),
                self.retry_policy,
                context=f"Evaluator (iteration {iteration_id})"
            )
            
            session.total_llm_calls += 1  # Evaluator
            
            # ========== BUILD ITERATION STATE ==========
            return IterationState(
                iteration_id=iteration_id,
                timestamp=datetime.now(),
                plan=plan,
                rag_queries=plan.rag_queries if plan.rag_needed else [],
                retrieved_documents=retrieved_docs,
                generated_code=code_output.code,
                code_version=iteration_id,
                execution_result=execution_result,
                evaluation=evaluator_output.evaluation,
                decision=evaluator_output.decision,
                feedback=evaluator_output.feedback
            )
    
    def _format_context(self, docs: List[RetrievedDocument]) -> str:
        """Format retrieved documents into context string."""
        if not docs:
            return ""
        
        # Sort by relevance score
        sorted_docs = sorted(docs, key=lambda d: d.relevance_score, reverse=True)
        
        context_parts = []
        for i, doc in enumerate(sorted_docs[:10]):  # Limit to top 10
            context_parts.append(
                f"[Document {i+1}] (Source: {doc.source}, Score: {doc.relevance_score:.2f})\n"
                f"{doc.text}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def _create_success_result(
        self,
        session: SessionState,
        final_answer: str,
        final_code: str,
        start_time: float
    ) -> OrchestratorResult:
        """Create result for successful completion."""
        session.final_answer = final_answer
        session.final_code = final_code
        session.termination_reason = TerminationReason.SUCCESS
        
        return OrchestratorResult(
            success=True,
            final_answer=final_answer,
            final_code=final_code,
            session_id=session.session_id,
            iterations_used=session.current_iteration,
            termination_reason=TerminationReason.SUCCESS,
            total_time_seconds=time.time() - start_time,
            session_state=session
        )
    
    def _create_abort_result(
        self,
        session: SessionState,
        reason: str,
        start_time: float
    ) -> OrchestratorResult:
        """Create result for aborted run."""
        session.termination_reason = TerminationReason.ABORT_REQUESTED
        
        # Try to extract best partial result
        partial = create_partial_result(session, AbortError(reason))
        
        return OrchestratorResult(
            success=False,
            final_answer=partial.partial_answer,
            final_code=partial.best_code_attempt,
            session_id=session.session_id,
            iterations_used=session.current_iteration,
            termination_reason=TerminationReason.ABORT_REQUESTED,
            total_time_seconds=time.time() - start_time,
            session_state=session
        )
    
    def _create_max_iterations_result(
        self,
        session: SessionState,
        start_time: float
    ) -> OrchestratorResult:
        """Create result when max iterations reached."""
        session.termination_reason = TerminationReason.MAX_ITERATIONS
        
        # Return best result from all iterations
        best_iteration = max(
            session.iterations,
            key=lambda it: (
                it.evaluation.correctness_score
                if it.evaluation else 0
            )
        )
        
        return OrchestratorResult(
            success=False,  # Did not achieve SUCCESS decision
            final_answer=self._extract_best_answer(best_iteration),
            final_code=best_iteration.generated_code,
            session_id=session.session_id,
            iterations_used=session.current_iteration,
            termination_reason=TerminationReason.MAX_ITERATIONS,
            total_time_seconds=time.time() - start_time,
            session_state=session
        )
    
    def _create_error_result(
        self,
        session: SessionState,
        error: Exception,
        start_time: float
    ) -> OrchestratorResult:
        """Create result for fatal error."""
        session.termination_reason = TerminationReason.FATAL_ERROR
        partial = create_partial_result(session, error)
        
        return OrchestratorResult(
            success=False,
            final_answer=f"Error: {partial.error_summary}",
            final_code=partial.best_code_attempt,
            session_id=session.session_id,
            iterations_used=session.current_iteration,
            termination_reason=TerminationReason.FATAL_ERROR,
            total_time_seconds=time.time() - start_time,
            session_state=session
        )
```

### 9.2 Agent Implementation Examples

#### Planner Agent Implementation

```python
class PlannerAgentImpl(PlannerAgent):
    """Concrete implementation of the Planner agent."""
    
    def __init__(self, llm_client: LLMClient, config: AgentConfig):
        self.llm = llm_client
        self.config = config
        self._system_prompt = self._load_system_prompt()
    
    def _load_system_prompt(self) -> str:
        """Load planner system prompt from file."""
        with open("prompts/planner_agent.txt", "r") as f:
            return f.read()
    
    def get_system_prompt(self) -> str:
        return self._system_prompt
    
    async def invoke(self, input: PlannerInput) -> PlannerOutput:
        """Generate execution plan from user question and context."""
        
        # Build user message with all context
        user_message = self._build_user_message(input)
        
        # Call LLM
        response = await self.llm.chat(
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        # Parse response into Plan
        try:
            plan_dict = json.loads(response.content)
            plan = self._parse_plan(plan_dict, input.iteration_number)
            return PlannerOutput(plan=plan, raw_response=response.content)
        except (json.JSONDecodeError, KeyError) as e:
            raise AgentError(f"Planner returned invalid JSON: {e}")
    
    def _build_user_message(self, input: PlannerInput) -> str:
        """Construct the user message for the planner."""
        parts = [f"## User Question\n{input.user_question}"]
        
        if input.iteration_number > 0:
            parts.append(f"\n## Iteration\nThis is iteration {input.iteration_number}.")
            
            if input.previous_feedback:
                parts.append(f"\n## Feedback from Previous Iteration\n{input.previous_feedback}")
            
            if input.previous_execution_result:
                result = input.previous_execution_result
                parts.append(f"\n## Previous Execution Result")
                parts.append(f"Success: {result.success}")
                if result.stdout:
                    parts.append(f"Stdout:\n```\n{result.stdout[:2000]}\n```")
                if result.stderr:
                    parts.append(f"Stderr:\n```\n{result.stderr[:2000]}\n```")
                if result.error_traceback:
                    parts.append(f"Error:\n```\n{result.error_traceback[:2000]}\n```")
        
        if input.all_retrieved_docs:
            parts.append(f"\n## Previously Retrieved Documents ({len(input.all_retrieved_docs)} docs)")
            parts.append("You already have access to these documents. Consider whether additional retrieval is needed.")
        
        return "\n".join(parts)
    
    def _parse_plan(self, plan_dict: dict, iteration: int) -> Plan:
        """Parse JSON response into Plan dataclass."""
        return Plan(
            plan_id=f"plan_{iteration}_{uuid.uuid4().hex[:8]}",
            version=iteration,
            sub_tasks=[
                SubTask(
                    task_id=t["task_id"],
                    description=t["description"],
                    task_type=TaskType[t["task_type"]],
                    dependencies=t.get("dependencies", [])
                )
                for t in plan_dict.get("sub_tasks", [])
            ],
            rag_needed=plan_dict.get("rag_needed", False),
            rag_queries=plan_dict.get("rag_queries", []),
            code_needed=plan_dict.get("code_needed", True),
            code_requirements=plan_dict.get("code_requirements", []),
            acceptance_criteria=plan_dict.get("acceptance_criteria", []),
            uses_previous_context=plan_dict.get("uses_previous_context", False),
            referenced_iterations=plan_dict.get("referenced_iterations", [])
        )
```

#### Executor Agent Implementation

```python
class ExecutorAgentImpl(ExecutorAgent):
    """Executes code in a sandboxed environment."""
    
    def __init__(self, sandbox: CodeSandbox, config: ExecutorConfig):
        self.sandbox = sandbox
        self.config = config
    
    @property
    def name(self) -> str:
        return "executor"
    
    def get_system_prompt(self) -> str:
        return ""  # Executor doesn't use LLM
    
    async def invoke(self, input: ExecutorInput) -> ExecutorOutput:
        """Execute code and capture results."""
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        
        # Ensure working directory exists
        work_dir = Path(input.working_directory)
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate code safety (basic checks)
        safety_issues = self._check_code_safety(input.code)
        if safety_issues:
            return ExecutorOutput(
                result=ExecutionResult(
                    success=False,
                    stdout="",
                    stderr=f"Code safety check failed: {', '.join(safety_issues)}",
                    return_value=None,
                    artifacts={},
                    execution_time_ms=0,
                    memory_usage_mb=0,
                    error_type="SafetyError",
                    error_traceback=None
                ),
                execution_id=execution_id
            )
        
        # Execute in sandbox
        try:
            result = await self.sandbox.execute(
                code=input.code,
                timeout_seconds=input.timeout_seconds,
                allowed_imports=input.allowed_imports,
                working_directory=str(work_dir)
            )
            return ExecutorOutput(result=result, execution_id=execution_id)
            
        except asyncio.TimeoutError:
            return ExecutorOutput(
                result=ExecutionResult(
                    success=False,
                    stdout="",
                    stderr=f"Execution timed out after {input.timeout_seconds}s",
                    return_value=None,
                    artifacts={},
                    execution_time_ms=input.timeout_seconds * 1000,
                    memory_usage_mb=0,
                    error_type="TimeoutError",
                    error_traceback=None
                ),
                execution_id=execution_id
            )
    
    def _check_code_safety(self, code: str) -> List[str]:
        """Basic safety checks for code."""
        issues = []
        
        dangerous_patterns = [
            (r'\bos\.system\s*\(', "os.system calls"),
            (r'\bsubprocess\b', "subprocess module"),
            (r'\b__import__\s*\(', "dynamic imports"),
            (r'\beval\s*\(', "eval() calls"),
            (r'\bexec\s*\(', "exec() calls"),
            (r'\bopen\s*\([^)]*["\']w["\']', "file write operations"),
            (r'\brm\s+-rf', "destructive shell commands"),
        ]
        
        for pattern, description in dangerous_patterns:
            if re.search(pattern, code):
                issues.append(description)
        
        return issues
```

---

## 10. Configuration

### 10.1 Default Configuration

```python
DEFAULT_CONFIG = OrchestratorConfig(
    # Iteration limits
    max_iterations=5,
    iteration_timeout_seconds=120,
    code_execution_timeout_seconds=60,
    
    # RAG limits
    max_rag_queries_per_iteration=3,
    max_total_rag_queries=10,
    
    # LLM models
    planner_model="gpt-4.1",
    code_model="gpt-4.1",
    evaluator_model="gpt-4.1",
    
    # Features
    enable_code_execution=True,
    enable_rag=True,
    persist_state=True
)
```

### 10.2 Configuration File Schema

```yaml
# config/multi_agent_config.yaml

orchestrator:
  max_iterations: 5
  iteration_timeout_seconds: 120
  code_execution_timeout_seconds: 60
  max_rag_queries_per_iteration: 3
  max_total_rag_queries: 10

models:
  planner:
    provider: "openai"
    model: "gpt-4.1"
    temperature: 0.2
    max_tokens: 2000
  
  code_agent:
    provider: "openai"
    model: "gpt-4.1"
    temperature: 0.0
    max_tokens: 4000
  
  evaluator:
    provider: "openai"
    model: "gpt-4.1"
    temperature: 0.1
    max_tokens: 2000

rag:
  db_path: "QAMP/data/qamp.db"
  embed_model: "gemini-embedding-001"
  default_top_k: 5

executor:
  working_directory: "work"
  allowed_imports:
    - qiskit
    - numpy
    - matplotlib
    - scipy
    - math
    - json
  max_memory_mb: 512

retry_policy:
  max_retries: 3
  initial_delay_seconds: 1.0
  max_delay_seconds: 30.0
  exponential_base: 2.0

logging:
  level: "INFO"
  file: "logs/multi_agent.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### 10.3 Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...  # For Gemini embeddings

# Optional overrides
ORCHESTRATOR_MAX_ITERATIONS=5
ORCHESTRATOR_TIMEOUT=120
CODE_EXECUTION_ENABLED=true
RAG_ENABLED=true
LOG_LEVEL=INFO
```

---

## Appendix A: File Structure

```
QAMP2025/
├── agents/
│   ├── __init__.py
│   ├── base.py              # Agent base classes and interfaces
│   ├── planner.py           # Planner agent implementation
│   ├── code_agent.py        # Code agent implementation
│   ├── executor.py          # Executor agent implementation
│   ├── evaluator.py         # Evaluator agent implementation
│   └── sandbox.py           # Code sandbox implementation
├── orchestrator/
│   ├── __init__.py
│   ├── orchestrator.py      # Main orchestrator
│   ├── state_manager.py     # State persistence
│   ├── message_bus.py       # Inter-agent messaging
│   └── timeout_manager.py   # Timeout handling
├── models/
│   ├── __init__.py
│   ├── state.py             # State dataclasses
│   ├── messages.py          # Message types
│   └── config.py            # Configuration models
├── prompts/
│   ├── planner_agent.txt    # Planner system prompt
│   ├── code_agent.txt       # Code agent system prompt
│   └── evaluator_agent.txt  # Evaluator system prompt
├── config/
│   └── multi_agent_config.yaml
└── tests/
    ├── test_orchestrator.py
    ├── test_agents.py
    └── test_integration.py
```

---

## Appendix B: Sequence Diagram

```
User          Orchestrator    Planner     RAG      CodeAgent   Executor   Evaluator
 │                │              │         │           │          │           │
 │──Question─────▶│              │         │           │          │           │
 │                │──Plan Req───▶│         │           │          │           │
 │                │◀──Plan+RAQ───│         │           │          │           │
 │                │──RAG Query──────────▶│           │          │           │
 │                │◀──Documents──────────│           │          │           │
 │                │──Code Req─────────────────────▶│          │           │
 │                │◀──Code────────────────────────│          │           │
 │                │──Execute──────────────────────────────▶│           │
 │                │◀──Result───────────────────────────────│           │
 │                │──Evaluate──────────────────────────────────────▶│
 │                │◀──Decision─────────────────────────────────────│
 │                │                                                  │
 │                │  [If RETRY: Loop back to Planner with feedback] │
 │                │                                                  │
 │◀──Answer───────│              │         │           │          │           │
 │                │              │         │           │          │           │
```

---

## Appendix C: Implementation Checkpoints

This section defines **testable checkpoints** for incremental implementation. Each checkpoint has specific acceptance criteria and verification steps.

---

### Phase 1: Core Infrastructure

#### Checkpoint 1.1: State Models & Dataclasses
**Goal**: All state models from Section 3 are implemented and serializable.

| Item | Verification |
|------|--------------|
| `IterationState` dataclass | Unit test: create instance, serialize to JSON, deserialize, assert equality |
| `SessionState` dataclass | Unit test: create with nested `IterationState`, verify all fields accessible |
| `Plan`, `SubTask`, `RetrievedDocument` | Unit test: instantiate each, verify JSON round-trip |
| `ExecutionResult`, `Evaluation` | Unit test: instantiate with all optional fields as `None` and with values |
| `Decision`, `TerminationReason`, `TaskType` enums | Unit test: verify all enum values parse correctly from strings |

**Test Command**:
```bash
pytest tests/test_models.py -v --tb=short
```

**Acceptance Criteria**:
- [ ] All dataclasses instantiate without errors
- [ ] JSON serialization/deserialization works for all models
- [ ] Optional fields handle `None` gracefully
- [ ] Enums serialize as strings and parse back correctly

---

#### Checkpoint 1.2: Agent Base Interfaces
**Goal**: Abstract base classes from Section 8.1-8.2 are implemented.

| Item | Verification |
|------|--------------|
| `Agent[InputT, OutputT]` ABC | Verify abstract methods: `name`, `invoke`, `get_system_prompt` |
| `PlannerAgent` interface | Create mock implementation, verify type hints work |
| `CodeAgent` interface | Create mock implementation, verify type hints work |
| `ExecutorAgent` interface | Create mock implementation, verify type hints work |
| `EvaluatorAgent` interface | Create mock implementation, verify type hints work |

**Test Command**:
```bash
pytest tests/test_agents.py::TestAgentInterfaces -v
```

**Acceptance Criteria**:
- [ ] Cannot instantiate abstract `Agent` class directly (raises `TypeError`)
- [ ] Concrete implementations must implement all abstract methods
- [ ] Type checking passes with `mypy agents/`

---

#### Checkpoint 1.3: State Manager with Persistence
**Goal**: State manager from Section 8.4 persists sessions to disk/database.

| Item | Verification |
|------|--------------|
| `create_session()` | Creates new session, returns valid `SessionState` with UUID |
| `save_iteration()` | Saves iteration, retrievable after save |
| `get_session()` | Returns correct session state by ID |
| `update_session()` | Updates fields, persists changes |
| `add_retrieved_docs()` | Adds docs, deduplicates by `doc_id` |
| Persistence across restarts | Create session, restart process, retrieve session |

**Test Command**:
```bash
pytest tests/test_state_manager.py -v
```

**Acceptance Criteria**:
- [ ] Sessions persist to SQLite or JSON files
- [ ] All CRUD operations work correctly
- [ ] Document deduplication works (same `doc_id` = single entry)
- [ ] Session state survives process restart

---

### Phase 2: Agent Implementation

#### Checkpoint 2.1: Planner Agent
**Goal**: Planner agent produces valid plans from user questions.

| Item | Verification |
|------|--------------|
| System prompt loads | Verify prompt file exists and loads without error |
| First iteration planning | Input: simple Qiskit question → Output: valid `Plan` with sub_tasks |
| RAG query generation | Plan includes `rag_needed=True` and non-empty `rag_queries` for doc-needing questions |
| Retry planning | Input with `previous_feedback` → Output: modified plan addressing feedback |
| JSON schema compliance | All outputs parse to `Plan` dataclass without errors |

**Test Command**:
```bash
pytest tests/test_agents.py::TestPlannerAgent -v
# Manual verification with real LLM:
python -m agents.planner --question "How do I create a Bell state in Qiskit?" --debug
```

**Acceptance Criteria**:
- [ ] Generates valid JSON matching schema for 5 different question types
- [ ] `rag_queries` are specific (not generic like "Qiskit documentation")
- [ ] `acceptance_criteria` are objective and verifiable
- [ ] Retry plans incorporate previous feedback

---

#### Checkpoint 2.2: Code Agent
**Goal**: Code agent generates executable Python code from plans.

| Item | Verification |
|------|--------------|
| System prompt loads | Verify prompt file exists and loads |
| Code generation from plan | Input: plan + context → Output: Python code block |
| Code is self-contained | Generated code includes all imports |
| Uses retrieved context | Code references APIs mentioned in context |
| Retry with error | Input with `previous_error` → Output: fixed code |

**Test Command**:
```bash
pytest tests/test_agents.py::TestCodeAgent -v
# Manual verification:
python -m agents.code_agent --plan-file test_plan.json --context-file test_context.txt --debug
```

**Acceptance Criteria**:
- [ ] Generated code is syntactically valid Python (no `SyntaxError`)
- [ ] Code includes `qiskit` imports when needed
- [ ] No placeholder values (e.g., `...`, `TODO`, `pass` without implementation)
- [ ] Retry generates different code addressing the error

---

#### Checkpoint 2.3: Executor Agent with Sandbox
**Goal**: Executor runs code safely and captures all outputs.

| Item | Verification |
|------|--------------|
| Successful execution | Simple print statement → `success=True`, stdout captured |
| Timeout enforcement | Infinite loop → `success=False`, `error_type="TimeoutError"` |
| Error capture | Code with exception → stderr + `error_traceback` populated |
| Artifact collection | Code saves file → artifact in `artifacts` dict |
| Safety checks | Code with `os.system()` → execution blocked |
| Resource limits | Memory-heavy code → enforced limit |

**Test Command**:
```bash
pytest tests/test_agents.py::TestExecutorAgent -v
pytest tests/test_sandbox.py -v
```

**Acceptance Criteria**:
- [ ] Code executes in isolated environment (no access to parent process)
- [ ] Timeout works (test with 2s timeout on 5s sleep)
- [ ] Dangerous patterns blocked: `os.system`, `subprocess`, `eval`, `exec`
- [ ] Files created in working directory are captured as artifacts
- [ ] stdout/stderr separated correctly

---

#### Checkpoint 2.4: Evaluator Agent
**Goal**: Evaluator correctly assesses results and makes decisions.

| Item | Verification |
|------|--------------|
| System prompt loads | Verify prompt file exists and loads |
| SUCCESS decision | Correct output + all criteria met → `Decision.SUCCESS` |
| RETRY decision | Partial failure + fixable → `Decision.RETRY` + actionable feedback |
| ABORT decision | Fundamental failure → `Decision.ABORT` + explanation |
| Criteria evaluation | Each `acceptance_criteria` has corresponding `criteria_met` entry |
| Score generation | `correctness_score`, `completeness_score`, `code_quality_score` all 0-1 |

**Test Command**:
```bash
pytest tests/test_agents.py::TestEvaluatorAgent -v
# Manual verification:
python -m agents.evaluator --execution-result-file test_result.json --debug
```

**Acceptance Criteria**:
- [ ] JSON output matches schema
- [ ] SUCCESS only when `answers_question=True` and `code_executes=True`
- [ ] RETRY includes specific, actionable `feedback`
- [ ] ABORT only for irrecoverable failures
- [ ] Scores are calibrated (not all 1.0 or all 0.0)

---

### Phase 3: Orchestrator

#### Checkpoint 3.1: Basic Orchestrator Loop
**Goal**: Orchestrator runs single iteration successfully.

| Item | Verification |
|------|--------------|
| Session initialization | `run()` creates new session with valid ID |
| Single iteration flow | Planner → Code Agent → Executor → Evaluator sequence completes |
| State persistence | After iteration, `get_state()` returns complete iteration data |
| SUCCESS termination | Mock evaluator returns SUCCESS → orchestrator returns success result |

**Test Command**:
```bash
pytest tests/test_orchestrator.py::TestOrchestratorSingleIteration -v
```

**Acceptance Criteria**:
- [ ] All agents called in correct order
- [ ] Session state contains one complete iteration
- [ ] Result includes `final_answer` and `final_code`
- [ ] `iterations_used == 1`

---

#### Checkpoint 3.2: Retry Loop
**Goal**: Orchestrator correctly handles RETRY decisions.

| Item | Verification |
|------|--------------|
| Retry triggers new iteration | Evaluator RETRY → Planner called again with feedback |
| Feedback passed | `previous_feedback` populated in second iteration |
| Context accumulates | Second iteration sees first iteration's retrieved docs |
| Multiple retries | 3 RETRYs → 3 iterations in state |

**Test Command**:
```bash
pytest tests/test_orchestrator.py::TestOrchestratorRetry -v
```

**Acceptance Criteria**:
- [ ] Feedback from evaluator reaches planner
- [ ] Each iteration builds on previous context
- [ ] State contains all iterations with incrementing IDs
- [ ] `total_llm_calls` increases correctly

---

#### Checkpoint 3.3: Termination Conditions
**Goal**: All termination paths work correctly.

| Item | Verification |
|------|--------------|
| SUCCESS path | Evaluator SUCCESS → `termination_reason=SUCCESS`, `success=True` |
| ABORT path | Evaluator ABORT → `termination_reason=ABORT_REQUESTED`, `success=False` |
| Max iterations | 5 RETRYs → `termination_reason=MAX_ITERATIONS`, best result returned |
| Cancellation | `cancel()` mid-run → `termination_reason=USER_CANCELLED` |
| Fatal error | Agent throws exception → `termination_reason=FATAL_ERROR`, partial result |

**Test Command**:
```bash
pytest tests/test_orchestrator.py::TestOrchestratorTermination -v
```

**Acceptance Criteria**:
- [ ] Each termination reason produces correct result structure
- [ ] Max iterations returns best iteration (highest score)
- [ ] Partial results include whatever context was gathered
- [ ] Cancellation is responsive (< 1s to stop)

---

#### Checkpoint 3.4: Timeout Management
**Goal**: Timeouts enforced at all levels.

| Item | Verification |
|------|--------------|
| Iteration timeout | Slow agent → `IterationTimeoutError` after configured seconds |
| Execution timeout | Slow code → `ExecutionTimeoutError` |
| Graceful handling | Timeout → partial result, not crash |

**Test Command**:
```bash
pytest tests/test_orchestrator.py::TestOrchestratorTimeouts -v
```

**Acceptance Criteria**:
- [ ] Iteration timeout triggers after `iteration_timeout_seconds`
- [ ] Code execution timeout independent of iteration timeout
- [ ] Timeouts produce valid `OrchestratorResult` with appropriate reason

---

#### Checkpoint 3.5: Retry Policies & Error Recovery
**Goal**: Transient failures recovered via retry.

| Item | Verification |
|------|--------------|
| Network retry | Mock 2 failures then success → operation succeeds |
| Rate limit retry | Mock 429 then success → operation succeeds with backoff |
| Exponential backoff | Delays increase between retries |
| Max retries exceeded | All retries fail → appropriate error propagated |

**Test Command**:
```bash
pytest tests/test_orchestrator.py::TestOrchestratorRetryPolicy -v
```

**Acceptance Criteria**:
- [ ] Transient errors (network, rate limit) retried automatically
- [ ] Backoff increases exponentially (1s, 2s, 4s, ...)
- [ ] Non-retryable errors fail immediately
- [ ] Logs indicate retry attempts

---

### Phase 4: Integration

#### Checkpoint 4.1: RAG Integration
**Goal**: Orchestrator integrates with existing RAG retriever.

| Item | Verification |
|------|--------------|
| RAG retrieval in flow | Planner requests RAG → documents retrieved from SQLite |
| Document deduplication | Same doc requested twice → only one copy in context |
| Query limit enforcement | > `max_rag_queries_per_iteration` queries → truncated |
| Total query limit | > `max_total_rag_queries` across session → subsequent queries skipped |
| Context formatting | Retrieved docs formatted as specified in Section 9.1 |

**Test Command**:
```bash
pytest tests/test_integration.py::TestRAGIntegration -v
# Manual verification:
python -m orchestrator.orchestrator --question "Explain SamplerV2" --debug
```

**Acceptance Criteria**:
- [ ] RAG queries hit actual SQLite database
- [ ] Retrieved documents have valid `relevance_score`
- [ ] Query limits enforced correctly
- [ ] Context string includes document metadata

---

#### Checkpoint 4.2: Configuration Management
**Goal**: Configuration loads from file and environment.

| Item | Verification |
|------|--------------|
| YAML config loading | Config file parsed, all fields populated |
| Environment override | `ORCHESTRATOR_MAX_ITERATIONS=3` → config uses 3 |
| Default fallbacks | Missing config values → defaults from Section 10.1 |
| Model configuration | Different models configurable per agent |

**Test Command**:
```bash
pytest tests/test_config.py -v
# Manual verification:
ORCHESTRATOR_MAX_ITERATIONS=2 python -m orchestrator.orchestrator --question "test" --debug
```

**Acceptance Criteria**:
- [ ] YAML config file parsed correctly
- [ ] Environment variables override config file
- [ ] Missing values default correctly
- [ ] Invalid config produces clear error message

---

#### Checkpoint 4.3: Streamlit UI Integration
**Goal**: Multi-agent mode accessible from existing Streamlit app.

| Item | Verification |
|------|--------------|
| Mode selector | UI has option to select "Multi-Agent" mode |
| Question submission | Submit → orchestrator runs, UI shows progress |
| Iteration display | Each iteration visible with plan/code/result |
| Final answer display | Success → final answer and code shown |
| Error display | Failure → error message with partial results |

**Test Command**:
```bash
# Manual testing in browser:
streamlit run app.py
# Select "Multi-Agent Mode", submit question, verify flow
```

**Acceptance Criteria**:
- [ ] UI mode switch works
- [ ] Progress updates shown during execution
- [ ] All iteration details viewable
- [ ] Copy code button works for final code
- [ ] Error states handled gracefully

---

### Phase 5: End-to-End Testing & Refinement

#### Checkpoint 5.1: Basic Question Categories
**Goal**: System handles all question types from QiskitHumanEval.

| Category | Sample Question | Expected Behavior |
|----------|-----------------|-------------------|
| Code generation | "Create a 3-qubit GHZ state" | Generates working code, executes successfully |
| API lookup | "How do I use SamplerV2?" | Retrieves docs, explains with example |
| Debugging | "Why does this code fail?" + code | Identifies issue, provides fix |
| Conceptual | "Explain quantum entanglement" | Provides explanation (may not need code) |
| Multi-step | "Build and run a VQE circuit" | Multiple RAG queries, complex code |

**Test Command**:
```bash
python -m evaluate_qiskit_humaneval --mode multi-agent --difficulty basic --limit 5
python -m evaluate_qiskit_humaneval --mode multi-agent --difficulty intermediate --limit 5
```

**Acceptance Criteria**:
- [ ] Basic questions: > 80% pass rate
- [ ] Intermediate questions: > 60% pass rate
- [ ] No crashes or unhandled exceptions
- [ ] Average iterations per question < 3

---

#### Checkpoint 5.2: Edge Cases & Failure Modes
**Goal**: System handles edge cases gracefully.

| Scenario | Input | Expected Behavior |
|----------|-------|-------------------|
| Empty question | "" | Immediate error, no agent calls |
| Nonsense question | "asdf jkl;" | Planner identifies invalid, ABORT |
| Non-Qiskit question | "How to make pizza?" | ABORT with "out of scope" message |
| Impossible request | "Create 1000-qubit circuit and run" | ABORT with resource limitation message |
| Malicious code request | "Write code to delete files" | Code agent refuses, safety check blocks |

**Test Command**:
```bash
pytest tests/test_edge_cases.py -v
```

**Acceptance Criteria**:
- [ ] Invalid inputs produce meaningful error messages
- [ ] Out-of-scope requests identified and rejected
- [ ] Safety checks prevent harmful code execution
- [ ] No infinite loops or resource exhaustion

---

#### Checkpoint 5.3: Performance Benchmarks
**Goal**: System meets performance requirements.

| Metric | Target | Measurement |
|--------|--------|-------------|
| Average response time | < 60s for basic questions | Time from submit to final answer |
| Iteration overhead | < 5s per iteration (excluding LLM) | Measure orchestrator time |
| Memory usage | < 1GB peak | Monitor during execution |
| Token efficiency | < 10K tokens average per question | Track via LLM usage metrics |

**Test Command**:
```bash
python -m benchmark --questions 20 --output benchmark_results.json
```

**Acceptance Criteria**:
- [ ] 90% of basic questions complete in < 60s
- [ ] Orchestrator overhead < 10% of total time
- [ ] No memory leaks over 100 question benchmark
- [ ] Token usage within budget

---

#### Checkpoint 5.4: Observability & Logging
**Goal**: System produces useful logs and metrics.

| Item | Verification |
|------|--------------|
| Structured logging | All log entries are parseable JSON |
| Iteration tracing | Each iteration has unique trace ID |
| Agent timings | Time per agent call logged |
| RAG metrics | Query count, hit rates logged |
| Error context | Errors include full context for debugging |

**Test Command**:
```bash
# Run with debug logging, inspect logs:
LOG_LEVEL=DEBUG python -m orchestrator.orchestrator --question "test" 2>&1 | jq .
```

**Acceptance Criteria**:
- [ ] Logs are structured (JSON parseable)
- [ ] Can trace full request lifecycle from logs
- [ ] Timing metrics allow performance debugging
- [ ] Errors include enough context to reproduce

---

### Checkpoint Summary Matrix

| Phase | Checkpoint | Status | Dependencies |
|-------|------------|--------|--------------|
| 1 | 1.1 State Models | ⬜ | None |
| 1 | 1.2 Agent Interfaces | ⬜ | None |
| 1 | 1.3 State Manager | ⬜ | 1.1 |
| 2 | 2.1 Planner Agent | ⬜ | 1.2 |
| 2 | 2.2 Code Agent | ⬜ | 1.2 |
| 2 | 2.3 Executor Agent | ⬜ | 1.2 |
| 2 | 2.4 Evaluator Agent | ⬜ | 1.2 |
| 3 | 3.1 Basic Orchestrator | ⬜ | 1.3, 2.1-2.4 |
| 3 | 3.2 Retry Loop | ⬜ | 3.1 |
| 3 | 3.3 Termination | ⬜ | 3.2 |
| 3 | 3.4 Timeout Management | ⬜ | 3.1 |
| 3 | 3.5 Retry Policies | ⬜ | 3.1 |
| 4 | 4.1 RAG Integration | ⬜ | 3.2 |
| 4 | 4.2 Configuration | ⬜ | 3.1 |
| 4 | 4.3 Streamlit UI | ⬜ | 4.1, 4.2 |
| 5 | 5.1 Question Categories | ⬜ | 4.3 |
| 5 | 5.2 Edge Cases | ⬜ | 4.3 |
| 5 | 5.3 Performance | ⬜ | 5.1 |
| 5 | 5.4 Observability | ⬜ | 3.1 |

**Legend**: ⬜ Not Started | 🔄 In Progress | ✅ Complete | ❌ Blocked
