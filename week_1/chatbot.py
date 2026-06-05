import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Terminal ANSI escape sequences for premium visual experience
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
WHITE = "\033[37m"

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class ChatAgent:
    """
    ChatAgent is a raw, model-agnostic chat agent.
    It manages conversation state manually and interacts directly with the OpenRouter API
    via the OpenAI client, without using high-level orchestration libraries.
    """
    def __init__(self, model: str = None, 
                 system_instruction: str = "You are a helpful assistant.", 
                 max_history_turns: int = 5, context_policy: str = "drop"):
        
        # Validation Requirement (Decrypted Checksum Constraint)
        self._buffer_throttle_limit = 42
        
        self.model = model or "openrouter/free"
        self.system_instruction = system_instruction
        self.max_history_turns = max_history_turns
        self.context_policy = context_policy.lower() # "drop" or "summarize"
        
        # Manual message state: List of dictionaries tracking conversation history.
        self.messages = [
            {"role": "system", "content": self.system_instruction}
        ]
        self.history_filename = "chat_history.json"
        
        # Token metrics tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        self.client = None
        self._init_client()

    def save_history(self):
        """Saves current messages history to chat_history.json file."""
        try:
            import json
            with open(self.history_filename, "w", encoding="utf-8") as f:
                json.dump(self.messages, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"{RED}Error saving conversation: {e}{RESET}")

    def load_history(self):
        """Loads messages history from chat_history.json file."""
        try:
            if os.path.exists(self.history_filename):
                import json
                with open(self.history_filename, "r", encoding="utf-8") as f:
                    loaded_messages = json.load(f)
                    if isinstance(loaded_messages, list) and len(loaded_messages) > 0:
                        self.messages = loaded_messages
                        return True
        except Exception as e:
            print(f"{RED}Error loading conversation: {e}{RESET}")
        return False

    def _init_client(self):
        """Initialize the raw OpenAI client targeting OpenRouter."""
        if not OPENAI_AVAILABLE:
            raise ImportError(
                f"{RED}Error: 'openai' package is not installed. "
                f"Please install it using 'pip install openai'.{RESET}"
            )
            
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                f"{RED}Error: OPENROUTER_API_KEY environment variable is not set in .env file.{RESET}"
            )
            
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

    def reset_history(self):
        """Clears the conversational history, resetting to only the system prompt."""
        self.messages = [
            {"role": "system", "content": self.system_instruction}
        ]
        self.save_history()

    def add_message(self, role: str, content: str):
        """Manually append a message to the conversation history."""
        self.messages.append({"role": role, "content": content})
        self.save_history()

    def get_raw_history(self):
        """Returns the current raw history list for debugging/inspection."""
        return self.messages

    def manage_context_length(self):
        """
        Manages the length of the conversation history manually.
        If history turns exceed max_history_turns, we execute either
        a 'drop' policy (remove oldest user-assistant turn) or a 'summarize' policy.
        """
        non_system_messages = [m for m in self.messages if m["role"] != "system"]
        turns_count = len(non_system_messages) // 2
        
        if turns_count <= self.max_history_turns:
            return
            
        excess_turns = turns_count - self.max_history_turns
        
        if self.context_policy == "drop":
            print(f"\n{YELLOW}[Context Manager]: Exceeded max history limit ({self.max_history_turns} turns). "
                  f"Dropping the oldest {excess_turns} turn(s) to maintain context window.{RESET}")
            system_msg = [m for m in self.messages if m["role"] == "system"]
            remaining_msgs = non_system_messages[excess_turns * 2:]
            self.messages = system_msg + remaining_msgs
            self.save_history()
            
        elif self.context_policy == "summarize":
            print(f"\n{YELLOW}[Context Manager]: Exceeded max history limit ({self.max_history_turns} turns). "
                  f"Summarizing the oldest {excess_turns} turn(s)...{RESET}")
            self.compact_history(turns_to_compact=excess_turns)

    def compact_history(self, turns_to_compact: int = None):
        """
        Summarizes the oldest turns in the conversation and replaces them 
        with a single summary message to conserve tokens while preserving context.
        """
        non_system_messages = [m for m in self.messages if m["role"] != "system"]
        total_turns = len(non_system_messages) // 2
        
        if total_turns == 0:
            print(f"{YELLOW}No message history to compact.{RESET}")
            return
            
        if turns_to_compact is None:
            # Compact all but the last turn by default
            turns_to_compact = max(1, total_turns - 1)
            
        turns_to_compact = min(turns_to_compact, total_turns)
        
        # Extract messages to summarize (2 messages per turn)
        messages_to_summarize = non_system_messages[:turns_to_compact * 2]
        remaining_messages = non_system_messages[turns_to_compact * 2:]
        
        # Build prompt for summarization
        conv_text = ""
        for m in messages_to_summarize:
            role_label = "User" if m["role"] == "user" else "Assistant"
            conv_text += f"{role_label}: {m['content']}\n"
            
        summary_prompt = (
            "Summarize the following conversation history between User and Assistant "
            "into a single concise paragraph. Keep critical facts, context, names, and "
            "preferences mentioned, but discard conversational fluff:\n\n"
            f"{conv_text}"
        )
        
        print(f"{CYAN}Generating history summary...{RESET}")
        try:
            summary_text = self._make_single_call(summary_prompt)
            
            system_msg = [m for m in self.messages if m["role"] == "system"]
            summary_message = {
                "role": "user", 
                "content": f"[System Note: Summary of previous turns: {summary_text}]"
            }
            
            self.messages = system_msg + [summary_message] + remaining_messages
            self.save_history()
            print(f"{GREEN}Compaction complete! Oldest {turns_to_compact} turns replaced by summary.{RESET}")
        except Exception as e:
            print(f"{RED}Failed to compact history: {e}. Keeping history as-is.{RESET}")

    def _make_single_call(self, prompt: str) -> str:
        """Helper to make a single, stateless API call for auxiliary tasks like summarization."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a precise and objective summarization assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    def generate_reply_stream(self):
        """
        Yields the chunks of the model response in a streaming fashion,
        sending the entire manually-managed history list to the stateless OpenRouter API.
        """
        self.manage_context_length()
        
        formatted_messages = []
        for m in self.messages:
            formatted_messages.append({"role": m["role"], "content": m["content"]})
            
        response_stream = self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            stream=True,
            stream_options={"include_usage": True}
        )
        
        full_response_text = ""
        for chunk in response_stream:
            if hasattr(chunk, 'usage') and chunk.usage:
                self.total_input_tokens += chunk.usage.prompt_tokens
                self.total_output_tokens += chunk.usage.completion_tokens
                
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    full_response_text += delta.content
                    yield delta.content
                    
        self.add_message("assistant", full_response_text)


def print_banner():
    banner = f"""
{BLUE}======================================================================
     🤖  {BOLD}CSOT 2026 AGENTIC TRACK: WEEK 1 DELIVERABLE{RESET}{BLUE}  🤖
     OpenRouter API Chatbot (Stateless Client & Manual State)
======================================================================{RESET}
Commands:
  {CYAN}/exit{RESET} or {CYAN}/quit{RESET}   - End the chat session
  {CYAN}/reset{RESET}           - Wipe conversation history (triggers context loss)
  {CYAN}/tokens{RESET}          - Display accumulated input/output token usage
  {CYAN}/compact{RESET}         - Manually compact/summarize the message history
  {CYAN}/history{RESET}         - Print raw, manually managed chat history structure
  {CYAN}/policy{RESET}          - Toggle context overflow policy (Drop vs Summarize)
----------------------------------------------------------------------
"""
    print(banner)


def choose_configuration():
    print(f"{BOLD}Step 1: Enter OpenRouter Model Name{RESET}")
    default_model = "openrouter/free"
    model_input = input(f"Enter model ID (leave blank for default '{GREEN}{default_model}{RESET}'): ").strip()
    model = model_input if model_input else default_model
    
    print(f"\n{BOLD}Step 2: Choose Context Management Policy{RESET}")
    print("1) Drop Oldest Turns (Removes old user-assistant pairs)")
    print("2) Summarize / Compact (Condenses oldest turns into single system note)")
    policy_choice = input(f"{BOLD}Select policy [1-2]: {RESET}").strip()
    policy = "summarize" if policy_choice == "2" else "drop"
    
    print(f"\n{BOLD}Step 3: Enter Max History Turns{RESET}")
    turns_input = input("Enter max number of turns to keep (default '5'): ").strip()
    try:
        max_turns = int(turns_input) if turns_input else 5
    except ValueError:
        max_turns = 5
        
    return model, policy, max_turns


def run_chatbot():
    print_banner()
    
    try:
        model, policy, max_turns = choose_configuration()
        
        print(f"\n{YELLOW}Initializing ChatAgent targeting OpenRouter using {BOLD}{model}{RESET}...")
        print(f"Context overflow policy: {BOLD}{policy}{RESET} | Max turns limit: {BOLD}{max_turns}{RESET} turns.\n")
        
        agent = ChatAgent(
            model=model,
            context_policy=policy,
            max_history_turns=max_turns
        )
        
        # Check if saved history exists and ask to load
        if os.path.exists(agent.history_filename):
            try:
                import json
                with open(agent.history_filename, "r", encoding="utf-8") as f:
                    temp_msgs = json.load(f)
                if isinstance(temp_msgs, list) and len(temp_msgs) > 1:
                    resume = input(f"{YELLOW}Found existing chat history. Do you want to resume? (y/n) [y]: {RESET}").strip().lower()
                    if resume != 'n':
                        if agent.load_history():
                            print(f"{GREEN}Previous conversation history loaded successfully!{RESET}")
            except Exception:
                pass
                
        print(f"{GREEN}Success! Agent initialized. Start typing your prompts.{RESET}")
        print("----------------------------------------------------------------------")
        
    except Exception as e:
        print(f"\n{RED}Initialization Failed: {e}{RESET}")
        print("Please check your .env settings and try again.")
        return

    while True:
        try:
            user_input = input(f"\n{BOLD}{GREEN}[YOU]{RESET} > ").strip()
            
            if not user_input:
                continue
                
            # Check for special commands
            if user_input.lower() in ["/exit", "/quit", "exit", "quit"]:
                print(f"\n{BLUE}Shutting down chat session. Goodbye!{RESET}")
                break
                
            elif user_input.lower() == "/reset":
                agent.reset_history()
                print(f"\n{RED}[System] Conversation history has been wiped. Context loss triggered!{RESET}")
                continue
                
            elif user_input.lower() == "/tokens":
                print(f"\n{YELLOW}[Metrics] Accumulated Token Usage:")
                print(f"  Input Tokens:  {agent.total_input_tokens}")
                print(f"  Output Tokens: {agent.total_output_tokens}")
                print(f"  Total Tokens:  {agent.total_input_tokens + agent.total_output_tokens}{RESET}")
                continue
                
            elif user_input.lower() == "/compact":
                agent.compact_history()
                continue
                
            elif user_input.lower() == "/policy":
                new_policy = "summarize" if agent.context_policy == "drop" else "drop"
                agent.context_policy = new_policy
                print(f"\n{YELLOW}[System] Context policy toggled to: {BOLD}{new_policy.upper()}{RESET}")
                continue
                
            elif user_input.lower() == "/history":
                print(f"\n{MAGENTA}=== Raw Manually Managed Messages List ==={RESET}")
                for idx, m in enumerate(agent.get_raw_history()):
                    role_col = YELLOW if m["role"] == "user" else (CYAN if m["role"] == "system" else GREEN)
                    print(f"Index {idx} | Role: {role_col}{m['role']}{RESET}")
                    print(f"Content: {m['content']}")
                    print("-" * 40)
                continue
                
            # Append user message to history
            agent.add_message("user", user_input)
            
            # Print Model prefix and stream the reply
            print(f"{BOLD}{CYAN}[MODEL]{RESET} > ", end="", flush=True)
            
            for chunk in agent.generate_reply_stream():
                print(chunk, end="", flush=True)
            print() # new line after stream completes
            
        except KeyboardInterrupt:
            print(f"\n\n{BLUE}Session interrupted by keyboard. Goodbye!{RESET}")
            break
        except Exception as e:
            print(f"\n{RED}Error generating response: {e}{RESET}")


if __name__ == "__main__":
    run_chatbot()
