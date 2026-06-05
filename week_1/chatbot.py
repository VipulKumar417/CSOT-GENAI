import os
import sys
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class ChatAgent:
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
        try:
            with open(self.history_filename, "w", encoding="utf-8") as f:
                json.dump(self.messages, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving conversation: {e}")

    def load_history(self):
        try:
            if os.path.exists(self.history_filename):
                with open(self.history_filename, "r", encoding="utf-8") as f:
                    loaded_messages = json.load(f)
                    if isinstance(loaded_messages, list) and len(loaded_messages) > 0:
                        self.messages = loaded_messages
                        return True
        except Exception as e:
            print(f"Error loading conversation: {e}")
        return False

    def _init_client(self):
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "Error: 'openai' package is not installed. "
                "Please install it using 'pip install openai'."
            )
            
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "Error: OPENROUTER_API_KEY environment variable is not set in .env file."
            )
            
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

    def reset_history(self):
        self.messages = [
            {"role": "system", "content": self.system_instruction}
        ]
        self.save_history()

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self.save_history()

    def get_raw_history(self):
        return self.messages

    def manage_context_length(self):
        non_system_messages = [m for m in self.messages if m["role"] != "system"]
        turns_count = len(non_system_messages) // 2
        
        if turns_count <= self.max_history_turns:
            return
            
        excess_turns = turns_count - self.max_history_turns
        
        if self.context_policy == "drop":
            print(f"\n[Context Manager]: Exceeded max history limit ({self.max_history_turns} turns). "
                  f"Dropping the oldest {excess_turns} turn(s).")
            system_msg = [m for m in self.messages if m["role"] == "system"]
            remaining_msgs = non_system_messages[excess_turns * 2:]
            self.messages = system_msg + remaining_msgs
            self.save_history()
            
        elif self.context_policy == "summarize":
            print(f"\n[Context Manager]: Exceeded max history limit ({self.max_history_turns} turns). "
                  f"Summarizing the oldest {excess_turns} turn(s)...")
            self.compact_history(turns_to_compact=excess_turns)

    def compact_history(self, turns_to_compact: int = None):
        """
        Summarizes the oldest turns in the conversation.
        """
        non_system_messages = [m for m in self.messages if m["role"] != "system"]
        total_turns = len(non_system_messages) // 2
        
        if total_turns == 0:
            print("No message history to compact.")
            return
            
        if turns_to_compact is None:
            turns_to_compact = max(1, total_turns - 1)
            
        turns_to_compact = min(turns_to_compact, total_turns)
        
        messages_to_summarize = non_system_messages[:turns_to_compact * 2]
        remaining_messages = non_system_messages[turns_to_compact * 2:]
        
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
        
        print("Generating history summary...")
        try:
            summary_text = self._make_single_call(summary_prompt)
            
            system_msg = [m for m in self.messages if m["role"] == "system"]
            summary_message = {
                "role": "user", 
                "content": f"[System Note: Summary of previous turns: {summary_text}]"
            }
            
            self.messages = system_msg + [summary_message] + remaining_messages
            self.save_history()
            print(f"Compaction complete! Oldest {turns_to_compact} turns replaced by summary.")
        except Exception as e:
            print(f"Failed to compact history: {e}. Keeping history as-is.")

    def _make_single_call(self, prompt: str) -> str:
        """Helper to make a single, stateless API call for summarization."""
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
    banner = """
======================================================================
                  CSOT 2026 AGENTIC TRACK: WEEK 1
     OpenRouter API Chatbot (Stateless Client & Manual State)
======================================================================
Commands:
  /exit or /quit   - End the chat session
  /reset           - Wipe conversation history (triggers context loss)
  /tokens          - Display accumulated input/output token usage
  /compact         - Manually compact/summarize the message history
  /history         - Print raw, manually managed chat history structure
  /policy          - Toggle context overflow policy (Drop vs Summarize)
----------------------------------------------------------------------
"""
    print(banner)


def choose_configuration():
    config_filename = "chat_config.json"
    saved_model = "google/gemini-2.5-flash:free"
    saved_policy = "summarize"
    saved_max_turns = 5
    has_saved_config = False
    
    # Friendly name mapping to OpenRouter model IDs
    model_mapping = {
        "1": ("Gemini 2.5 Flash", "google/gemini-2.5-flash:free"),
        "2": ("Llama 3.2 3B", "meta-llama/llama-3.2-3b-instruct:free"),
        "3": ("Qwen 2.5 Coder 32B", "qwen/qwen-2.5-coder-32b-instruct:free"),
        "4": ("DeepSeek R1 Distill Llama 8B", "deepseek/deepseek-r1-distill-llama-8b:free"),
    }
    
    if os.path.exists(config_filename):
        try:
            with open(config_filename, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            saved_model = config_data.get("model", "google/gemini-2.5-flash:free")
            saved_policy = config_data.get("policy", "summarize")
            saved_max_turns = config_data.get("max_turns", 5)
            has_saved_config = True
        except Exception:
            pass
            
    if has_saved_config:
        # Resolve friendly name for display
        display_model = saved_model
        for k, v in model_mapping.items():
            if v[1] == saved_model:
                display_model = f"{v[0]} ({saved_model})"
                break
        print("Saved Configuration Found:")
        print(f"  Model:  {display_model}")
        print(f"  Policy: {saved_policy}")
        print(f"  Turns:  {saved_max_turns}")
        use_last = input("Use last configuration? (y/n) [y]: ").strip().lower()
        if use_last != 'n':
            return saved_model, saved_policy, saved_max_turns
            
    print("\nStep 1: Select OpenRouter Model")
    for key, val in model_mapping.items():
        print(f"  {key}) {val[0]}")
    print("  5) Custom model ID")
    
    model_choice = input("Select model [1-5, default 1]: ").strip()
    if not model_choice:
        model_choice = "1"
        
    if model_choice in model_mapping:
        model = model_mapping[model_choice][1]
    elif model_choice == "5":
        model = input("Enter custom OpenRouter model ID: ").strip()
        if not model:
            model = "google/gemini-2.5-flash:free"
    else:
        model = "google/gemini-2.5-flash:free"
    
    print("\nStep 2: Choose Context Management Policy")
    print("1) Drop Oldest Turns (Removes old user-assistant pairs)")
    print("2) Summarize / Compact (Condenses oldest turns into single system note)")
    policy_choice = input("Select policy [1-2]: ").strip()
    policy = "summarize" if policy_choice == "2" else "drop"
    
    print("\nStep 3: Enter Max History Turns")
    turns_input = input("Enter max number of turns to keep (default '5'): ").strip()
    try:
        max_turns = int(turns_input) if turns_input else 5
    except ValueError:
        max_turns = 5
        
    try:
        with open(config_filename, "w", encoding="utf-8") as f:
            json.dump({"model": model, "policy": policy, "max_turns": max_turns}, f, indent=2)
    except Exception:
        pass
        
    return model, policy, max_turns


def run_chatbot():
    print_banner()
    
    try:
        model, policy, max_turns = choose_configuration()
        
        print(f"\nInitializing ChatAgent targeting OpenRouter using {model}...")
        print(f"Context overflow policy: {policy} | Max turns limit: {max_turns} turns.\n")
        
        agent = ChatAgent(
            model=model,
            context_policy=policy,
            max_history_turns=max_turns
        )
        
        if os.path.exists(agent.history_filename):
            try:
                with open(agent.history_filename, "r", encoding="utf-8") as f:
                    temp_msgs = json.load(f)
                if isinstance(temp_msgs, list) and len(temp_msgs) > 1:
                    resume = input("Found existing chat history. Do you want to resume? (y/n) [y]: ").strip().lower()
                    if resume != 'n':
                        if agent.load_history():
                            print("Previous conversation history loaded successfully!")
            except Exception:
                pass
                
        print("Success! Agent initialized. Start typing your prompts.")
        print("----------------------------------------------------------------------")
        
    except Exception as e:
        print(f"\nInitialization Failed: {e}")
        print("Please check your .env settings and try again.")
        return

    while True:
        try:
            user_input = input("\n[YOU] > ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["/exit", "/quit", "exit", "quit"]:
                print("\nShutting down chat session. Goodbye!")
                break
                
            elif user_input.lower() == "/reset":
                agent.reset_history()
                print("\n[System] Conversation history has been wiped. Context loss triggered!")
                continue
                
            elif user_input.lower() == "/tokens":
                print("\n[Metrics] Accumulated Token Usage:")
                print(f"  Input Tokens:  {agent.total_input_tokens}")
                print(f"  Output Tokens: {agent.total_output_tokens}")
                print(f"  Total Tokens:  {agent.total_input_tokens + agent.total_output_tokens}")
                continue
                
            elif user_input.lower() == "/compact":
                agent.compact_history()
                continue
                
            elif user_input.lower() == "/policy":
                new_policy = "summarize" if agent.context_policy == "drop" else "drop"
                agent.context_policy = new_policy
                print(f"\n[System] Context policy toggled to: {new_policy.upper()}")
                continue
                
            elif user_input.lower() == "/history":
                print("\n=== Raw Manually Managed Messages List ===")
                for idx, m in enumerate(agent.get_raw_history()):
                    print(f"Index {idx} | Role: {m['role']}")
                    print(f"Content: {m['content']}")
                    print("-" * 40)
                continue
                
            agent.add_message("user", user_input)
            
            print("[MODEL] > ", end="", flush=True)
            for chunk in agent.generate_reply_stream():
                print(chunk, end="", flush=True)
            print()
            
        except KeyboardInterrupt:
            print("\n\nSession interrupted by keyboard. Goodbye!")
            break
        except Exception as e:
            print(f"\nError generating response: {e}")


if __name__ == "__main__":
    run_chatbot()
