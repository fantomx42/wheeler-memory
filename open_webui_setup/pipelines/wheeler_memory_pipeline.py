"""
title: Wheeler Memory Pipeline
author: tristan
date: 2024-03-25
version: 1.0
license: MIT
description: Episodic memory injection for Open WebUI using Wheeler Memory.
requirements: numpy scipy
"""

import os
import sys
from typing import List, Union, Generator, Iterator

# Add /app/wheeler_memory to python path so we can import it
# (The launch script will mount the local source here)
sys.path.append("/app/wheeler_memory")

try:
    from wheeler_memory import recall_memory
    print("Successfully imported wheeler_memory")
except ImportError as e:
    print(f"Failed to import wheeler_memory: {e}")
    recall_memory = None


class Pipeline:
    def __init__(self):
        self.name = "Wheeler Memory"
        # Settings
        self.top_k = 5
        self.alpha = 0.3
        self.min_similarity = 0.1
        self.max_context_length = 2000

    async def on_startup(self):
        print(f"on_startup: {self.name}")

    async def on_shutdown(self):
        print(f"on_shutdown: {self.name}")

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where the magic happens
        print(f"pipe: processing message: {user_message}")

        if not recall_memory:
            return f"Error: wheeler_memory module not loaded. Check logs."

        # 1. Recall memories
        # We use embedding + reconstruction for the best results
        try:
            results = recall_memory(
                user_message,
                top_k=self.top_k,
                use_embedding=True,
                reconstruct=True,
                reconstruct_alpha=self.alpha,
                # We assume the data dir is standard or mapped
                data_dir="/app/data/.wheeler_memory" 
            )
        except Exception as e:
            print(f"Error enabling recall: {e}")
            return f"Error during memory recall: {e}"

        if not results:
            print("No relevant memories found.")
            return body

        # 2. Format the context
        context_str = "\n[Wheeler Memory - Episodic Context]\n"
        for r in results:
            if r['effective_similarity'] < self.min_similarity:
                continue
            
            # Use reconstructed attractor's similarity if available, else standard
            sim = r.get('effective_similarity', 0.0)
            text = r['text']
            tier = r['temperature_tier'].upper()
            temp = r['temperature']
            
            # Format: [TIER temp] "text" (similarity)
            context_str += f"[{tier} {temp:.2f}] \"{text}\" (sim={sim:.2f})\n"

        context_str += "Use this context to inform your response. Cold memories are uncertain.\n"

        # 3. Inject into system prompt
        # We find the system message in 'messages' or prepend a new one
        
        # NOTE: Open WebUI pipelines can modify the 'messages' list in the body
        # or return a generator. To modify the context, we should update the body 
        # and pass it to the next handler, but here we act as a filter.
        
        # Actually, for a pure context injection, we modify the last user message 
        # or the system message.
        
        print(f"Injected {len(results)} memories.")
        
        # Strategy: Prepend to the last user message for immediate visibility
        # This is often more robust than trying to find/edit the system prompt
        # which might be hidden or protected.
        
        if messages:
            # Check for system message
            if messages[0]['role'] == 'system':
                messages[0]['content'] += f"\n\n{context_str}"
            else:
                # Insert new system message at start
                messages.insert(0, {"role": "system", "content": context_str})
        
        return body
