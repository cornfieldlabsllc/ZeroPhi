from ZeroPhiMemory import ZeroPhiMemory

# Note: The ZeroPhiMemory.py file must be in the same directory as this script.

if __name__ == "__main__":
    
    # 1. Initialize the Zero Phi System
    # This system models memory persistence through geometric invariance (P_infinity)
    memory_system = ZeroPhiMemory(name="Zero-Phi-Model-Demo")
    
    # 2. Store memory items
    # The 'Geometric Position' (vk) is the structural value being stored, 
    # representing where the probability distribution P(v) stabilizes.
    
    # Store 'Zero' at a low violation count (v=10)
    # The system stabilizes P at a geometric peak around v=10.
    memory_system.store_memory("Zero", 10) 
    
    # Store 'Phi' at a high violation count (v=90)
    # The system stabilizes P at a geometric peak around v=90.
    memory_system.store_memory("Phi", 90)
    
    # 3. Read/Retrieve the memory items
    print("\n--- Retrieval Results ---")
    memory_system.read_memory("Zero")
    memory_system.read_memory("Phi")
    
    # 4. Show status
    print(f"\n--- System Status ---")
    print(f"Total Memory Sheets Stored: {memory_system.memory_count}")
