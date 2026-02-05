# Physis Python Minimal

A single-file implementation of a digital evolution system where self-replicating programs evolve.

---

## Two Modes of Operation

| Mode | Description | Analogy |
|------|-------------|---------|
| **Static** | Fixed instruction set (Tierra-style) | Assembly language - direct hardware access |
| **Dynamic** | Self-defined instruction set | Organisms define their own "programming language" |

---

## Components

### 1. Gene (The Alphabet)

```python
class Gene(enum.IntEnum):
    # Structural markers (dynamic mode only)
    R = 0    # Register definition
    S = 1    # Stack definition  
    B = 2    # Begin instructions section
    I = 3    # Instruction definition marker
    SEP = 4  # Separator (end of definitions)
    
    # Atomic operations
    MOVE = 10; LOAD = 11; STORE = 12
    JUMP = 20; IFZERO = 21
    INC = 30; DEC = 31; ADD = 32; SUB = 33
    ALLOCATE = 40; DIVIDE = 41; READ_SIZE = 50
```

### 2. Phenotype (The Hardware)

**Static mode:** Fixed hardware (4 registers, 1 stack, all atomic ops available directly)

**Dynamic mode:** The genome defines its own hardware:
```
[R, R, R, R, B]     →  4 registers
[I, READ_SIZE, 1]   →  Instruction 0: read size into R1
[I, ALLOCATE, 1]    →  Instruction 1: allocate using R1
[I, LOAD, 2, 0, STORE, 0, 2, INC, 2]  →  Instruction 2: copy step (macro)
...
[SEP]               →  End of definitions
[0, 1, 2, ...]      →  Code (indices into instruction table)
```

### 3. Organism (The Virtual Machine)

| Attribute | Description |
|-----------|-------------|
| `genome` | NumPy array of integers |
| `ip` | Instruction pointer |
| `registers` | List of working memory values |
| `stacks` | List of deques (stack memory) |
| `child_buffer` | Buffer for offspring genome during replication |
| `phenotype` | Hardware spec (built from genome in dynamic mode) |
| `age` | Age in simulation steps |

### 4. World (The Environment)

```python
World.step():
    1. Each organism executes up to 30 instructions
    2. If DIVIDE succeeds → offspring created
    3. Offspring genome is mutated (point mutation + indels)
    4. New Organism created from mutated genome
    5. Death: organisms with age > 80 die
    6. Culling: random removal if population exceeds capacity
```

---

## The Self-Replication Loop

Both ancestors implement the same algorithm:

```
1. READ_SIZE   → Get own genome length into register
2. ALLOCATE    → Create child buffer of that size
3. LOAD/STORE  → Copy one byte from self to child (loop)
4. INC         → Increment copy pointer
5. SUB/IFZERO  → Check if done copying
6. JUMP        → Loop back if not done
7. DIVIDE      → Split off child as new organism
```

---

## Key Innovation: Dynamic Phenotype

In dynamic mode, **the genome encodes both the language AND the program**:

```
Genome = [Definitions Section] + [Code Section]
              ↓                       ↓
         Hardware spec          Actual program
         (registers,            (instruction indices)
          instruction set)
```

This means mutations can:
1. Change the **code** (like traditional ALife)
2. Change the **instruction set** itself
3. Change the **hardware** (number of registers/stacks)

---

## Mutation Model

| Type | Rate | Effect |
|------|------|--------|
| Point mutation | 1% per birth | Random gene replaced with random value (0-60) |
| Insertion | 0.25% per birth | Random gene inserted at random position |
| Deletion | 0.25% per birth | Random gene removed (if genome > 5) |

---

## Selection Pressure

- **Age limit:** Organisms die after 80 time steps
- **Capacity limit:** Random culling when population exceeds `pop_size`
- **Replication race:** Faster replicators leave more offspring

---

## Running the Experiment

```bash
python physis_python_minimal_py.py
```

Output compares static vs dynamic mode over 200 epochs:
- Population size
- Average genome length

---

## Research Question

> *"What evolves faster/better: organisms with a fixed instruction set (static), or organisms that can define their own instruction set (dynamic)?"*