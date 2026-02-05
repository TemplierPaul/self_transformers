"""
Physis Minimal - Single-file digital evolution simulation.

Tracks genome composition metrics over evolution:
- Hardware length (registers/stacks section)
- Language length (instruction definitions)
- Code length (actual program)
- Number of unique instructions

Saves results to pickle file and optionally logs to Weights & Biases.
"""

import enum
import numpy as np
import random
import pickle
from collections import deque
from datetime import datetime
from pathlib import Path

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ==========================================
# 1. THE PHYSICS (Atomic Laws)
# ==========================================

class Gene(enum.IntEnum):
    # Structural (Only used in Dynamic Mode)
    R = 0; S = 1; B = 2; I = 3; SEP = 4

    # Atomic Operations (The "Assembler")
    # These are the fundamental laws of the universe
    MOVE = 10; LOAD = 11; STORE = 12
    JUMP = 20; IFZERO = 21
    INC = 30; DEC = 31; ADD = 32; SUB = 33
    ALLOCATE = 40; DIVIDE = 41; READ_SIZE = 50


# ==========================================
# 2. THE PHENOTYPE (The Machine)
# ==========================================

class Instruction:
    """A macro-instruction composed of atomic ops."""
    def __init__(self, ops=None, args=None):
        self.ops = ops if ops is not None else []
        self.args = args if args is not None else []


class Phenotype:
    """The hardware spec."""
    def __init__(self, static_mode=False):
        self.n_regs = 0
        self.n_stacks = 0
        self.instructions = []
        self.code_start = 0

        if static_mode:
            self._build_static_standard_library()

    def _build_static_standard_library(self):
        """
        For Static Mode: Creates a 1-to-1 mapping where every Atomic Op
        is directly available as an instruction.
        """
        self.n_regs = 4  # Fixed hardware
        self.n_stacks = 1

        for i in range(60):
            new_instr = Instruction()
            if i in [m.value for m in Gene]:
                arity = 1
                if i in [Gene.MOVE, Gene.LOAD, Gene.STORE, Gene.ADD, Gene.SUB]: arity = 2
                if i in [Gene.DIVIDE, Gene.READ_SIZE, Gene.JUMP, Gene.IFZERO]: arity = 1
                new_instr.ops.append(i)
            self.instructions.append(new_instr)


# ==========================================
# 3. THE ORGANISM (Virtual Machine)
# ==========================================

class Organism:
    def __init__(self, genome: np.ndarray, mode='dynamic'):
        self.genome = genome
        self.mode = mode
        self.ip = 0
        self.child_buffer = None
        self.age = 0
        self.registers = []
        self.stacks = []

        # Build Phenotype
        if self.mode == 'static':
            self.phenotype = Phenotype(static_mode=True)
            self.phenotype.code_start = 0
        else:
            self.phenotype = self._build_dynamic_phenotype()

        self._init_memory()
        self.ip = self.phenotype.code_start

    def _build_dynamic_phenotype(self) -> Phenotype:
        p = Phenotype()
        ptr = 0
        limit = len(self.genome)

        # A. Define Structure
        while ptr < limit:
            g = self.genome[ptr]
            if g == Gene.R: p.n_regs += 1
            elif g == Gene.S: p.n_stacks += 1
            elif g == Gene.B: ptr += 1; break
            elif g in [Gene.I, Gene.SEP]: break
            ptr += 1

        if p.n_regs == 0: p.n_regs = 1

        # B. Define Instructions
        while ptr < limit:
            if self.genome[ptr] == Gene.SEP:
                ptr += 1; break

            if self.genome[ptr] == Gene.I:
                new_instr = Instruction()
                ptr += 1
                while ptr < limit:
                    atom = self.genome[ptr]
                    if atom in [Gene.I, Gene.SEP]: break

                    # Determine arity for each atomic operation
                    if atom in [Gene.MOVE, Gene.LOAD, Gene.STORE, Gene.ADD, Gene.SUB]:
                        arity = 2
                    elif atom in [Gene.READ_SIZE, Gene.ALLOCATE, Gene.INC, Gene.DEC, Gene.JUMP, Gene.IFZERO]:
                        arity = 1
                    elif atom == Gene.DIVIDE:
                        arity = 0
                    else:
                        arity = 1  # default

                    args = []
                    ptr += 1
                    for _ in range(arity):
                        args.append(self.genome[ptr] if ptr < limit else 0)
                        ptr += 1

                    new_instr.ops.append(atom)
                    new_instr.args.append(args)

                p.instructions.append(new_instr)
            else:
                ptr += 1

        p.code_start = ptr
        return p

    def _init_memory(self):
        self.registers = [0] * self.phenotype.n_regs
        self.stacks = [deque(maxlen=16) for _ in range(self.phenotype.n_stacks)]

    def _get(self, idx):
        total = self.phenotype.n_regs + self.phenotype.n_stacks
        if total == 0: return 0
        target = idx % total
        if target < self.phenotype.n_regs: return self.registers[target]
        s_idx = target - self.phenotype.n_regs
        return self.stacks[s_idx][-1] if self.stacks[s_idx] else 0

    def _set(self, idx, val):
        total = self.phenotype.n_regs + self.phenotype.n_stacks
        if total == 0: return
        target = idx % total
        if target < self.phenotype.n_regs: self.registers[target] = val
        else:
            s_idx = target - self.phenotype.n_regs
            self.stacks[s_idx].append(val)

    def step(self):
        if not self.phenotype.instructions: return None

        # Wrap IP to code section if out of bounds
        code_len = len(self.genome) - self.phenotype.code_start
        if code_len <= 0: return None
        
        if self.ip < self.phenotype.code_start or self.ip >= len(self.genome):
            self.ip = self.phenotype.code_start

        op_code = self.genome[self.ip]
        self.ip += 1
        
        # Wrap after increment
        if self.ip >= len(self.genome):
            self.ip = self.phenotype.code_start

        if self.mode == 'static':
            instr_idx = op_code % len(self.phenotype.instructions)
            instr = self.phenotype.instructions[instr_idx]
        else:
            if len(self.phenotype.instructions) == 0: return None
            instr_idx = op_code % len(self.phenotype.instructions)
            instr = self.phenotype.instructions[instr_idx]

        ops_to_run = instr.ops
        baked_args = instr.args

        for i, atom in enumerate(ops_to_run):
            current_args = []

            if self.mode == 'dynamic':
                current_args = baked_args[i]
            else:
                arity = 2 if atom in [Gene.MOVE, Gene.LOAD, Gene.STORE, Gene.ADD, Gene.SUB] else 1
                if atom in [Gene.DIVIDE, Gene.READ_SIZE, Gene.JUMP, Gene.IFZERO]: 
                    arity = 1 if atom != Gene.DIVIDE and atom != Gene.READ_SIZE else 0

                for _ in range(arity):
                    if self.ip < len(self.genome):
                        current_args.append(self.genome[self.ip])
                        self.ip += 1
                    else:
                        current_args.append(0)

            # --- ATOMIC LOGIC ---
            if atom == Gene.READ_SIZE:
                dest = current_args[0] if current_args else 0
                self._set(dest, len(self.genome))

            elif atom == Gene.ALLOCATE:
                sz = self._get(current_args[0])
                if 0 < sz < 300: self.child_buffer = np.zeros(sz, dtype=int)

            elif atom == Gene.LOAD:
                addr = self._get(current_args[0])
                val = self.genome[addr] if 0 <= addr < len(self.genome) else 0
                self._set(current_args[1], val)

            elif atom == Gene.STORE:
                if self.child_buffer is not None:
                    addr = self._get(current_args[1])
                    if 0 <= addr < len(self.child_buffer):
                        self.child_buffer[addr] = self._get(current_args[0])

            elif atom == Gene.MOVE:
                self._set(current_args[1], self._get(current_args[0]))

            elif atom == Gene.INC:
                self._set(current_args[0], self._get(current_args[0]) + 1)

            elif atom == Gene.SUB:
                self._set(current_args[0], self._get(current_args[0]) - self._get(current_args[1]))

            elif atom == Gene.IFZERO:
                if self._get(current_args[0]) == 0:
                    self.ip += 1

            elif atom == Gene.JUMP:
                target = current_args[0]
                if self.mode == 'static':
                    self.ip = target % len(self.genome)
                else:
                    code_len = len(self.genome) - self.phenotype.code_start
                    if code_len > 0:
                        self.ip = self.phenotype.code_start + (target % code_len)

            elif atom == Gene.DIVIDE:
                if self.child_buffer is not None:
                    baby = Organism(self.child_buffer, mode=self.mode)
                    self.child_buffer = None
                    return baby

        return None


# ==========================================
# 4. ANCESTOR GENOMES
# ==========================================

def create_static_ancestor():
    """Standard 'Tierra-style' ancestor."""
    g = [
        Gene.READ_SIZE, 1,
        Gene.ALLOCATE, 1,
        Gene.LOAD, 2, 0,
        Gene.STORE, 0, 2,
        Gene.INC, 2,
        Gene.SUB, 3, 1, 2,
        Gene.IFZERO, 3,
        Gene.JUMP, 4,
        Gene.DIVIDE
    ]
    return np.array(g, dtype=int)


def create_dynamic_ancestor():
    """'Physis-style' ancestor with definitions + code.
    
    Uses 4 registers:
    - R0: temp for copy
    - R1: genome size
    - R2: copy index (0, 1, 2, ...)
    - R3: remaining = size - copied
    """
    g = []
    # Hardware: 4 registers
    g += [Gene.R, Gene.R, Gene.R, Gene.R, Gene.B]

    # Instructions (each starts with I marker)
    g += [Gene.I, Gene.READ_SIZE, 1]                              # I0: R1 = size
    g += [Gene.I, Gene.ALLOCATE, 1]                               # I1: allocate R1 bytes
    g += [Gene.I, Gene.LOAD, 2, 0, Gene.STORE, 0, 2, Gene.INC, 2] # I2: R0=genome[R2]; child[R2]=R0; R2++
    g += [Gene.I, Gene.MOVE, 1, 3, Gene.SUB, 3, 2]                # I3: R3=R1; R3=R3-R2 (remaining)
    g += [Gene.I, Gene.IFZERO, 3]                                 # I4: skip next if R3==0
    g += [Gene.I, Gene.JUMP, 2]                                   # I5: jump to I2 (loop)
    g += [Gene.I, Gene.DIVIDE]                                    # I6: divide (give birth)
    g += [Gene.SEP]

    # Code: call instructions 0,1,2,3,4,5,2,3,4,5,... until divide
    # Simple linear sequence: 0,1,2,3,4,5,6
    g += [0, 1, 2, 3, 4, 5, 6]

    return np.array(g, dtype=int)


# ==========================================
# 5. WORLD (Environment)
# ==========================================

class World:
    def __init__(self, mode='dynamic', pop_size=100, cpu_cycles=200, initial_pop=10):
        self.mode = mode
        self.pop_size = pop_size
        self.cpu_cycles = cpu_cycles
        ancestor = create_static_ancestor() if mode == 'static' else create_dynamic_ancestor()
        self.population = [Organism(ancestor.copy(), mode=mode) for _ in range(initial_pop)]
        
        # Stats tracking
        self.births_this_epoch = 0
        self.deaths_this_epoch = 0
        self.total_births = 0
        self.total_deaths = 0

    def mutate(self, genome):
        g = genome.copy()
        rate = 0.01
        if random.random() < rate:
            g[random.randint(0, len(g)-1)] = random.randint(0, 60)

        if random.random() < 0.005:
            idx = random.randint(0, len(g)-1)
            if random.random() < 0.5 and len(g) > 5:
                g = np.delete(g, idx)
            else:
                g = np.insert(g, idx, random.randint(0, 60))
        return g

    def step(self):
        new_babies = []
        for org in self.population:
            for _ in range(self.cpu_cycles):
                baby = org.step()
                if baby:
                    mutated_g = self.mutate(baby.genome)
                    viable_baby = Organism(mutated_g, mode=self.mode)
                    new_babies.append(viable_baby)
                    break
            org.age += 1

        # Track deaths (aged out)
        prev_count = len(self.population)
        survivors = [o for o in self.population if o.age < 80]
        aged_deaths = prev_count - len(survivors)
        
        # Track births
        self.births_this_epoch = len(new_babies)
        self.total_births += self.births_this_epoch
        
        self.population = survivors + new_babies

        # Track culled organisms (population cap)
        culled = 0
        if len(self.population) > self.pop_size:
            culled = len(self.population) - self.pop_size
            random.shuffle(self.population)
            self.population = self.population[:self.pop_size]
        
        self.deaths_this_epoch = aged_deaths + culled
        self.total_deaths += self.deaths_this_epoch

    def get_avg_len(self):
        if not self.population: return 0
        return np.mean([len(o.genome) for o in self.population])


# ==========================================
# 6. GENOME ANALYSIS
# ==========================================

def analyze_genome(genome: np.ndarray) -> dict:
    """Parse a dynamic-mode genome and return section lengths."""
    ptr = 0
    limit = len(genome)
    
    # A. Hardware section
    hardware_start = 0
    while ptr < limit:
        g = genome[ptr]
        if g == Gene.B:
            ptr += 1
            break
        elif g in [Gene.I, Gene.SEP]:
            break
        ptr += 1
    
    hardware_len = ptr - hardware_start
    if ptr > 0 and genome[ptr - 1] == Gene.B:
        hardware_len -= 1

    language_start = ptr
    
    # B. Instruction definitions
    num_instructions = 0
    while ptr < limit:
        if genome[ptr] == Gene.SEP:
            ptr += 1
            break
        if genome[ptr] == Gene.I:
            num_instructions += 1
        ptr += 1
    
    language_len = ptr - language_start
    if ptr > language_start and genome[ptr - 1] == Gene.SEP:
        language_len -= 1

    # C. Code section
    code_len = limit - ptr
    
    return {
        'hardware_len': hardware_len,
        'language_len': language_len,
        'code_len': code_len,
        'num_instructions': num_instructions,
        'total_len': len(genome)
    }


def get_population_stats(population: list, world: 'World' = None) -> dict:
    """Get average genome composition stats for a population."""
    if not population:
        return {
            'hardware_len': 0, 'language_len': 0, 'code_len': 0,
            'num_instructions': 0, 'total_len': 0, 'pop_size': 0,
            'births': 0, 'deaths': 0, 'total_births': 0, 'total_deaths': 0
        }
    
    stats = [analyze_genome(org.genome) for org in population]
    
    result = {
        'hardware_len': np.mean([s['hardware_len'] for s in stats]),
        'language_len': np.mean([s['language_len'] for s in stats]),
        'code_len': np.mean([s['code_len'] for s in stats]),
        'num_instructions': np.mean([s['num_instructions'] for s in stats]),
        'total_len': np.mean([s['total_len'] for s in stats]),
        'pop_size': len(population),
        'births': world.births_this_epoch if world else 0,
        'deaths': world.deaths_this_epoch if world else 0,
        'total_births': world.total_births if world else 0,
        'total_deaths': world.total_deaths if world else 0
    }
    return result


# ==========================================
# 7. EXPERIMENT RUNNER
# ==========================================

def run_tracking_experiment(epochs=500, pop_size=150, cpu_cycles=200, log_interval=10, 
                           use_wandb=False, wandb_project="physis-minimal", initial_pop=10,
                           trial_num=None):
    """Run experiment and track genome composition over time."""
    
    trial_str = f" (Trial {trial_num})" if trial_num is not None else ""
    print(f"=== GENOME COMPOSITION TRACKING{trial_str} ===")
    print(f"Running {epochs} epochs, initial pop {initial_pop}, max pop {pop_size}, CPU cycles {cpu_cycles}")
    print()
    
    if use_wandb:
        if not WANDB_AVAILABLE:
            print("WARNING: wandb not installed. Run: pip install wandb")
            use_wandb = False
        else:
            run_name = f"trial-{trial_num}" if trial_num is not None else None
            wandb.init(
                project=wandb_project,
                name=run_name,
                config={
                    "epochs": epochs,
                    "pop_size": pop_size,
                    "initial_pop": initial_pop,
                    "cpu_cycles": cpu_cycles,
                    "log_interval": log_interval,
                    "mode": "dynamic",
                    "trial_num": trial_num,
                }
            )
            print(f"Logging to wandb project: {wandb_project}")
    
    world = World(mode='dynamic', pop_size=pop_size, cpu_cycles=cpu_cycles, initial_pop=initial_pop)
    
    history = {
        'epoch': [],
        'pop_size': [],
        'hardware_len': [],
        'language_len': [],
        'code_len': [],
        'num_instructions': [],
        'total_len': [],
        'births': [],
        'deaths': [],
        'total_births': [],
        'total_deaths': []
    }
    
    metadata = {
        'epochs': epochs,
        'pop_size': pop_size,
        'initial_pop': initial_pop,
        'cpu_cycles': cpu_cycles,
        'log_interval': log_interval,
        'mode': 'dynamic',
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"{'Epoch':<8} {'Pop':<6} {'Births':<8} {'Deaths':<8} {'Hardware':<10} {'Language':<10} {'Code':<8} {'Total':<8}")
    print("-" * 90)
    
    for t in range(epochs):
        world.step()
        
        if t % log_interval == 0:
            stats = get_population_stats(world.population, world)
            
            history['epoch'].append(t)
            history['pop_size'].append(stats['pop_size'])
            history['hardware_len'].append(stats['hardware_len'])
            history['language_len'].append(stats['language_len'])
            history['code_len'].append(stats['code_len'])
            history['num_instructions'].append(stats['num_instructions'])
            history['total_len'].append(stats['total_len'])
            history['births'].append(stats['births'])
            history['deaths'].append(stats['deaths'])
            history['total_births'].append(stats['total_births'])
            history['total_deaths'].append(stats['total_deaths'])
            
            if use_wandb:
                wandb.log({
                    "epoch": t,
                    "population/size": stats['pop_size'],
                    "population/births": stats['births'],
                    "population/deaths": stats['deaths'],
                    "population/total_births": stats['total_births'],
                    "population/total_deaths": stats['total_deaths'],
                    "genome/hardware_len": stats['hardware_len'],
                    "genome/language_len": stats['language_len'],
                    "genome/code_len": stats['code_len'],
                    "genome/num_instructions": stats['num_instructions'],
                    "genome/total_len": stats['total_len'],
                    "ratios/hardware_pct": 100 * stats['hardware_len'] / stats['total_len'] if stats['total_len'] > 0 else 0,
                    "ratios/language_pct": 100 * stats['language_len'] / stats['total_len'] if stats['total_len'] > 0 else 0,
                    "ratios/code_pct": 100 * stats['code_len'] / stats['total_len'] if stats['total_len'] > 0 else 0,
                })
            
            print(f"{t:<8} {stats['pop_size']:<6} {stats['births']:<8} {stats['deaths']:<8} "
                  f"{stats['hardware_len']:<10.1f} {stats['language_len']:<10.1f} "
                  f"{stats['code_len']:<8.1f} {stats['total_len']:<8.1f}")
            
            if stats['pop_size'] == 0:
                print("\nPopulation extinct!")
                break
    
    return {'history': history, 'metadata': metadata}


def save_results(data: dict, output_dir: str = 'output', use_wandb: bool = False):
    """Save results to pickle file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = output_path / f'genome_composition_{timestamp}.pkl'
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\nResults saved to: {filename}")
    
    if use_wandb and WANDB_AVAILABLE:
        artifact = wandb.Artifact('genome_composition', type='dataset')
        artifact.add_file(str(filename))
        wandb.log_artifact(artifact)
        wandb.finish()
        print("wandb run finished.")
    
    return filename


# ==========================================
# 8. MAIN
# ==========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Physis Minimal - Digital Evolution Simulation')
    parser.add_argument('--epochs', type=int, default=50000, help='Number of epochs')
    parser.add_argument('--pop-size', type=int, default=1000, help='Max population size')
    parser.add_argument('--initial-pop', type=int, default=10, help='Initial population size')
    parser.add_argument('--cpu-cycles', type=int, default=2000, help='CPU cycles per organism per epoch')
    parser.add_argument('--log-interval', type=int, default=1, help='Logging interval')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--wandb', action='store_true', help='Log to Weights & Biases')
    parser.add_argument('--wandb-project', type=str, default='physis-minimal', help='W&B project name')
    parser.add_argument('--trials', type=int, default=1, help='Number of trials to run')
    
    args = parser.parse_args()
    
    for trial in range(args.trials):
        trial_num = trial + 1 if args.trials > 1 else None
        
        data = run_tracking_experiment(
            epochs=args.epochs,
            pop_size=args.pop_size,
            cpu_cycles=args.cpu_cycles,
            log_interval=args.log_interval,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project,
            initial_pop=args.initial_pop,
            trial_num=trial_num
        )
        
        save_results(data, args.output_dir, use_wandb=args.wandb)
