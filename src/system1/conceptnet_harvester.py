import json
import urllib.request
import urllib.parse
import urllib.error
import argparse
import re
import sys

# Structured categories mapping terms to domains for copy-paste comments
CONCEPT_DOMAINS = {
    "Logic and Reasoning": [
        "negation statement", "logical conjunction", "logical disjunction", "implication rule",
        "conditional branching", "binary comparison", "equality check", "inequality comparison",
        "boolean algebra", "propositional logic", "predicate logic", "quantifier resolution",
        "logic", "reasoning", "boolean", "negation", "logical deduction", "deduction", "syllogism",
        "truth table", "formal logic", "predicate"
    ],
    "Arithmetic & Mathematics": [
        "arithmetic addition", "subtraction calculation", "multiplication operation", "division calculation",
        "modulo calculation", "exponentiation math", "absolute value", "logarithmic calculation",
        "algebraic equation", "matrix transformation", "vector dot product", "set intersection",
        "mathematics", "addition", "subtraction", "multiplication", "division", "modulo", "exponentiation",
        "logarithm", "equation", "matrix", "vector", "dot product", "intersection"
    ],
    "Programming Structures & Syntax": [
        "variable assignment", "constant declaration", "function definition", "class instantiation",
        "recursive backtracking", "recursion depth base-case", "iterative loop syntax", "while loop condition",
        "for loop iteration", "exception handling", "null pointer check", "string concatenation",
        "list comprehension", "regular expression matching", "comment notation", "type casting",
        "programming", "recursion", "variable", "constant", "function", "class", "backtracking",
        "loop", "while loop", "for loop", "exception", "null pointer", "concatenation", "comprehension",
        "typecast", "syntax"
    ],
    "Data Structures & Operations": [
        "array indexing", "hashmap lookup", "stack push pop", "queue enqueue dequeue",
        "binary tree traversal", "graph node search", "linked list node traversal", "sorting algorithm",
        "data structure", "array", "hashmap", "stack", "queue", "tree", "traversal", "graph",
        "linked list", "sorting"
    ],
    "Algorithmic Themes & Puzzles": [
        "greedy choice", "dynamic programming transition", "divide and conquer", "binary search path",
        "spatial puzzle solving", "matrix grid rotation", "pattern matching detection", "sequence alignment",
        "puzzle", "algorithm", "greedy", "dynamic programming", "divide & conquer", "binary search",
        "spatial puzzle", "grid rotation", "pattern matching", "alignment"
    ]
}

# Flattend list for default fallback usage
FALLBACK_CONCEPTS = [item for sublist in CONCEPT_DOMAINS.values() for item in sublist]

def clean_label(label):
    """Clean the text label by lowecasing, stripping, and keeping only alphanumeric, spaces, and hyphens."""
    label = label.lower().strip()
    label = re.sub(r'[^a-z0-9\s\-]', '', label)
    label = re.sub(r'\s+', ' ', label)
    return label

def is_valid_concept(label, max_words=3):
    """Filters out noisy, too short, too long, or non-English-like labels."""
    if not label or len(label) < 3:
        return False
    words = label.split()
    if len(words) > max_words:
        return False
    # Avoid purely numeric labels
    if label.isdigit():
        return False
    return True

def fetch_concepts_from_conceptnet(seed, limit=50, max_words=3):
    """Query ConceptNet for a single seed word and extract related concepts."""
    formatted_seed = seed.replace(' ', '_')
    url = f"http://api.conceptnet.io/c/en/{urllib.parse.quote(formatted_seed)}?limit={limit}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    extracted = set()
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            edges = res_data.get("edges", [])
            for edge in edges:
                rel = edge.get("rel", {}).get("@id", "")
                if "/r/Antonym" in rel or "/r/NotRelatedTo" in rel:
                    continue
                
                for node_key in ["start", "end"]:
                    node = edge.get(node_key, {})
                    lang = node.get("language", "")
                    label = node.get("label", "")
                    
                    if lang == "en":
                        cleaned = clean_label(label)
                        if is_valid_concept(cleaned, max_words) and cleaned != seed:
                            extracted.add(cleaned)
    except urllib.error.HTTPError as e:
        print(f"Warning: ConceptNet API returned HTTP {e.code} for seed '{seed}'. Using fallback mechanisms.", file=sys.stderr)
        raise e
    except Exception as e:
        print(f"Warning: Failed to fetch connections for seed '{seed}': {e}", file=sys.stderr)
        raise e
        
    return extracted

def main():
    parser = argparse.ArgumentParser(description="ConceptNet Domain Harvester for HybridCBM")
    parser.add_argument(
        "--seeds", 
        type=str, 
        default="logic,programming,mathematics,puzzle,algorithm,reasoning,recursion,boolean",
        help="Comma-separated seed terms to query"
    )
    parser.add_argument("--limit_per_seed", type=int, default=50, help="Max edge limit per seed")
    parser.add_argument("--max_words", type=int, default=3, help="Max words allowed in a harvested concept label")
    args = parser.parse_args()
    
    seed_list = [s.strip().lower() for s in args.seeds.split(",") if s.strip()]
    print(f"Starting harvest using seeds: {seed_list}", file=sys.stderr)
    
    all_harvested = set()
    api_failed = False
    
    for seed in seed_list:
        print(f"Harvesting relations for '{seed}'...", file=sys.stderr)
        try:
            harvested = fetch_concepts_from_conceptnet(seed, limit=args.limit_per_seed, max_words=args.max_words)
            all_harvested.update(harvested)
            print(f"Found {len(harvested)} concepts for '{seed}'.", file=sys.stderr)
        except Exception:
            api_failed = True
            break
            
    if api_failed or len(all_harvested) == 0:
        print("\n[API FALLBACK] ConceptNet API is currently rate-limited or offline. Initializing curated domain concepts...", file=sys.stderr)
        for item in FALLBACK_CONCEPTS:
            cleaned = clean_label(item)
            if is_valid_concept(cleaned, args.max_words):
                all_harvested.add(cleaned)
                
    sorted_concepts = sorted(list(all_harvested))
    print(f"\nSuccessfully harvested {len(sorted_concepts)} unique concepts!\n", file=sys.stderr)
    
    # Classify concepts into domains for commenting
    grouped_concepts = {domain: [] for domain in CONCEPT_DOMAINS.keys()}
    grouped_concepts["General & Unclassified harvested concepts"] = []
    
    for concept in sorted_concepts:
        assigned = False
        # 1. Exact match in domain definition
        for domain, items in CONCEPT_DOMAINS.items():
            if concept in items:
                grouped_concepts[domain].append(concept)
                assigned = True
                break
        if assigned:
            continue
            
        # 2. Substring matching for general categories
        for domain, items in CONCEPT_DOMAINS.items():
            for item in items:
                # Use longer words in items list as keyword guides
                if len(item) > 3 and item in concept:
                    grouped_concepts[domain].append(concept)
                    assigned = True
                    break
            if assigned:
                break
                
        if not assigned:
            grouped_concepts["General & Unclassified harvested concepts"].append(concept)
            
    # Format and output as a copy-pasteable python list with domain comments
    print("candidate_labels = [")
    for domain, items in grouped_concepts.items():
        if not items:
            continue
        print(f"    # {domain}")
        sorted_items = sorted(list(set(items)))
        for i in range(0, len(sorted_items), 4):
            chunk = sorted_items[i:i+4]
            formatted_chunk = ", ".join(f'"{item}"' for item in chunk)
            print(f"    {formatted_chunk},")
    print("]")

if __name__ == "__main__":
    main()
