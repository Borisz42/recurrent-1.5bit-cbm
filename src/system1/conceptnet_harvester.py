import json
import urllib.request
import urllib.parse
import urllib.error
import argparse
import re
import sys
import time
import os

# Structured categories mapping terms to domains for copy-paste comments and fallbacks
CONCEPT_DOMAINS = {
    "Code Generation": [
        "code generation", "programming", "software development", "debugging", "regular expression",
        "string formatting", "variable declaration", "function definition", "syntax highlighting",
        "scripting", "variable assignment", "comment notation", "type casting", "loop iteration",
        "recursive call", "return output", "writing scripts", "bug tracking", "troubleshooting",
        "regex matching", "formatting code", "syntax formatting", "indentation syntax",
        "generating functions", "commenting code", "exception handling", "code refactoring"
    ],
    "Mathematical Reasoning": [
        "mathematical reasoning", "arithmetic calculation", "algebraic equation", "logical deduction",
        "binary comparison", "boolean logic", "matrix operation", "vector calculus", "probability theory",
        "counting", "addition", "subtraction", "multiplication", "division", "modulo", "exponentiation",
        "logic puzzles", "riddles logic", "calculating math", "multiplication division",
        "addition subtraction", "vector matrices", "counting elements", "discrete math",
        "statistical calculation", "modulo arithmetic", "recursive calculation"
    ],
    "Information Extraction": [
        "information extraction", "named entity recognition", "text parsing", "data structuring",
        "json parsing", "regex extraction", "database query", "schema mapping", "unstructured text",
        "entity linking", "ner extraction", "data parsing", "unstructured to structured",
        "json formatting", "database querying", "schema definition", "key value parsing",
        "xml parsing", "parsing tables", "metadata extraction"
    ],
    "Text Summarization": [
        "text summarization", "abstracting", "tldr generation", "bullet point summary", "paraphrasing",
        "text shortening", "executive summary", "synopsis generation", "condensation", "outline creation",
        "abstracting text", "summarizing documents", "extracting key points", "condensation text",
        "document outline"
    ],
    "Creative & Narrative Writing": [
        "creative narrative writing", "storytelling", "essay writing", "brainstorming ideas",
        "poetry generation", "dialogue writing", "character creation", "metaphor usage",
        "plot outline", "scriptwriting", "storytelling narrative", "fiction writing", "essay composing",
        "creative writing", "dialogue drafting", "metaphor creation", "character development",
        "creative brainstorming", "narrative descriptions"
    ],
    "Linguistic Translation": [
        "linguistic translation", "language conversion", "bilingual translation", "vocabulary lookup",
        "machine translation", "cross-lingual", "sentence translation", "idiomatic translation",
        "grammar correction", "dictionary definition", "translating languages", "vocabulary translation",
        "converting text language", "cross lingual translation", "converting text", "language conversion",
        "sentence translating", "grammar conversion"
    ],
    "Factual Recall & Q&A": [
        "factual recall qa", "question answering", "fact retrieval", "historical summary",
        "encyclopedia definition", "trivia answer", "knowledge retrieval", "information verification",
        "biographical detail", "concept explanation", "factual recall", "historical summaries",
        "dictionary definitions", "retrieving facts", "biographical recall", "trivia answering",
        "encyclopedic explanation", "verifying facts", "defining concepts"
    ],
    "Roleplay & Persona Simulation": [
        "roleplay persona simulation", "chatbot conversation", "stylistic constraint", "expert persona",
        "character dialogue", "conversational agent", "tone adjustment", "empathy simulation",
        "narrative roleplay", "fictional character", "roleplay conversation", "persona simulation",
        "chatbot agent", "stylistic constraints", "acting as expert", "acting as an expert",
        "tone simulation", "fictional roleplay", "expert advice simulation"
    ],
    "Linguistic Classification": [
        "linguistic classification", "sentiment analysis", "intent detection", "text categorization",
        "sorting list items", "topic classification", "spam filtering", "parts of speech tagging",
        "urgency detection", "language identification", "sorting lists", "text classification",
        "categorizing text", "spam detection", "parts of speech classification", "topic categorization",
        "urgency classification", "labeling sentences"
    ]
}

# Mapping seed terms to their parent domains to trace harvesting sources
SEED_TO_DOMAIN = {
    # Code Generation
    "code generation": "Code Generation",
    "writing scripts": "Code Generation",
    "debugging": "Code Generation",
    "regular expression": "Code Generation",
    "regex": "Code Generation",
    "formatting code": "Code Generation",
    "programming": "Code Generation",
    "software development": "Code Generation",
    "scripting": "Code Generation",
    
    # Mathematical Reasoning
    "mathematical reasoning": "Mathematical Reasoning",
    "logic puzzles": "Mathematical Reasoning",
    "arithmetic": "Mathematical Reasoning",
    "calculations": "Mathematical Reasoning",
    "algebraic equation": "Mathematical Reasoning",
    "logical deduction": "Mathematical Reasoning",
    "boolean logic": "Mathematical Reasoning",
    
    # Information Extraction
    "information extraction": "Information Extraction",
    "named entity recognition": "Information Extraction",
    "ner": "Information Extraction",
    "text parsing": "Information Extraction",
    "parsing": "Information Extraction",
    "unstructured to structured": "Information Extraction",
    "unstructured to structured text": "Information Extraction",
    "json parsing": "Information Extraction",
    
    # Text Summarization
    "text summarization": "Text Summarization",
    "summarization": "Text Summarization",
    "shortening": "Text Summarization",
    "abstracting": "Text Summarization",
    "tldr generation": "Text Summarization",
    "tldr": "Text Summarization",
    
    # Creative & Narrative Writing
    "creative narrative writing": "Creative & Narrative Writing",
    "creative writing": "Creative & Narrative Writing",
    "storytelling": "Creative & Narrative Writing",
    "essay writing": "Creative & Narrative Writing",
    "essays": "Creative & Narrative Writing",
    "brainstorming": "Creative & Narrative Writing",
    "brainstorming ideas": "Creative & Narrative Writing",
    
    # Linguistic Translation
    "linguistic translation": "Linguistic Translation",
    "translation": "Linguistic Translation",
    "converting text": "Linguistic Translation",
    "converting text between languages": "Linguistic Translation",
    "language conversion": "Linguistic Translation",
    
    # Factual Recall & Q&A
    "factual recall qa": "Factual Recall & Q&A",
    "factual recall": "Factual Recall & Q&A",
    "retrieving facts": "Factual Recall & Q&A",
    "historical summaries": "Factual Recall & Q&A",
    "definitions": "Factual Recall & Q&A",
    "question answering": "Factual Recall & Q&A",
    "fact retrieval": "Factual Recall & Q&A",
    
    # Roleplay & Persona Simulation
    "roleplay persona simulation": "Roleplay & Persona Simulation",
    "roleplay": "Roleplay & Persona Simulation",
    "persona simulation": "Roleplay & Persona Simulation",
    "chatbots": "Roleplay & Persona Simulation",
    "chatbot": "Roleplay & Persona Simulation",
    "stylistic constraints": "Roleplay & Persona Simulation",
    "acting as an expert": "Roleplay & Persona Simulation",
    "expert persona": "Roleplay & Persona Simulation",
    
    # Linguistic Classification
    "linguistic classification": "Linguistic Classification",
    "classification": "Linguistic Classification",
    "sentiment analysis": "Linguistic Classification",
    "intent detection": "Linguistic Classification",
    "sorting list items": "Linguistic Classification",
    "sorting list": "Linguistic Classification",
    "text classification": "Linguistic Classification"
}

# Flattened list for default fallback usage
FALLBACK_CONCEPTS = [item for sublist in CONCEPT_DOMAINS.values() for item in sublist]

def clean_label(label):
    """Clean the text label by lowercasing, stripping, and keeping only alphanumeric, spaces, and hyphens."""
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
        default=",".join(sorted(list(SEED_TO_DOMAIN.keys()))),
        help="Comma-separated seed terms to query"
    )
    parser.add_argument("--limit_per_seed", type=int, default=50, help="Max edge limit per seed")
    parser.add_argument("--max_words", type=int, default=3, help="Max words allowed in a harvested concept label")
    parser.add_argument("--output", type=str, default="", help="Path to save harvested candidate labels as a JSON file")
    args = parser.parse_args()
    
    seed_list = [s.strip().lower() for s in args.seeds.split(",") if s.strip()]
    print(f"Starting harvest using {len(seed_list)} seeds.", file=sys.stderr)
    
    concept_to_source_domains = {}
    api_failed = False
    
    for seed in seed_list:
        print(f"Harvesting relations for '{seed}'...", file=sys.stderr)
        domain = SEED_TO_DOMAIN.get(seed, "General & Unclassified harvested concepts")
        try:
            harvested = fetch_concepts_from_conceptnet(seed, limit=args.limit_per_seed, max_words=args.max_words)
            for c in harvested:
                if c not in concept_to_source_domains:
                    concept_to_source_domains[c] = set()
                concept_to_source_domains[c].add(domain)
            print(f"Found {len(harvested)} concepts for '{seed}'.", file=sys.stderr)
            time.sleep(0.5)  # Add rate-limiting delay
        except Exception:
            api_failed = True
            break
            
    if api_failed or len(concept_to_source_domains) == 0:
        print("\n[API FALLBACK] ConceptNet API is currently rate-limited or offline. Initializing curated domain concepts...", file=sys.stderr)
        concept_to_source_domains = {}
        for domain, items in CONCEPT_DOMAINS.items():
            for item in items:
                cleaned = clean_label(item)
                if is_valid_concept(cleaned, args.max_words):
                    if cleaned not in concept_to_source_domains:
                        concept_to_source_domains[cleaned] = set()
                    concept_to_source_domains[cleaned].add(domain)
                
    sorted_concepts = sorted(list(concept_to_source_domains.keys()))
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
                
        # 3. Source-domain tracing fallback
        if not assigned:
            source_domains = concept_to_source_domains.get(concept, set())
            valid_source_domains = [d for d in source_domains if d in CONCEPT_DOMAINS]
            if valid_source_domains:
                grouped_concepts[valid_source_domains[0]].append(concept)
                assigned = True
                
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
    
    # Save to output JSON file if requested
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(sorted_concepts, f, indent=4)
        print(f"Saved {len(sorted_concepts)} candidate labels to {args.output}", file=sys.stderr)

if __name__ == "__main__":
    main()

